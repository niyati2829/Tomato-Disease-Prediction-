"""Multi-Task Frame Analysis Engine for Tomato Surveillance.

Performs:
1. Calibrated Tomato Disease Classification:
   - Frozen production ResNet18 (artifacts/checkpoints/best_resnet18_weighted.pt).
   - Temperature scaling (T* = 1.3021).
   - 10 tomato health/disease classes.
2. Real Tomato Pest Detection:
   - Checkpointed TinyYOLODetector (artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt).
   - Bounding boxes, confidence scores, and count of Tuta absoluta mine lesions.
3. Lesion Segmentation Status:
   - Explicitly records 'Unavailable (Blocked by Dataset Availability)' for academic integrity.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.calibration.temperature_scaling import ModelWithTemperature
from src.detection.model import TinyYOLODetector
from src.detection.real_pest_dataset import REAL_PEST_CLASSES
from src.models.transfer_learning import get_resnet18
from src.utils.device import get_device

TOMATO_CLASSES = {
    0: "Tomato Bacterial Spot",
    1: "Tomato Early Blight",
    2: "Tomato Late Blight",
    3: "Tomato Leaf Mold",
    4: "Tomato Septoria Leaf Spot",
    5: "Tomato Two-spotted Spider Mite",
    6: "Tomato Target Spot",
    7: "Tomato Yellow Leaf Curl Virus",
    8: "Tomato Mosaic Virus",
    9: "Tomato Healthy",
}


@dataclass
class FrameAnalysisResult:
    """Quantitative results container for single surveillance frame."""

    frame_idx: int
    timestamp_seconds: float
    disease_class_id: int
    disease_label: str
    disease_confidence: float
    disease_probabilities: Dict[str, float]
    pest_count: int
    pest_boxes: List[List[float]]  # [xmin, ymin, xmax, ymax] in native frame pixels
    pest_scores: List[float]
    pest_labels: List[str]
    segmentation_status: str = "Unavailable (Blocked by Dataset Availability)"
    pest_masks: Optional[List[Any]] = None
    pest_detector_type: str = "tinyyolo"

    @property
    def timestamp(self) -> float:
        """Alias for timestamp_seconds for backwards-compatibility."""
        return self.timestamp_seconds


class SurveillanceFrameAnalyzer:
    """High-throughput multi-task vision analyzer for surveillance frames."""

    def __init__(
        self,
        classifier_checkpoint: Path | str = "artifacts/checkpoints/best_resnet18_weighted.pt",
        pest_detector_checkpoint: Path | str = "artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt",
        temperature: float = 1.3021,
        pest_conf_threshold: float = 0.20,
        device: Optional[torch.device] = None,
        use_yolov8: bool = False,
        yolov8_checkpoint: Path | str = "artifacts/checkpoints/pest_detection_yolov8/best_yolov8n_seg.pt",
    ) -> None:
        self.device = device or get_device(preferred="cpu")
        self.pest_conf_threshold = pest_conf_threshold
        self.use_yolov8 = use_yolov8

        # 1. Initialize Calibrated Disease Classifier
        ckpt_cls = Path(classifier_checkpoint)
        if not ckpt_cls.exists():
            raise FileNotFoundError(f"Classifier checkpoint missing: {ckpt_cls}")

        base_classifier = get_resnet18(num_classes=10, pretrained=False)
        cls_state = torch.load(ckpt_cls, map_location=self.device, weights_only=False)
        base_classifier.load_state_dict(cls_state["model_state_dict"])
        self.classifier = ModelWithTemperature(base_classifier)
        self.classifier.set_temperature(temperature)
        self.classifier.to(self.device)
        self.classifier.eval()

        self.cls_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Initialize Pest Detector (YOLOv8n-Seg or TinyYOLO)
        self.yolov8_detector = None
        if self.use_yolov8:
            ckpt_yolo = Path(yolov8_checkpoint)
            if ckpt_yolo.exists():
                try:
                    from ultralytics import YOLO
                    self.yolov8_detector = YOLO(str(ckpt_yolo))
                except Exception:
                    self.yolov8_detector = None

        ckpt_pest = Path(pest_detector_checkpoint)
        self.pest_detector: Optional[TinyYOLODetector] = None
        if not self.use_yolov8 and ckpt_pest.exists():
            pest_state = torch.load(ckpt_pest, map_location=self.device, weights_only=False)
            num_classes = pest_state.get("num_classes", 2)
            grid_size = pest_state.get("grid_size", 8)
            num_boxes = pest_state.get("num_boxes_per_cell", 1)

            det = TinyYOLODetector(
                in_channels=3,
                num_classes=num_classes,
                grid_size=grid_size,
                num_boxes_per_cell=num_boxes,
            )
            det.load_state_dict(pest_state["model_state_dict"])
            det.to(self.device)
            det.eval()
            self.pest_detector = det

        self.pest_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def analyze_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int = 0,
        timestamp: float = 0.0,
    ) -> FrameAnalysisResult:
        """Run disease diagnosis and pest detection on raw BGR video frame."""
        H_orig, W_orig = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_rgb = Image.fromarray(frame_rgb)

        # 1. Disease Classification
        pil_cls = pil_rgb.resize((224, 224), Image.BILINEAR)
        t_cls = self.cls_transform(pil_cls).unsqueeze(0).to(self.device)

        with torch.no_grad():
            scaled_logits = self.classifier(t_cls)
            probs = F.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy()

        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        top_label = TOMATO_CLASSES.get(top_idx, f"Class {top_idx}")
        prob_dict = {
            TOMATO_CLASSES.get(i, f"Class {i}"): float(probs[i])
            for i in range(len(probs))
        }

        # 2. Pest Detection
        pest_boxes_native: List[List[float]] = []
        pest_scores: List[float] = []
        pest_labels: List[str] = []
        pest_masks: Optional[List[Any]] = None
        pest_count = 0
        detector_type = "yolov8n-seg" if self.use_yolov8 else "tinyyolo"

        if self.use_yolov8 and self.yolov8_detector is not None:
            results = self.yolov8_detector.predict(pil_rgb, conf=self.pest_conf_threshold, verbose=False)[0]
            if results.boxes is not None and len(results.boxes) > 0:
                boxes_xyxy = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                clss = results.boxes.cls.cpu().numpy()

                masks_xy = results.masks.xy if results.masks is not None else [None] * len(boxes_xyxy)
                pest_masks = []

                for b, s, c, m in zip(boxes_xyxy, confs, clss, masks_xy):
                    c_id = int(c)
                    is_tuta = (c_id == 1)
                    lbl = "Tuta absoluta Feeding Damage" if is_tuta else "Healthy Foliage"
                    pest_boxes_native.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                    pest_scores.append(float(s))
                    pest_labels.append(lbl)
                    if m is not None and len(m) > 0:
                        pest_masks.append(np.asarray(m, dtype=np.float32))
                    else:
                        pest_masks.append(None)

                    if is_tuta:
                        pest_count += 1
        elif self.pest_detector is not None:
            pil_pest = pil_rgb.resize((256, 256), Image.BILINEAR)
            t_pest = self.pest_transform(pil_pest).unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds = self.pest_detector.predict_boxes(
                    t_pest, conf_threshold=self.pest_conf_threshold, iou_threshold=0.5
                )[0]

            boxes_256 = preds["boxes"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()

            scale_x = W_orig / 256.0
            scale_y = H_orig / 256.0

            for b, s, l in zip(boxes_256, scores, labels):
                c_name = REAL_PEST_CLASSES.get(int(l), f"Class {l}")
                x1 = float(b[0] * scale_x)
                y1 = float(b[1] * scale_y)
                x2 = float(b[2] * scale_x)
                y2 = float(b[3] * scale_y)

                pest_boxes_native.append([x1, y1, x2, y2])
                pest_scores.append(float(s))
                pest_labels.append(c_name)

                # Count pest damage instances (class 1: tuta_absoluta_damage)
                if int(l) == 1:
                    pest_count += 1

        return FrameAnalysisResult(
            frame_idx=frame_idx,
            timestamp_seconds=timestamp,
            disease_class_id=top_idx,
            disease_label=top_label,
            disease_confidence=top_conf,
            disease_probabilities=prob_dict,
            pest_count=pest_count,
            pest_boxes=pest_boxes_native,
            pest_scores=pest_scores,
            pest_labels=pest_labels,
            segmentation_status="Unavailable (Blocked by Dataset Availability)",
            pest_masks=pest_masks,
            pest_detector_type=detector_type,
        )
