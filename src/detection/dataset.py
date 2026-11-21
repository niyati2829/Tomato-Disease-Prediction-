"""Pest Object Detection Dataset & Annotation Parsers (REQ-08).

Supported Annotation Formats:
1. YOLO format: `<class_id> <cx> <cy> <w> <h>` (normalized to [0.0, 1.0]).
2. Pascal VOC format: XML `<annotation>` trees with `<bndbox>` bounds.

Academic Integrity Note:
PlantVillage provides exclusively image-level classification labels and lacks bounding-box
pest annotations. This module provides verified parsers for external datasets (IP102, VOC)
and a transparent synthetic benchmark generator for pipeline testing.

Author: Crop Disease Detection Project Team
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.detection.boxes import clip_boxes_to_image, cxcywh_to_xyxy, validate_boxes


PEST_CLASSES = {
    0: "Tomato Whitefly",
    1: "Aphid Cluster",
    2: "Spider Mite",
}


def parse_yolo_annotation(
    txt_path: Path | str,
    image_size: Tuple[int, int],  # (H, W)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse YOLO txt annotation into pixel coordinates [xmin, ymin, xmax, ymax].

    Args:
        txt_path: Path to .txt file.
        image_size: Tuple of (H, W).

    Returns:
        Tuple of (boxes_tensor of shape (N, 4), labels_tensor of shape (N,)).
    """
    H, W = image_size
    path = Path(txt_path)
    if not path.exists():
        return torch.empty((0, 4), dtype=torch.float32), torch.empty(0, dtype=torch.long)

    boxes = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                c_id = int(parts[0])
                cx = float(parts[1]) * W
                cy = float(parts[2]) * H
                w = float(parts[3]) * W
                h = float(parts[4]) * H

                xmin = cx - 0.5 * w
                ymin = cy - 0.5 * h
                xmax = cx + 0.5 * w
                ymax = cy + 0.5 * h

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(c_id)

    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty(0, dtype=torch.long)

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    boxes_tensor = clip_boxes_to_image(boxes_tensor, image_size)

    return boxes_tensor, labels_tensor


def parse_voc_xml(
    xml_path: Path | str,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse Pascal VOC XML annotation file.

    Args:
        xml_path: Path to XML file.
        class_to_idx: Optional mapping from string class name to integer ID.

    Returns:
        Tuple of (boxes_tensor of shape (N, 4), labels_tensor of shape (N,)).
    """
    path = Path(xml_path)
    if not path.exists():
        return torch.empty((0, 4), dtype=torch.float32), torch.empty(0, dtype=torch.long)

    tree = ET.parse(path)
    root = tree.getroot()

    size_elem = root.find("size")
    if size_elem is not None:
        W = float(size_elem.find("width").text)
        H = float(size_elem.find("height").text)
    else:
        W, H = 256.0, 256.0

    boxes = []
    labels = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        c_id = class_to_idx.get(name, 0) if class_to_idx else 0

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(c_id)

    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty(0, dtype=torch.long)

    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    boxes_t = clip_boxes_to_image(boxes_t, (int(H), int(W)))

    return boxes_t, labels_t


class PestDetectionDataset(Dataset):
    """PyTorch Dataset returning (image_tensor, target_dict)."""

    def __init__(
        self,
        image_paths: List[Path | str],
        annotation_paths: List[Path | str],
        image_size: Tuple[int, int] = (256, 256),
        format: str = "yolo",
    ) -> None:
        self.image_paths = [Path(p) for p in image_paths]
        self.annotation_paths = [Path(p) for p in annotation_paths]
        self.image_size = image_size
        self.format = format.lower()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_p = self.image_paths[idx]
        ann_p = self.annotation_paths[idx]

        pil_img = Image.open(img_p).convert("RGB")
        orig_w, orig_h = pil_img.size

        # Resize to target
        pil_resized = pil_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        img_tensor = self.transform(pil_resized)

        # Parse boxes
        if self.format == "voc":
            boxes, labels = parse_voc_xml(ann_p)
        else:
            boxes, labels = parse_yolo_annotation(ann_p, image_size=self.image_size)

        # Scale coordinates if parsed from original size
        if boxes.numel() > 0 and (orig_w != self.image_size[1] or orig_h != self.image_size[0]):
            scale_x = self.image_size[1] / orig_w
            scale_y = self.image_size[0] / orig_h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            boxes = clip_boxes_to_image(boxes, self.image_size)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return img_tensor, target


def generate_synthetic_pest_benchmark(
    n_samples: int = 50,
    image_size: int = 256,
    out_dir: Path | str = "data/processed/synthetic_pest_detection",
    seed: int = 42,
) -> Dict[str, Tuple[List[Path], List[Path]]]:
    """Generate benchmark dataset with synthetic insect pests on foliage.

    Pest Classes:
        0: Tomato Whitefly (Tiny white/translucent triangular speck)
        1: Aphid Cluster (Small grouped yellowish/green ovals)
        2: Spider Mite (Minute reddish-orange dots with webbing)

    Returns:
        Dict with keys 'train' (30), 'val' (10), 'test' (10).
    """
    out_p = Path(out_dir)
    img_dir = out_p / "images"
    ann_dir = out_p / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    all_imgs = []
    all_anns = []

    for i in range(n_samples):
        # Canvas: leaf background
        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        # Background foliage green
        img[:, :] = [rng.randint(35, 60), rng.randint(110, 160), rng.randint(25, 45)]

        # Add leaf veins
        for _ in range(rng.randint(5, 10)):
            pt1 = (rng.randint(0, image_size), rng.randint(0, image_size))
            pt2 = (rng.randint(0, image_size), rng.randint(0, image_size))
            cv2.line(img, pt1, pt2, (80, 190, 65), 1)

        n_pests = rng.randint(1, 5)
        boxes_yolo = []

        for _ in range(n_pests):
            c_id = rng.randint(0, 3)
            cx = rng.randint(30, image_size - 30)
            cy = rng.randint(30, image_size - 30)

            if c_id == 0:
                # Whitefly: small white triangle
                bw, bh = rng.randint(16, 26), rng.randint(16, 26)
                pts = np.array([
                    [cx, cy - bh // 2],
                    [cx - bw // 2, cy + bh // 2],
                    [cx + bw // 2, cy + bh // 2],
                ])
                cv2.fillPoly(img, [pts], (240, 245, 250))
            elif c_id == 1:
                # Aphid cluster: 3-5 small yellowish-green ovals
                bw, bh = rng.randint(22, 34), rng.randint(22, 34)
                for _ in range(4):
                    ox, oy = rng.randint(-6, 6), rng.randint(-6, 6)
                    cv2.ellipse(img, (cx + ox, cy + oy), (4, 3), rng.randint(0, 180), 0, 360, (60, 220, 210), -1)
            else:
                # Spider mite: reddish dot with webbing
                bw, bh = rng.randint(14, 24), rng.randint(14, 24)
                cv2.circle(img, (cx, cy), 4, (30, 40, 220), -1)
                # Webbing fine line
                cv2.line(img, (cx - 6, cy - 6), (cx + 6, cy + 6), (200, 200, 200), 1)

            # Store in normalized YOLO format: class_id cx cy w h
            norm_cx = cx / image_size
            norm_cy = cy / image_size
            norm_w = bw / image_size
            norm_h = bh / image_size
            boxes_yolo.append(f"{c_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")

        img_file = img_dir / f"pest_sample_{i:04d}.png"
        ann_file = ann_dir / f"pest_sample_{i:04d}.txt"

        Image.fromarray(img).save(img_file)
        with open(ann_file, "w", encoding="utf-8") as f:
            f.write("\n".join(boxes_yolo) + "\n")

        all_imgs.append(img_file)
        all_anns.append(ann_file)

    splits = {
        "train": (all_imgs[:30], all_anns[:30]),
        "val": (all_imgs[30:40], all_anns[30:40]),
        "test": (all_imgs[40:], all_anns[40:]),
    }

    return splits

