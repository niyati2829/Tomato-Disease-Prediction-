"""Object Detection Models: Lightweight Single-Stage YOLO & SSDLite (REQ-08).

Implements:
1. `TinyYOLODetector`: Scratch-built single-stage grid detector:
       Input: (B, 3, 256, 256) -> Feature Grid: S x S = 8 x 8
       Predictions per cell: B=2 bounding boxes [cx, cy, w, h, obj] + C class logits.
2. `create_ssdlite_detector`: Factory adapting torchvision's SSDLite MobileNetV3
       for standardized 3-class agricultural pest detection.

Author: Crop Disease Detection Project Team
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.detection as detection
from torchvision.models.detection.ssd import SSDClassificationHead

from src.detection.boxes import non_max_suppression


class TinyYOLODetector(nn.Module):
    """Lightweight single-stage grid object detector."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        grid_size: int = 8,
        num_boxes_per_cell: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.S = grid_size
        self.B = num_boxes_per_cell

        # 5 Convolutional Downsampling Blocks (256 -> 128 -> 64 -> 32 -> 16 -> 8)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 256 -> 128

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )

        # Output channels per cell: B * 5 (cx, cy, w, h, conf) + C (class logits)
        out_depth = self.B * 5 + self.num_classes
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, out_depth, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor of shape (Batch, 3, H, W).

        Returns:
            Tensor of shape (Batch, S, S, B*5 + C).
        """
        features = self.backbone(x)
        out = self.head(features)
        # Permute to (Batch, S, S, Depth)
        out = out.permute(0, 2, 3, 1)
        return out

    def predict_boxes(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode raw grid output into final NMS-filtered bounding box predictions."""
        self.eval()
        with torch.no_grad():
            raw = self.forward(x)  # (Batch, S, S, B*5 + C)

        batch_size = raw.shape[0]
        H_img, W_img = x.shape[2], x.shape[3]
        cell_size_x = W_img / float(self.S)
        cell_size_y = H_img / float(self.S)

        results = []

        for b in range(batch_size):
            sample_out = raw[b]
            class_logits = sample_out[..., self.B * 5 :]  # (S, S, C)
            class_probs = F.softmax(class_logits, dim=-1)

            candidate_boxes = []
            candidate_scores = []
            candidate_labels = []

            for i in range(self.S):
                for j in range(self.S):
                    c_prob, c_label = torch.max(class_probs[i, j], dim=0)

                    for box_idx in range(self.B):
                        offset = box_idx * 5
                        tx = torch.sigmoid(sample_out[i, j, offset + 0])
                        ty = torch.sigmoid(sample_out[i, j, offset + 1])
                        tw = torch.sigmoid(sample_out[i, j, offset + 2])
                        th = torch.sigmoid(sample_out[i, j, offset + 3])
                        obj = torch.sigmoid(sample_out[i, j, offset + 4])

                        confidence = float((obj * c_prob).item())
                        if confidence >= conf_threshold:
                            # Decode center to absolute coordinates
                            cx = (j + tx.item()) * cell_size_x
                            cy = (i + ty.item()) * cell_size_y
                            w = tw.item() * W_img
                            h = th.item() * H_img

                            xmin = max(0.0, cx - 0.5 * w)
                            ymin = max(0.0, cy - 0.5 * h)
                            xmax = min(float(W_img), cx + 0.5 * w)
                            ymax = min(float(H_img), cy + 0.5 * h)

                            if xmax > xmin and ymax > ymin:
                                candidate_boxes.append([xmin, ymin, xmax, ymax])
                                candidate_scores.append(confidence)
                                candidate_labels.append(int(c_label.item()))

            if candidate_boxes:
                boxes_t = torch.tensor(candidate_boxes, dtype=torch.float32)
                scores_t = torch.tensor(candidate_scores, dtype=torch.float32)
                labels_t = torch.tensor(candidate_labels, dtype=torch.long)

                keep_idx = non_max_suppression(boxes_t, scores_t, iou_threshold=iou_threshold)
                results.append({
                    "boxes": boxes_t[keep_idx],
                    "scores": scores_t[keep_idx],
                    "labels": labels_t[keep_idx],
                })
            else:
                results.append({
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "scores": torch.empty(0, dtype=torch.float32),
                    "labels": torch.empty(0, dtype=torch.long),
                })

        return results


def create_ssdlite_detector(num_classes: int = 3, pretrained_backbone: bool = True) -> nn.Module:
    """Create SSDLite MobileNetV3 Large object detector adapted for pest detection.

    Note: Class count includes background as class 0 in torchvision SSD (total = num_classes + 1).
    """
    model = detection.ssdlite320_mobilenet_v3_large(weights_backbone=detection.MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None)
    # Re-initialize classification head for num_classes + 1 (background)
    in_channels = [head.in_channels for head in model.head.classification_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes + 1)
    return model

