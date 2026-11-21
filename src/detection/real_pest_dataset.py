"""Real Tomato Pest Detection Dataset (Tuta absoluta / Tomato Leafminer).

Loads genuine field bounding-box annotations from:
data/pest_detection/real_tuta_absoluta/{train, valid, test}
Reference: Shehu et al. (2025), Frontiers in Plant Science (DOI: 10.3389/fpls.2025.1524630).
License: CC BY 4.0.

Classes:
    0: healthy (healthy leaf region)
    1: tuta_absoluta_damage (tomato leafminer infestation / mine lesion)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.detection.boxes import clip_boxes_to_image

REAL_PEST_CLASSES = {
    0: "healthy",
    1: "tuta_absoluta_damage",
}


def parse_real_yolo_labels(
    label_path: Path | str,
    image_width: int,
    image_height: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse YOLO format annotations into absolute coordinates [xmin, ymin, xmax, ymax].

    Args:
        label_path: Path to YOLO label .txt file.
        image_width: Width of image in pixels.
        image_height: Height of image in pixels.

    Returns:
        boxes: Tensor of shape (N, 4) in [xmin, ymin, xmax, ymax].
        labels: Tensor of shape (N,) with class indices (0 or 1).
    """
    path = Path(label_path)
    if not path.exists():
        return torch.empty((0, 4), dtype=torch.float32), torch.empty(0, dtype=torch.long)

    boxes: List[List[float]] = []
    labels: List[int] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    c_id = int(parts[0])
                    cx = float(parts[1]) * image_width
                    cy = float(parts[2]) * image_height
                    w = float(parts[3]) * image_width
                    h = float(parts[4]) * image_height

                    xmin = cx - 0.5 * w
                    ymin = cy - 0.5 * h
                    xmax = cx + 0.5 * w
                    ymax = cy + 0.5 * h

                    # Ensure valid positive area
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(c_id)
                except ValueError:
                    continue

    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty(0, dtype=torch.long)

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    boxes_tensor = clip_boxes_to_image(boxes_tensor, (image_height, image_width))

    return boxes_tensor, labels_tensor


class RealTomatoPestDataset(Dataset):
    """PyTorch Dataset for real tomato pest detection (Tuta absoluta)."""

    def __init__(
        self,
        split: str = "train",
        dataset_root: Path | str = "data/pest_detection/real_tuta_absoluta",
        image_size: Tuple[int, int] = (256, 256),
    ) -> None:
        self.split = split
        self.root = Path(dataset_root)
        self.image_size = image_size  # (H, W)

        split_dir = self.root / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        # Collect paired image and label files
        self.samples: List[Tuple[Path, Path]] = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            self.samples.append((img_path, lbl_path))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_p, lbl_p = self.samples[idx]

        pil_img = Image.open(img_p).convert("RGB")
        orig_w, orig_h = pil_img.size

        # Parse ground-truth boxes in original coordinate space (640, 640)
        boxes, labels = parse_real_yolo_labels(lbl_p, image_width=orig_w, image_height=orig_h)

        # Scale bounding boxes to target model input size (image_size)
        target_h, target_w = self.image_size
        if boxes.numel() > 0 and (orig_w != target_w or orig_h != target_h):
            scale_x = target_w / float(orig_w)
            scale_y = target_h / float(orig_h)
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            boxes = clip_boxes_to_image(boxes, self.image_size)

        # Resize image
        pil_resized = pil_img.resize((target_w, target_h), Image.BILINEAR)
        img_tensor = self.transform(pil_resized)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([orig_h, orig_w]),
        }

        return img_tensor, target
