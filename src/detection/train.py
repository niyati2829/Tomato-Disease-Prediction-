"""Training Engine for Lightweight Pest Object Detection (REQ-08).

Trains TinyYOLODetector on benchmark pest data, checkpointing best weights
to artifacts/checkpoints/detection/best_pest_detector.pt.

Author: Crop Disease Detection Project Team
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.detection.dataset import PestDetectionDataset, generate_synthetic_pest_benchmark
from src.detection.metrics import evaluate_detections
from src.detection.model import TinyYOLODetector
from src.utils.device import get_device


def collate_fn(batch):
    """Custom collator handling variable box counts per image."""
    images, targets = zip(*batch)
    return torch.stack(images, 0), list(targets)


def compute_yolo_loss(
    raw_preds: torch.Tensor,
    targets: List[Dict[str, torch.Tensor]],
    grid_size: int = 8,
    num_boxes: int = 2,
    num_classes: int = 3,
    lambda_coord: float = 5.0,
    lambda_noobj: float = 0.5,
) -> torch.Tensor:
    """Compute grid detection loss across bounding boxes, objectness, and class logits."""
    device = raw_preds.device
    B = len(targets)
    S = grid_size

    # Build ground truth target tensor: (B, S, S, num_boxes * 5 + num_classes)
    target_tensor = torch.zeros_like(raw_preds)
    obj_mask = torch.zeros((B, S, S, num_boxes), dtype=torch.bool, device=device)
    noobj_mask = torch.ones((B, S, S, num_boxes), dtype=torch.bool, device=device)

    for b in range(B):
        gt_boxes = targets[b]["boxes"]  # (N, 4) in [xmin, ymin, xmax, ymax]
        gt_labels = targets[b]["labels"]

        for box, label in zip(gt_boxes, gt_labels):
            xmin, ymin, xmax, ymax = box
            w = (xmax - xmin).item()
            h = (ymax - ymin).item()
            cx = (xmin + xmax).item() / 2.0
            cy = (ymin + ymax).item() / 2.0

            # Cell indices
            cell_x = min(int((cx / 256.0) * S), S - 1)
            cell_y = min(int((cy / 256.0) * S), S - 1)

            tx = (cx / 256.0) * S - cell_x
            ty = (cy / 256.0) * S - cell_y
            tw = w / 256.0
            th = h / 256.0

            # Assign to primary box anchor in cell
            target_tensor[b, cell_y, cell_x, 0] = tx
            target_tensor[b, cell_y, cell_x, 1] = ty
            target_tensor[b, cell_y, cell_x, 2] = tw
            target_tensor[b, cell_y, cell_x, 3] = th
            target_tensor[b, cell_y, cell_x, 4] = 1.0  # Objectness
            obj_mask[b, cell_y, cell_x, 0] = True
            noobj_mask[b, cell_y, cell_x, 0] = False

            # One-hot class target
            class_offset = num_boxes * 5
            target_tensor[b, cell_y, cell_x, class_offset + int(label.item())] = 1.0

    # 1. Coordinate Loss (apply sigmoid to match predict_boxes decoding)
    pred_box0 = torch.sigmoid(raw_preds[..., :4])
    target_box0 = target_tensor[..., :4]
    if obj_mask[..., 0].sum() > 0:
        loss_coord = lambda_coord * F.mse_loss(
            pred_box0[obj_mask[..., 0]],
            target_box0[obj_mask[..., 0]],
            reduction="sum",
        ) / max(1, B)
    else:
        loss_coord = torch.tensor(0.0, device=device)

    # 2. Objectness Confidence Loss
    pred_obj = raw_preds[..., 4]
    target_obj = target_tensor[..., 4]
    if obj_mask[..., 0].sum() > 0:
        loss_obj = 5.0 * F.binary_cross_entropy_with_logits(
            pred_obj[obj_mask[..., 0]],
            target_obj[obj_mask[..., 0]],
            reduction="mean",
        )
    else:
        loss_obj = torch.tensor(0.0, device=device)

    # 3. No-Object Confidence Loss
    loss_noobj = 1.0 * F.binary_cross_entropy_with_logits(
        pred_obj[noobj_mask[..., 0]],
        target_obj[noobj_mask[..., 0]],
        reduction="mean",
    )

    # 4. Classification Loss
    pred_cls = raw_preds[..., num_boxes * 5 :]
    target_cls = target_tensor[..., num_boxes * 5 :]
    has_obj = obj_mask[..., 0]
    if has_obj.sum() > 0:
        loss_cls = F.cross_entropy(
            pred_cls[has_obj],
            torch.argmax(target_cls[has_obj], dim=-1),
            reduction="mean",
        )
    else:
        loss_cls = torch.tensor(0.0, device=device)

    total_loss = loss_coord + loss_obj + loss_noobj + loss_cls
    return total_loss


def train_pest_detector(
    epochs: int = 25,
    batch_size: int = 8,
    lr: float = 2e-3,
    checkpoint_dir: Path | str = "artifacts/checkpoints/detection",
    history_file: Path | str = "reports/detection_training_history.json",
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Train TinyYOLODetector on benchmark pest data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_p = Path(checkpoint_dir)
    ckpt_p.mkdir(parents=True, exist_ok=True)
    best_ckpt_file = ckpt_p / "best_pest_detector.pt"

    device = get_device(preferred="mps")
    print(f"[Phase D] Training Pest Object Detector on device: {device}")

    splits = generate_synthetic_pest_benchmark(n_samples=50, image_size=256, seed=seed)
    train_imgs, train_anns = splits["train"]
    val_imgs, val_anns = splits["val"]

    train_ds = PestDetectionDataset(train_imgs, train_anns, image_size=(256, 256))
    val_ds = PestDetectionDataset(val_imgs, val_anns, image_size=(256, 256))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = TinyYOLODetector(in_channels=3, num_classes=3, grid_size=8, num_boxes_per_cell=1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "val_mAP_0.5": [],
    }

    best_val_map = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for images, targets in train_loader:
            images = images.to(device)
            targets_dev = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]

            optimizer.zero_grad()
            raw_preds = model(images)
            loss = compute_yolo_loss(raw_preds, targets_dev, grid_size=8, num_boxes=1, num_classes=3)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        # Validation evaluation
        model.eval()
        val_preds = []
        val_gts = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                preds = model.predict_boxes(images, conf_threshold=0.25, iou_threshold=0.5)
                val_preds.extend(preds)
                val_gts.extend(targets)

        val_metrics = evaluate_detections(val_preds, val_gts, num_classes=3)
        val_map = val_metrics["mAP_0.5"]

        history["train_loss"].append(avg_train_loss)
        history["val_mAP_0.5"].append(val_map)

        if val_map >= best_val_map:
            best_val_map = val_map
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mAP_0.5": val_map,
                "num_classes": 3,
                "grid_size": 8,
                "num_boxes_per_cell": 1,
            }, best_ckpt_file)
            saved_tag = " -> Checkpointed (*)"
        else:
            saved_tag = ""

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val mAP@0.5: {val_map:.4f}{saved_tag}")

    # If all epochs produce initial mAP, ensure checkpoint exists
    if not best_ckpt_file.exists():
        torch.save({
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "val_mAP_0.5": val_map,
            "num_classes": 3,
            "grid_size": 8,
        }, best_ckpt_file)

    hist_p = Path(history_file)
    hist_p.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_p, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Phase D] Best Pest Detector Checkpoint saved to: {best_ckpt_file}")
    return history


if __name__ == "__main__":
    train_pest_detector()
