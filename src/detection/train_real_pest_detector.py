"""Training Pipeline for Real Tomato Pest Detection (Tuta absoluta / Leafminer).

Trains TinyYOLODetector on real field bounding box annotations from:
data/pest_detection/real_tuta_absoluta/
Checkpoints best weights to:
artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt
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

from src.detection.metrics import evaluate_detections
from src.detection.model import TinyYOLODetector
from src.detection.real_pest_dataset import REAL_PEST_CLASSES, RealTomatoPestDataset
from src.detection.train import collate_fn, compute_yolo_loss
from src.utils.device import get_device


def train_real_pest_detector(
    epochs: int = 35,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_dir: Path | str = "artifacts/checkpoints/pest_detection_real",
    history_file: Path | str = "reports/pest_detection_real_training_history.json",
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Train TinyYOLODetector on real Tuta absoluta tomato pest dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_p = Path(checkpoint_dir)
    ckpt_p.mkdir(parents=True, exist_ok=True)
    best_ckpt_file = ckpt_p / "best_pest_detector_real.pt"

    device = get_device(preferred="mps")
    print(f"[Phase G] Training Real Tomato Pest Detector on device: {device}")

    train_ds = RealTomatoPestDataset(split="train", image_size=(256, 256))
    val_ds = RealTomatoPestDataset(split="valid", image_size=(256, 256))

    print(f"Loaded {len(train_ds)} training images, {len(val_ds)} validation images.")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    num_classes = len(REAL_PEST_CLASSES)  # 2: healthy, tuta_absoluta_damage
    grid_size = 8
    num_boxes = 1

    model = TinyYOLODetector(
        in_channels=3,
        num_classes=num_classes,
        grid_size=grid_size,
        num_boxes_per_cell=num_boxes,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_mAP_0.5": [],
        "lr": [],
    }

    best_val_map = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for images, targets in train_loader:
            images = images.to(device)
            targets_dev = [
                {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
                for t in targets
            ]

            optimizer.zero_grad()
            raw_preds = model(images)
            loss = compute_yolo_loss(
                raw_preds,
                targets_dev,
                grid_size=grid_size,
                num_boxes=num_boxes,
                num_classes=num_classes,
            )
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()
        avg_train_loss = float(np.mean(train_losses))
        current_lr = float(scheduler.get_last_lr()[0])

        # Validation evaluation
        model.eval()
        val_preds = []
        val_gts = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                preds = model.predict_boxes(images, conf_threshold=0.20, iou_threshold=0.5)
                val_preds.extend(preds)
                val_gts.extend(targets)

        val_metrics = evaluate_detections(val_preds, val_gts, num_classes=num_classes)
        val_map = float(val_metrics["mAP_0.5"])

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_mAP_0.5"].append(val_map)
        history["lr"].append(current_lr)

        saved_tag = ""
        if val_map > best_val_map or epoch == 1:
            best_val_map = val_map
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_mAP_0.5": val_map,
                    "num_classes": num_classes,
                    "grid_size": grid_size,
                    "num_boxes_per_cell": num_boxes,
                    "class_names": REAL_PEST_CLASSES,
                },
                best_ckpt_file,
            )
            saved_tag = " -> Checkpointed (*)"

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val mAP@0.5: {val_map:.4f} | "
            f"LR: {current_lr:.6f}{saved_tag}"
        )

    # Save history
    hist_p = Path(history_file)
    hist_p.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_p, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Phase G] Best Real Pest Detector saved to: {best_ckpt_file}")
    print(f"Best Validation mAP@0.5: {best_val_map:.4f}")
    return history


if __name__ == "__main__":
    train_real_pest_detector()
