"""Training Engine for U-Net Leaf Lesion Semantic Segmentation (REQ-08).

Saves best checkpoint to artifacts/checkpoints/segmentation/best_unet.pt
and training metrics to reports/unet_training_history.json.

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
from torch.utils.data import DataLoader

from src.segmentation.dataset import LeafSegmentationDataset, generate_synthetic_lesion_benchmark
from src.segmentation.losses_metrics import CombinedLoss, calculate_iou, calculate_pixel_accuracy
from src.segmentation.unet import UNet
from src.utils.device import get_device


def train_unet_model(
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    checkpoint_dir: Path | str = "artifacts/checkpoints/segmentation",
    history_file: Path | str = "reports/unet_training_history.json",
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Train U-Net on the benchmark segmentation dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_p = Path(checkpoint_dir)
    ckpt_p.mkdir(parents=True, exist_ok=True)
    best_ckpt_file = ckpt_p / "best_unet.pt"

    device = get_device(preferred="mps")
    print(f"[Phase D] Training U-Net on device: {device}")

    # Generate benchmark dataset
    splits = generate_synthetic_lesion_benchmark(n_samples=60, image_size=128, seed=seed)
    train_imgs, train_masks = splits["train"]
    val_imgs, val_masks = splits["val"]

    train_ds = LeafSegmentationDataset(train_imgs, train_masks, image_size=(128, 128))
    val_ds = LeafSegmentationDataset(val_imgs, val_masks, image_size=(128, 128))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize U-Net (3 classes: Background, Healthy Lamina, Diseased Lesion)
    model = UNet(in_channels=3, num_classes=3, init_features=32)
    model.to(device)

    criterion = CombinedLoss(weight_ce=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_mIoU": [],
        "val_loss": [],
        "val_mIoU": [],
        "val_pixel_acc": [],
    }

    best_val_miou = -1.0

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_losses = []
        train_mious = []

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            miou, _ = calculate_iou(logits, masks, num_classes=3)
            train_mious.append(miou)

        avg_train_loss = float(np.mean(train_losses))
        avg_train_miou = float(np.mean(train_mious))

        # 2. Validation Phase
        model.eval()
        val_losses = []
        val_mious = []
        val_accs = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)

                val_losses.append(loss.item())
                miou, _ = calculate_iou(logits, masks, num_classes=3)
                val_mious.append(miou)
                acc = calculate_pixel_accuracy(logits, masks)
                val_accs.append(acc)

        avg_val_loss = float(np.mean(val_losses))
        avg_val_miou = float(np.mean(val_mious))
        avg_val_acc = float(np.mean(val_accs))

        history["train_loss"].append(avg_train_loss)
        history["train_mIoU"].append(avg_train_miou)
        history["val_loss"].append(avg_val_loss)
        history["val_mIoU"].append(avg_val_miou)
        history["val_pixel_acc"].append(avg_val_acc)

        # Checkpoint if validation mIoU improved
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mIoU": avg_val_miou,
                "val_pixel_acc": avg_val_acc,
                "num_classes": 3,
                "init_features": 32,
            }, best_ckpt_file)
            saved_tag = " -> Checkpointed (*)"
        else:
            saved_tag = ""

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f}, mIoU: {avg_train_miou:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, mIoU: {avg_val_miou:.4f}, Acc: {avg_val_acc*100:.2f}%"
            f"{saved_tag}"
        )

    # Save training history
    hist_p = Path(history_file)
    hist_p.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_p, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Phase D] Best U-Net Checkpoint saved to: {best_ckpt_file}")
    print(f"  Best Val mIoU: {best_val_miou:.4f}")
    return history


if __name__ == "__main__":
    train_unet_model()

