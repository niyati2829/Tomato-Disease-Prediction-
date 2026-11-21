"""Phase K: Training Engine for Real Tomato Lesion Segmentation U-Net.

Trains a class-agnostic binary lesion segmentation model using the cleaned
and verified PlantSeg tomato dataset prepared in Phase J.

Task:
- Foreground (1): Disease lesion tissue
- Background (0): Healthy foliar tissue and neutral backdrop

Disciplines:
- Strictly trains on data/processed/tomato_lesions_plantseg/splits/train.csv (459 images).
- Strictly validates on data/processed/tomato_lesions_plantseg/splits/val.csv (99 images).
- Holds out data/processed/tomato_lesions_plantseg/splits/test.csv (99 images) untouched for Phase L.
- Preserves the existing synthetic U-Net checkpoint (artifacts/checkpoints/segmentation/best_unet.pt).
- Saves real model checkpoint to artifacts/checkpoints/segmentation_real/best_unet_real.pt.
- Records metrics to reports/unet_real_training_history.json.

Author: Crop Disease Detection Project Team
Phase: Phase K
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.segmentation.losses_metrics import BinaryCombinedLoss, calculate_binary_metrics
from src.segmentation.plantseg_dataset import PlantSegTomatoLesionDataset
from src.segmentation.unet import UNet
from src.utils.device import get_device, get_device_info


def train_real_unet_model(
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    weight_bce: float = 0.5,
    patience: int = 8,
    init_features: int = 16,
    image_size: Tuple[int, int] = (256, 256),
    checkpoint_dir: Path | str = "artifacts/checkpoints/segmentation_real",
    history_file: Path | str = "reports/unet_real_training_history.json",
    config_file: Path | str = "reports/unet_real_training_config.json",
    seed: int = 42,
) -> Dict[str, any]:
    """Train real tomato lesion segmentation U-Net model on PlantSeg benchmark."""
    # Set reproducible random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ckpt_p = Path(checkpoint_dir)
    ckpt_p.mkdir(parents=True, exist_ok=True)
    best_ckpt_file = ckpt_p / "best_unet_real.pt"

    device = get_device(preferred="mps")
    device_info = get_device_info(preferred="mps")
    print("=" * 70)
    print("PHASE K: REAL TOMATO LESION SEGMENTATION U-NET TRAINING")
    print("=" * 70)
    print(f"Device: {device} ({device_info.get('device_name', 'Host')})")
    print(f"Platform: {device_info.get('platform')}")
    print(f"Random seed: {seed}")
    print(f"Resolution: {image_size[0]}x{image_size[1]}")
    print(f"Batch size: {batch_size} | Base features: {init_features} | Max epochs: {epochs}")
    print(f"Initial LR: {lr:.2e} | Weight decay: {weight_decay:.2e} | Early stop patience: {patience}")
    print("=" * 70)

    # 1. Load Data (Strictly train and validation splits only; test split untouched)
    train_ds = PlantSegTomatoLesionDataset(
        split="train",
        image_size=image_size,
        as_float_mask=True,
        preserve_aspect_ratio=True,
    )
    val_ds = PlantSegTomatoLesionDataset(
        split="val",
        image_size=image_size,
        as_float_mask=True,
        preserve_aspect_ratio=True,
    )

    print(f"Dataset Loaded: {len(train_ds)} train samples, {len(val_ds)} validation samples")
    print("NOTE: Holdout test split (99 samples) is strictly reserved for Phase L evaluation.\n")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # 2. Initialize Model
    model = UNet(
        in_channels=3,
        num_classes=1,  # 1 output channel for binary lesion segmentation
        init_features=init_features,
        bilinear=True,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net Architecture: 3 in -> {init_features} base features -> 1 out")
    print(f"Total Parameters: {total_params:,} (Trainable: {trainable_params:,})")

    # 3. Loss, Optimizer, and Learning Rate Scheduler
    criterion = BinaryCombinedLoss(weight_bce=weight_bce, smooth=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    # 4. Training Records
    history = {
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_pixel_acc": [],
        "learning_rate": [],
        "epoch_times_seconds": [],
    }

    best_val_dice = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    start_total_time = time.time()

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training Phase
        model.train()
        train_losses = []
        train_dices = []
        train_ious = []

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            with torch.no_grad():
                metrics = calculate_binary_metrics(logits, masks)
                train_dices.append(metrics["dice"])
                train_ious.append(metrics["iou"])

        avg_train_loss = float(np.mean(train_losses))
        avg_train_dice = float(np.mean(train_dices))
        avg_train_iou = float(np.mean(train_ious))

        # Validation Phase
        model.eval()
        val_losses = []
        val_dices = []
        val_ious = []
        val_accs = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)

                val_losses.append(loss.item())
                metrics = calculate_binary_metrics(logits, masks)
                val_dices.append(metrics["dice"])
                val_ious.append(metrics["iou"])
                val_accs.append(metrics["pixel_accuracy"])

        avg_val_loss = float(np.mean(val_losses))
        avg_val_dice = float(np.mean(val_dices))
        avg_val_iou = float(np.mean(val_ious))
        avg_val_acc = float(np.mean(val_accs))

        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(round(avg_train_loss, 4))
        history["train_dice"].append(round(avg_train_dice, 4))
        history["train_iou"].append(round(avg_train_iou, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_dice"].append(round(avg_val_dice, 4))
        history["val_iou"].append(round(avg_val_iou, 4))
        history["val_pixel_acc"].append(round(avg_val_acc, 4))
        history["learning_rate"].append(current_lr)
        history["epoch_times_seconds"].append(round(epoch_duration, 2))

        # Learning rate scheduler step
        scheduler.step(avg_val_dice)

        # Early Stopping & Checkpointing based on validation Dice score
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": avg_val_dice,
                "val_iou": avg_val_iou,
                "val_pixel_acc": avg_val_acc,
                "val_loss": avg_val_loss,
                "train_dice": avg_train_dice,
                "train_loss": avg_train_loss,
                "in_channels": 3,
                "num_classes": 1,
                "init_features": init_features,
                "image_size": list(image_size),
                "device": str(device),
                "model_type": "UNet",
                "task": "binary_tomato_lesion_segmentation",
            }, best_ckpt_file)
            saved_tag = " -> BEST (*)"
        else:
            epochs_no_improve += 1
            saved_tag = f" (no improve: {epochs_no_improve}/{patience})"

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_duration:.1f}s, lr={current_lr:.1e}] | "
            f"Train Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}, IoU: {avg_train_iou:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}, IoU: {avg_val_iou:.4f}, Acc: {avg_val_acc*100:.2f}%"
            f"{saved_tag}"
        )

        if epochs_no_improve >= patience:
            print(f"\n[Early Stopping Triggered] Validation Dice did not improve for {patience} consecutive epochs.")
            print(f"Stopping training at Epoch {epoch}. Best epoch was Epoch {best_epoch} (Val Dice: {best_val_dice:.4f}).")
            break

    total_training_time = time.time() - start_total_time

    # 6. Save Training History
    hist_p = Path(history_file)
    hist_p.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_p, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # 7. Save Training Configuration
    config = {
        "model_architecture": "UNet",
        "in_channels": 3,
        "out_channels": 1,
        "init_features": init_features,
        "total_parameters": total_params,
        "input_resolution": list(image_size),
        "loss_function": "BinaryCombinedLoss (0.5 * BCEWithLogitsLoss + 0.5 * BinaryDiceLoss)",
        "optimizer": "AdamW",
        "initial_learning_rate": lr,
        "weight_decay": weight_decay,
        "lr_scheduler": "ReduceLROnPlateau (mode=max, factor=0.5, patience=3)",
        "batch_size": batch_size,
        "random_seed": seed,
        "max_epochs": epochs,
        "early_stopping_patience": patience,
        "completed_epochs": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_val_dice": round(best_val_dice, 4),
        "best_val_iou": round(history["val_iou"][best_epoch - 1], 4),
        "best_val_loss": round(history["val_loss"][best_epoch - 1], 4),
        "best_val_pixel_acc": round(history["val_pixel_acc"][best_epoch - 1], 4),
        "total_training_duration_seconds": round(total_training_time, 2),
        "device": str(device),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": 99,
        "test_split_touched": False,
        "checkpoint_path": str(best_ckpt_file),
    }

    conf_p = Path(config_file)
    with open(conf_p, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE K: TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total Training Duration: {total_training_time:.1f} s ({total_training_time/60:.2f} min)")
    print(f"Best Checkpoint: {best_ckpt_file}")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val Dice: {best_val_dice:.4f}")
    print(f"  Best Val IoU:  {config['best_val_iou']:.4f}")
    print(f"  Best Val Pixel Accuracy: {config['best_val_pixel_acc']*100:.2f}%")
    print(f"History saved: {hist_p}")
    print(f"Config saved:  {conf_p}")

    return {
        "history": history,
        "config": config,
        "best_ckpt_file": str(best_ckpt_file),
    }


if __name__ == "__main__":
    train_real_unet_model()
