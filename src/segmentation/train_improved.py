"""Phase M1: Training Engine for Improved Tomato Lesion Segmentation U-Net.

Incorporates controlled improvements derived from Phase L error analysis:
1. Synchronized spatial and photometric data augmentation (flips, rotations, color jitter).
2. Combined BCE + Tversky Loss (alpha=0.7, beta=0.3) to penalize over-segmentation and false positives.
3. Enhanced weight decay regularization (5e-4) and Cosine Annealing learning rate schedule.
4. Validation-based model selection (strictly no test set access).

Saves improved checkpoint to:
  artifacts/checkpoints/segmentation_real/best_unet_improved.pt
Saves training history to:
  reports/unet_improved_training_history.json
Saves validation overlays to:
  reports/plantseg_audit/val_prediction_overlays_improved/

Author: Crop Disease Detection Project Team
Phase: Phase M1
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from src.segmentation.losses_metrics import CombinedBCETverskyLoss, calculate_binary_metrics
from src.segmentation.plantseg_dataset import PlantSegTomatoLesionDataset
from src.segmentation.unet import UNet
from src.utils.device import get_device, get_device_info


def train_improved_unet_model(
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    weight_bce: float = 0.3,
    alpha_tversky: float = 0.7,
    beta_tversky: float = 0.3,
    patience: int = 8,
    init_features: int = 16,
    image_size: Tuple[int, int] = (256, 256),
    checkpoint_dir: Path | str = "artifacts/checkpoints/segmentation_real",
    history_file: Path | str = "reports/unet_improved_training_history.json",
    config_file: Path | str = "reports/unet_improved_training_config.json",
    val_overlay_dir: Path | str = "reports/plantseg_audit/val_prediction_overlays_improved",
    seed: int = 42,
) -> Dict[str, Any]:
    """Train improved U-Net model with data augmentation and Tversky false-positive suppression."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ckpt_p = Path(checkpoint_dir)
    ckpt_p.mkdir(parents=True, exist_ok=True)
    best_ckpt_file = ckpt_p / "best_unet_improved.pt"

    ovl_p = Path(val_overlay_dir)
    ovl_p.mkdir(parents=True, exist_ok=True)

    device = get_device(preferred="mps")
    device_info = get_device_info(preferred="mps")

    print("=" * 70)
    print("PHASE M1: IMPROVED TOMATO LESION SEGMENTATION U-NET TRAINING")
    print("=" * 70)
    print(f"Device: {device} ({device_info.get('device_name', 'Host')})")
    print(f"Platform: {device_info.get('platform')}")
    print(f"Random seed: {seed}")
    print(f"Resolution: {image_size[0]}x{image_size[1]}")
    print(f"Batch size: {batch_size} | Base features: {init_features} | Max epochs: {epochs}")
    print(f"Data Augmentation: ENABLED (Flips, Rotations, ColorJitter)")
    print(f"Loss Function: CombinedBCETverskyLoss (BCE={weight_bce}, Tversky alpha={alpha_tversky}, beta={beta_tversky})")
    print(f"Optimizer: AdamW (LR={lr:.1e}, WeightDecay={weight_decay:.1e}) + CosineAnnealingLR")
    print(f"Patience: {patience} epochs on validation Dice score")
    print("=" * 70)

    # 1. Dataset Loading (Strictly train and validation splits only)
    train_ds = PlantSegTomatoLesionDataset(
        split="train",
        image_size=image_size,
        as_float_mask=True,
        preserve_aspect_ratio=True,
        augment=True,  # Enable synchronized augmentation on training split
    )
    val_ds = PlantSegTomatoLesionDataset(
        split="val",
        image_size=image_size,
        as_float_mask=True,
        preserve_aspect_ratio=True,
        augment=False,  # Strictly deterministic for validation
    )

    print(f"Dataset Loaded: {len(train_ds)} train samples (augmented), {len(val_ds)} validation samples")
    print("NOTE: Holdout test split (99 samples) is strictly reserved and untouched.\n")

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

    # 2. Model Initialization
    model = UNet(
        in_channels=3,
        num_classes=1,
        init_features=init_features,
        bilinear=True,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Architecture: UNet(in=3, out=1, init_features={init_features})")
    print(f"Total Parameters: {total_params:,}")

    # 3. Loss & Optimizer
    criterion = CombinedBCETverskyLoss(
        weight_bce=weight_bce,
        alpha=alpha_tversky,
        beta=beta_tversky,
        smooth=1e-5,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 4. Training Records
    history = {
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_precision": [],
        "val_recall": [],
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
        val_precs = []
        val_recs = []
        val_accs = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)

                val_losses.append(loss.item())
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                tp = float((preds * masks).sum().item())
                fp = float((preds * (1.0 - masks)).sum().item())
                fn = float(((1.0 - preds) * masks).sum().item())
                tn = float(((1.0 - preds) * (1.0 - masks)).sum().item())

                d = (2.0 * tp + 1e-5) / (2.0 * tp + fp + fn + 1e-5)
                u = (tp + 1e-5) / (tp + fp + fn + 1e-5)
                p = (tp + 1e-5) / (tp + fp + 1e-5)
                r = (tp + 1e-5) / (tp + fn + 1e-5)
                a = (tp + tn) / masks.numel()

                val_dices.append(d)
                val_ious.append(u)
                val_precs.append(p)
                val_recs.append(r)
                val_accs.append(a)

        avg_val_loss = float(np.mean(val_losses))
        avg_val_dice = float(np.mean(val_dices))
        avg_val_iou = float(np.mean(val_ious))
        avg_val_prec = float(np.mean(val_precs))
        avg_val_rec = float(np.mean(val_recs))
        avg_val_acc = float(np.mean(val_accs))

        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(round(avg_train_loss, 4))
        history["train_dice"].append(round(avg_train_dice, 4))
        history["train_iou"].append(round(avg_train_iou, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_dice"].append(round(avg_val_dice, 4))
        history["val_iou"].append(round(avg_val_iou, 4))
        history["val_precision"].append(round(avg_val_prec, 4))
        history["val_recall"].append(round(avg_val_rec, 4))
        history["val_pixel_acc"].append(round(avg_val_acc, 4))
        history["learning_rate"].append(current_lr)
        history["epoch_times_seconds"].append(round(epoch_duration, 2))

        # Checkpointing based on validation Dice score
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
                "val_precision": avg_val_prec,
                "val_recall": avg_val_rec,
                "val_pixel_acc": avg_val_acc,
                "val_loss": avg_val_loss,
                "train_dice": avg_train_dice,
                "train_loss": avg_train_loss,
                "in_channels": 3,
                "num_classes": 1,
                "init_features": init_features,
                "image_size": list(image_size),
                "model_type": "UNet",
                "loss_type": "CombinedBCETverskyLoss",
                "augmentation": True,
                "task": "improved_binary_tomato_lesion_segmentation",
            }, best_ckpt_file)
            saved_tag = " -> BEST (*)"
        else:
            epochs_no_improve += 1
            saved_tag = f" (no improve: {epochs_no_improve}/{patience})"

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_duration:.1f}s, lr={current_lr:.1e}] | "
            f"Train Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}, IoU: {avg_val_iou:.4f}, Prec: {avg_val_prec:.4f}, Rec: {avg_val_rec:.4f}"
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
        "loss_function": f"CombinedBCETverskyLoss (BCE={weight_bce}, Tversky alpha={alpha_tversky}, beta={beta_tversky})",
        "optimizer": "AdamW",
        "initial_learning_rate": lr,
        "weight_decay": weight_decay,
        "lr_scheduler": "CosineAnnealingLR (T_max=30, eta_min=1e-5)",
        "batch_size": batch_size,
        "random_seed": seed,
        "max_epochs": epochs,
        "early_stopping_patience": patience,
        "completed_epochs": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_val_dice": round(best_val_dice, 4),
        "best_val_iou": round(history["val_iou"][best_epoch - 1], 4),
        "best_val_precision": round(history["val_precision"][best_epoch - 1], 4),
        "best_val_recall": round(history["val_recall"][best_epoch - 1], 4),
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

    # 8. Generate Validation Prediction Overlays using the Best Improved Model
    generate_val_prediction_overlays(best_ckpt_file, val_ds, ovl_p, device)

    print("\n" + "=" * 70)
    print("PHASE M1: IMPROVED TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total Duration: {total_training_time:.1f} s ({total_training_time/60:.2f} min)")
    print(f"Best Improved Checkpoint: {best_ckpt_file}")
    print(f"  Best Epoch:           {best_epoch}")
    print(f"  Best Val Dice:        {best_val_dice:.4f}")
    print(f"  Best Val IoU:         {config['best_val_iou']:.4f}")
    print(f"  Best Val Precision:   {config['best_val_precision']:.4f}")
    print(f"  Best Val Recall:      {config['best_val_recall']:.4f}")
    print(f"  Best Val Pixel Acc:   {config['best_val_pixel_acc']*100:.2f}%")
    print(f"History saved: {hist_p}")
    print(f"Config saved:  {conf_p}")
    print(f"Validation Overlays: {ovl_p}")

    return {
        "history": history,
        "config": config,
        "best_ckpt_file": str(best_ckpt_file),
    }


def generate_val_prediction_overlays(
    checkpoint_path: Path,
    val_ds: PlantSegTomatoLesionDataset,
    out_dir: Path,
    device: torch.device,
) -> None:
    """Generate 4-panel prediction overlays for validation samples."""
    print(f"\nGenerating validation prediction overlays in {out_dir}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = UNet(in_channels=3, num_classes=1, init_features=ckpt["init_features"], bilinear=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    # Group by disease to pick 1 sample per disease (7 diseases)
    disease_samples = {}
    for i in range(len(val_ds)):
        meta = val_ds.get_metadata(i)
        dis = meta["disease"]
        if dis not in disease_samples:
            disease_samples[dis] = i
        if len(disease_samples) == 7:
            break

    for dis, idx in disease_samples.items():
        img_t, mask_t = val_ds[idx]
        meta = val_ds.get_metadata(idx)

        img_in = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(img_in)
            prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
            pred = (prob >= 0.5)

        gt = mask_t[0].numpy() > 0.5
        img_rgb_t = img_t.permute(1, 2, 0).numpy()
        img_rgb = np.clip((img_rgb_t * std + mean) * 255.0, 0, 255).astype(np.uint8)

        # Overlap analysis
        gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        diff_overlay = np.stack([gray * 0.4] * 3, axis=-1).astype(np.uint8)

        tp_mask = pred & gt
        fp_mask = pred & (~gt)
        fn_mask = (~pred) & gt

        diff_overlay[tp_mask] = [40, 220, 60]   # Green = TP
        diff_overlay[fp_mask] = [230, 40, 40]   # Red = FP
        diff_overlay[fn_mask] = [40, 100, 240]  # Blue = FN

        # Metric for this sample
        tp_cnt = tp_mask.sum()
        fp_cnt = fp_mask.sum()
        fn_cnt = fn_mask.sum()
        d_val = (2 * tp_cnt + 1e-5) / (2 * tp_cnt + fp_cnt + fn_cnt + 1e-5)
        u_val = (tp_cnt + 1e-5) / (tp_cnt + fp_cnt + fn_cnt + 1e-5)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Val Image: {meta['image_name']}\n{dis.title()}", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Ground Truth Lesion Mask\nRatio: {gt.mean():.3f}", fontsize=9)
        axes[1].axis("off")

        im_prob = axes[2].imshow(prob, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[2].set_title(f"Improved U-Net Prob Map\nDice: {d_val:.3f} | IoU: {u_val:.3f}", fontsize=9)
        axes[2].axis("off")
        plt.colorbar(im_prob, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].imshow(diff_overlay)
        axes[3].set_title("Overlap Analysis\nGreen=TP, Red=FP, Blue=FN", fontsize=9)
        axes[3].axis("off")

        plt.tight_layout()
        safe_name = Path(meta["image_name"]).stem
        out_f = out_dir / f"val_improved_{safe_name}.png"
        plt.savefig(out_f, dpi=130)
        plt.close(fig)
        print(f"  Rendered validation overlay: {out_f.name}")


if __name__ == "__main__":
    train_improved_unet_model()
