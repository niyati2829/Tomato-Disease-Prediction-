"""Holdout Evaluation Engine for U-Net Semantic Segmentation (REQ-08).

Evaluates the trained U-Net checkpoint on the untouched test split,
computing Mean IoU, Per-class IoU, Dice coefficients, and Pixel Accuracy.
Exports results to reports/unet_evaluation_results.json.

Author: Crop Disease Detection Project Team
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.segmentation.dataset import LeafSegmentationDataset, generate_synthetic_lesion_benchmark
from src.segmentation.losses_metrics import (
    calculate_dice_score,
    calculate_iou,
    calculate_pixel_accuracy,
)
from src.segmentation.unet import UNet
from src.utils.device import get_device


def evaluate_unet_checkpoint(
    checkpoint_path: Path | str = "artifacts/checkpoints/segmentation/best_unet.pt",
    out_report_path: Path | str = "reports/unet_evaluation_results.json",
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate U-Net on test split."""
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

    device = get_device(preferred="mps")
    checkpoint = torch.load(ckpt_p, map_location=device)

    model = UNet(
        in_channels=3,
        num_classes=checkpoint.get("num_classes", 3),
        init_features=checkpoint.get("init_features", 32),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    splits = generate_synthetic_lesion_benchmark(n_samples=60, image_size=128, seed=seed)
    test_imgs, test_masks = splits["test"]
    test_ds = LeafSegmentationDataset(test_imgs, test_masks, image_size=(128, 128))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    all_ious = []
    all_dices = []
    all_accs = []
    per_class_ious = {0: [], 1: [], 2: []}
    per_class_dices = {0: [], 1: [], 2: []}

    class_names = {0: "Background", 1: "Healthy Lamina", 2: "Diseased Lesion"}

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            miou, c_ious = calculate_iou(preds, masks, num_classes=3)
            mdice, c_dices = calculate_dice_score(preds, masks, num_classes=3)
            acc = calculate_pixel_accuracy(preds, masks)

            all_ious.append(miou)
            all_dices.append(mdice)
            all_accs.append(acc)

            for c in range(3):
                per_class_ious[c].append(c_ious[c])
                per_class_dices[c].append(c_dices[c])

    results = {
        "checkpoint_evaluated": str(ckpt_p),
        "test_samples": len(test_ds),
        "mean_iou": float(np.mean(all_ious)),
        "mean_dice_score": float(np.mean(all_dices)),
        "pixel_accuracy": float(np.mean(all_accs)),
        "per_class_iou": {
            class_names[c]: float(np.mean(per_class_ious[c])) for c in range(3)
        },
        "per_class_dice": {
            class_names[c]: float(np.mean(per_class_dices[c])) for c in range(3)
        },
    }

    report_p = Path(out_report_path)
    report_p.parent.mkdir(parents=True, exist_ok=True)
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n================ U-Net Segmentation Test Evaluation ================")
    print(f"Mean IoU (Jaccard Index):     {results['mean_iou']:.4f}")
    print(f"Mean Dice Score (F1):          {results['mean_dice_score']:.4f}")
    print(f"Overall Pixel Accuracy:        {results['pixel_accuracy']*100:.2f}%\n")
    print("Per-Class Metrics:")
    for c_name in class_names.values():
        print(f"  • {c_name:16s} | IoU: {results['per_class_iou'][c_name]:.4f} | Dice: {results['per_class_dice'][c_name]:.4f}")
    print(f"\nResults saved to: {report_p}")

    return results


if __name__ == "__main__":
    evaluate_unet_checkpoint()

