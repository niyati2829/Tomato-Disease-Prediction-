"""Evaluation Engine for Real Tomato Pest Detector (Tuta absoluta).

Evaluates best_pest_detector_real.pt on the unseen holdout test split.
Computes mAP@0.5, mAP@0.5:0.95, per-class metrics, inference latency,
and exports qualitative prediction figures.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.detection.metrics import evaluate_detections
from src.detection.model import TinyYOLODetector
from src.detection.real_pest_dataset import REAL_PEST_CLASSES, RealTomatoPestDataset
from src.detection.train import collate_fn
from src.utils.device import get_device


def evaluate_real_pest_detector(
    checkpoint_path: Path | str = "artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt",
    report_path: Path | str = "reports/pest_detection_real_evaluation.json",
    figure_path: Path | str = "artifacts/figures/phase_G/pest_detection_real_predictions.png",
    conf_threshold: float = 0.20,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run full evaluation on holdout test set."""
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

    device = get_device(preferred="cpu")  # CPU for standardized benchmarking
    checkpoint = torch.load(ckpt_p, map_location=device, weights_only=False)

    num_classes = checkpoint.get("num_classes", 2)
    grid_size = checkpoint.get("grid_size", 8)
    num_boxes = checkpoint.get("num_boxes_per_cell", 1)

    model = TinyYOLODetector(
        in_channels=3,
        num_classes=num_classes,
        grid_size=grid_size,
        num_boxes_per_cell=num_boxes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_ds = RealTomatoPestDataset(split="test", image_size=(256, 256))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    print(f"[Phase G] Evaluating Real Pest Detector on {len(test_ds)} holdout test images...")

    all_preds: List[Dict[str, torch.Tensor]] = []
    all_gts: List[Dict[str, torch.Tensor]] = []
    raw_images: List[torch.Tensor] = []

    # Latency profiling
    latencies = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            for i in range(images.shape[0]):
                single_img = images[i : i + 1]
                t0 = time.perf_counter()
                pred = model.predict_boxes(
                    single_img, conf_threshold=conf_threshold, iou_threshold=iou_threshold
                )[0]
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

                all_preds.append(pred)
                all_gts.append(targets[i])
                raw_images.append(images[i].cpu())

    metrics = evaluate_detections(all_preds, all_gts, num_classes=num_classes)

    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    # Package results
    results = {
        "dataset": "Shehu et al. (2025) Real Tuta absoluta Pest Dataset",
        "split": "test",
        "n_test_images": len(test_ds),
        "classes": REAL_PEST_CLASSES,
        "mAP_0.5": float(metrics["mAP_0.5"]),
        "mAP_0.5:0.95": float(metrics["mAP_0.5:0.95"]),
        "per_class_AP_0.5": {
            REAL_PEST_CLASSES.get(int(k), str(k)): float(v)
            for k, v in metrics["per_class_AP_0.5"].items()
        },
        "per_class_Precision_0.5": {
            REAL_PEST_CLASSES.get(int(k), str(k)): float(v)
            for k, v in metrics["per_class_Precision_0.5"].items()
        },
        "per_class_Recall_0.5": {
            REAL_PEST_CLASSES.get(int(k), str(k)): float(v)
            for k, v in metrics["per_class_Recall_0.5"].items()
        },
        "latency_ms": {
            "mean": mean_latency,
            "std": std_latency,
            "fps": fps,
            "device": str(device),
        },
    }

    # Save JSON report
    rep_p = Path(report_path)
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot predictions
    fig_p = Path(figure_path)
    fig_p.parent.mkdir(parents=True, exist_ok=True)
    _plot_detection_predictions(raw_images[:6], all_gts[:6], all_preds[:6], fig_p)

    print("\n" + "=" * 60)
    print("REAL TOMATO PEST DETECTION (Tuta absoluta) TEST RESULTS")
    print("=" * 60)
    print(f"mAP@0.5:      {results['mAP_0.5']:.4f}")
    print(f"mAP@0.5:0.95: {results['mAP_0.5:0.95']:.4f}")
    for c_id, name in REAL_PEST_CLASSES.items():
        ap = results["per_class_AP_0.5"].get(name, 0.0)
        p = results["per_class_Precision_0.5"].get(name, 0.0)
        r = results["per_class_Recall_0.5"].get(name, 0.0)
        print(f"  [{c_id}] {name:<22} | AP@0.5: {ap:.4f} | Prec: {p:.4f} | Rec: {r:.4f}")
    print(f"Mean Latency: {mean_latency:.2f} ms ({fps:.1f} FPS) on {device}")
    print(f"Report saved to: {rep_p}")
    print(f"Figure saved to: {fig_p}")
    print("=" * 60)

    return results


def _plot_detection_predictions(
    images: List[torch.Tensor],
    gts: List[Dict[str, torch.Tensor]],
    preds: List[Dict[str, torch.Tensor]],
    save_path: Path,
) -> None:
    """Plot sample test images with Ground Truth (green) vs Predicted (red/cyan) boxes."""
    n = min(6, len(images))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    for i in range(n):
        ax = axes[i]
        img_np = images[i].permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * std + mean, 0.0, 1.0)
        ax.imshow(img_np)

        # Ground truth (Green dashed)
        gt_boxes = gts[i]["boxes"]
        gt_labels = gts[i]["labels"]
        for b, lbl in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = b.tolist()
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="#00FF00",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                max(0, y1 - 4),
                f"GT: {REAL_PEST_CLASSES.get(int(lbl), str(lbl))}",
                color="black",
                fontsize=8,
                bbox=dict(boxstyle="square,pad=0.2", facecolor="#00FF00", alpha=0.8),
            )

        # Predicted (Red solid)
        pred_boxes = preds[i]["boxes"]
        pred_scores = preds[i]["scores"]
        pred_labels = preds[i]["labels"]
        for b, s, lbl in zip(pred_boxes, pred_scores, pred_labels):
            x1, y1, x2, y2 = b.tolist()
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="#FF3333",
                facecolor="none",
                linestyle="-",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                min(250, y2 + 10),
                f"Pred: {REAL_PEST_CLASSES.get(int(lbl), str(lbl))} ({s:.2f})",
                color="white",
                fontsize=8,
                bbox=dict(boxstyle="square,pad=0.2", facecolor="#FF3333", alpha=0.8),
            )

        ax.set_title(f"Test Sample {i+1} (Green=GT, Red=Pred)", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    evaluate_real_pest_detector()

