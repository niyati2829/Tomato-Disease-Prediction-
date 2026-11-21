"""Visualization Utilities for Pest Object Detection (REQ-08).

Draws ground-truth bounding boxes (green) and predicted bounding boxes (cyan/magenta)
with class labels and confidence scores on RGB foliage images.
Saves figure to artifacts/figures/detection/pest_detection_predictions.png.

Author: Crop Disease Detection Project Team
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.detection.dataset import PEST_CLASSES, PestDetectionDataset


BOX_COLORS = {
    0: "#00ffff",  # Cyan for Whitefly
    1: "#ff00ff",  # Magenta for Aphid
    2: "#ffaa00",  # Orange for Spider Mite
}


def plot_detection_comparisons(
    model: nn.Module,
    dataset: PestDetectionDataset,
    device: torch.device,
    n_samples: int = 4,
    conf_threshold: float = 0.20,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Generate side-by-side ground truth vs predicted bounding box comparisons."""
    model.eval()

    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 5.5 * n_samples), dpi=300)
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(min(n_samples, len(dataset))):
        img_tensor, target = dataset[i]
        gt_boxes = target["boxes"].numpy()
        gt_labels = target["labels"].numpy()

        # Raw RGB image
        pil_raw = Image.open(dataset.image_paths[i]).convert("RGB").resize(dataset.image_size)
        rgb_raw = np.array(pil_raw)

        # Predictions
        with torch.no_grad():
            inp = img_tensor.unsqueeze(0).to(device)
            preds = model.predict_boxes(inp, conf_threshold=conf_threshold, iou_threshold=0.5)[0]

        pred_boxes = preds["boxes"].cpu().numpy()
        pred_scores = preds["scores"].cpu().numpy()
        pred_labels = preds["labels"].cpu().numpy()

        # Panel 0: Ground Truth
        axes[i, 0].imshow(rgb_raw)
        axes[i, 0].set_title(f"Sample #{i+1}: Ground-Truth Pest Annotations ({len(gt_boxes)} pests)", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        for box, label in zip(gt_boxes, gt_labels):
            xmin, ymin, xmax, ymax = box
            w = xmax - xmin
            h = ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin), w, h,
                linewidth=2.2, edgecolor="#00ff00", facecolor="none",
            )
            axes[i, 0].add_patch(rect)
            class_name = PEST_CLASSES.get(int(label), f"Class {label}")
            axes[i, 0].text(
                xmin, max(ymin - 4, 10), class_name,
                fontsize=8.5, color="white", fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.2", facecolor="#00aa00", alpha=0.8, edgecolor="none"),
            )

        # Panel 1: Model Predictions
        axes[i, 1].imshow(rgb_raw)
        axes[i, 1].set_title(f"Sample #{i+1}: Tiny-YOLO Predictions ({len(pred_boxes)} detections)", fontsize=11, fontweight="bold")
        axes[i, 1].axis("off")

        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            xmin, ymin, xmax, ymax = box
            w = xmax - xmin
            h = ymax - ymin
            c_color = BOX_COLORS.get(int(label), "#ffff00")
            rect = patches.Rectangle(
                (xmin, ymin), w, h,
                linewidth=2.2, edgecolor=c_color, facecolor="none",
            )
            axes[i, 1].add_patch(rect)
            class_name = PEST_CLASSES.get(int(label), f"Class {label}")
            axes[i, 1].text(
                xmin, max(ymin - 4, 10), f"{class_name}: {score*100:.1f}%",
                fontsize=8.5, color="black", fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.2", facecolor=c_color, alpha=0.85, edgecolor="none"),
            )

    fig.suptitle(
        "Agricultural Pest Object Detection: Ground-Truth Annotations vs Predicted Bounding Boxes",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    from src.detection.dataset import generate_synthetic_pest_benchmark
    from src.detection.model import TinyYOLODetector
    from src.utils.device import get_device

    ckpt_path = Path("artifacts/checkpoints/detection/best_pest_detector.pt")
    out_fig = Path("artifacts/figures/detection/pest_detection_predictions.png")

    device = get_device(preferred="mps")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = TinyYOLODetector(
        in_channels=3,
        num_classes=checkpoint.get("num_classes", 3),
        grid_size=checkpoint.get("grid_size", 8),
        num_boxes_per_cell=checkpoint.get("num_boxes_per_cell", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    splits = generate_synthetic_pest_benchmark(n_samples=50, image_size=256, seed=42)
    test_imgs, test_anns = splits["test"]
    test_ds = PestDetectionDataset(test_imgs, test_anns, image_size=(256, 256))

    fig = plot_detection_comparisons(model, test_ds, device, n_samples=3, save_path=out_fig)
    plt.close(fig)
    print(f"[Phase D] Saved Pest Detection visualization to: {out_fig}")
