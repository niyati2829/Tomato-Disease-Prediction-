"""Visualization Utilities for Semantic Segmentation (REQ-08).

Generates 4-panel diagnostic visual comparisons:
Original RGB Image | Ground-Truth Mask | U-Net Prediction | Disease Overlay

Author: Crop Disease Detection Project Team
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.segmentation.dataset import LeafSegmentationDataset


# Discrete RGB colors for each class:
# 0: Background -> Black (0, 0, 0)
# 1: Healthy Lamina -> Forest Green (34, 139, 34)
# 2: Diseased Lesion -> Bright Crimson Red (220, 20, 60)
CLASS_COLORS = {
    0: np.array([20, 20, 20], dtype=np.uint8),
    1: np.array([46, 139, 87], dtype=np.uint8),
    2: np.array([220, 20, 60], dtype=np.uint8),
}


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert 2D integer class mask (H, W) to 3D RGB color visualization (H, W, 3)."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask


def plot_segmentation_comparisons(
    model: nn.Module,
    dataset: LeafSegmentationDataset,
    device: torch.device,
    n_samples: int = 4,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Generate 4-column comparison panel across multiple test samples."""
    model.eval()

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 3.8 * n_samples), dpi=300)
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    col_headers = [
        "Original RGB Image",
        "Ground-Truth Mask",
        "U-Net Predicted Mask",
        "Lesion Saliency Overlay",
    ]

    for i in range(min(n_samples, len(dataset))):
        img_tensor, mask_tensor = dataset[i]
        raw_mask = mask_tensor.numpy()

        # Load raw RGB image from path for un-normalized display
        pil_raw = Image.open(dataset.image_paths[i]).convert("RGB").resize(dataset.image_size)
        rgb_raw = np.array(pil_raw)

        # Predict
        with torch.no_grad():
            inp = img_tensor.unsqueeze(0).to(device)
            logits = model(inp)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        gt_color = mask_to_color(raw_mask)
        pred_color = mask_to_color(pred)

        # Alpha overlay of predicted lesions on original image
        overlay = rgb_raw.copy().astype(float)
        lesion_mask = (pred == 2)
        lamina_mask = (pred == 1)

        # Slight green tint for lamina, vivid red for lesions
        overlay[lamina_mask] = 0.7 * overlay[lamina_mask] + 0.3 * np.array([46, 139, 87])
        overlay[lesion_mask] = 0.4 * overlay[lesion_mask] + 0.6 * np.array([220, 20, 60])
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Column 0: Original
        axes[i, 0].imshow(rgb_raw)
        axes[i, 0].set_ylabel(f"Sample #{i+1}", fontsize=11, fontweight="bold")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Column 1: Ground Truth
        axes[i, 1].imshow(gt_color)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        # Column 2: Predicted
        axes[i, 2].imshow(pred_color)
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        # Column 3: Overlay
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])

        if i == 0:
            for c_idx, title in enumerate(col_headers):
                axes[0, c_idx].set_title(title, fontsize=11, fontweight="bold", pad=8)

    # Class Legend Text in Figure Header
    legend_str = "[Legend: Black = Background (0) | Green = Leaf Lamina (1) | Red = Diseased Lesions (2)]"
    fig.suptitle(
        f"Semantic Segmentation: U-Net Lesion Delineation & Lamina Segmentation\n{legend_str}",
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
    from src.segmentation.dataset import generate_synthetic_lesion_benchmark
    from src.segmentation.unet import UNet
    from src.utils.device import get_device

    checkpoint_path = Path("artifacts/checkpoints/segmentation/best_unet.pt")
    out_fig = Path("artifacts/figures/segmentation/unet_leaf_lesion_segmentation.png")

    device = get_device(preferred="mps")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = UNet(
        in_channels=3,
        num_classes=checkpoint.get("num_classes", 3),
        init_features=checkpoint.get("init_features", 32),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    splits = generate_synthetic_lesion_benchmark(n_samples=60, image_size=128, seed=42)
    test_imgs, test_masks = splits["test"]
    test_ds = LeafSegmentationDataset(test_imgs, test_masks, image_size=(128, 128))

    fig = plot_segmentation_comparisons(model, test_ds, device, n_samples=4, save_path=out_fig)
    plt.close(fig)
    print(f"[Phase D] Saved U-Net segmentation visualization to: {out_fig}")

