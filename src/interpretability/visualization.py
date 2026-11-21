"""Grad-CAM Visualization Utilities (REQ-08).

Provides:
- Heatmap colormap coloring and RGB alpha blending.
- Multi-panel publication-grade diagnostic plots showing:
  Original Leaf | Class Activation Map | Grad-CAM Saliency Overlay.

Author: Crop Disease Detection Project Team
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def overlay_heatmap(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a 2D float heatmap in [0, 1] onto an RGB uint8 image.

    Args:
        rgb_image: uint8 RGB image of shape (H, W, 3) with values in [0, 255].
        heatmap: float32 2D array of shape (H, W) with values in [0, 1].
        alpha: Weight of the heatmap overlay (0.0 = only image, 1.0 = only heatmap).
        colormap: OpenCV colormap constant (default: cv2.COLORMAP_JET).

    Returns:
        uint8 RGB blended image of shape (H, W, 3).
    """
    if heatmap.shape[:2] != rgb_image.shape[:2]:
        heatmap = cv2.resize(heatmap, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Scale heatmap to [0, 255] uint8
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    # Apply colormap (OpenCV outputs BGR)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    # Alpha blending: (1 - alpha) * img + alpha * heatmap
    blended = np.clip(
        (1.0 - alpha) * rgb_image.astype(np.float32) + alpha * heatmap_colored_rgb.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    return blended


def plot_gradcam_explanation(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    predicted_label: str,
    confidence: float,
    ground_truth_label: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    target_layer_name: str = "layer4",
) -> plt.Figure:
    """Generate a clean 3-panel figure showing Original, Heatmap, and Overlay.

    Args:
        rgb_image: uint8 RGB image of shape (H, W, 3).
        heatmap: float32 heatmap of shape (H, W) in [0, 1].
        predicted_label: Name of predicted disease class.
        confidence: Prediction probability in [0, 1].
        ground_truth_label: Optional actual ground truth label.
        save_path: Optional output path to save the PNG file.
        target_layer_name: Name of convolutional layer used for hooks.

    Returns:
        matplotlib Figure object.
    """
    overlay = overlay_heatmap(rgb_image, heatmap, alpha=0.45)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    # Panel 1: Original Leaf
    axes[0].imshow(rgb_image)
    gt_str = f"\nGround Truth: {ground_truth_label}" if ground_truth_label else ""
    axes[0].set_title(f"Original Leaf Input{gt_str}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Grad-CAM Heatmap
    im = axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Grad-CAM Heatmap ({target_layer_name})\nImportance: α_k = GAP(∂y^c / ∂A^k)", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Saliency Overlay
    axes[2].imshow(overlay)
    correct_tag = "✓ Correct" if ground_truth_label and ground_truth_label == predicted_label else ""
    axes[2].set_title(
        f"Diagnosis Overlay: {predicted_label}\nConfidence: {confidence * 100:.2f}% {correct_tag}",
        fontsize=11,
        fontweight="bold",
        color="#1f77b4" if (not ground_truth_label or ground_truth_label == predicted_label) else "#d62728",
    )
    axes[2].axis("off")

    fig.suptitle(
        "Model Interpretability: Visual Explanation via Gradient-Weighted Class Activation Mapping",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")

    return fig

