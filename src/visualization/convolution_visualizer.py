"""2D Discrete Spatial Convolution from First Principles (REQ-08).

Mathematical Formulation:
Given an input matrix X in R^{H x W} and a 2D filter kernel K in R^{K_h x K_w}:
    Y(i, j) = sum_{m=0}^{K_h-1} sum_{n=0}^{K_w-1} X_{pad}(i * s + m, j * s + n) * K(m, n)

Output Dimensions with Stride s and Padding p:
    H_out = floor((H + 2*p - K_h) / s) + 1
    W_out = floor((W + 2*p - K_w) / s) + 1

Author: Crop Disease Detection Project Team
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def convolve2d_numpy(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Perform 2D discrete cross-correlation/convolution on a single-channel image.

    Args:
        image: 2D array of shape (H, W).
        kernel: 2D array of shape (K_h, K_w).
        stride: Integer step size s (default: 1).
        padding: Integer zero-padding p applied to all 4 borders (default: 0).

    Returns:
        2D output feature map Y of shape (H_out, W_out).
    """
    if image.ndim != 2:
        raise ValueError(f"convolve2d_numpy expects 2D input (H, W), got shape {image.shape}")
    if kernel.ndim != 2:
        raise ValueError(f"convolve2d_numpy expects 2D kernel (K_h, K_w), got shape {kernel.shape}")
    if stride < 1:
        raise ValueError(f"Stride must be positive integer >= 1, got {stride}")
    if padding < 0:
        raise ValueError(f"Padding must be non-negative integer >= 0, got {padding}")

    H, W = image.shape
    K_h, K_w = kernel.shape

    # Apply zero-padding
    if padding > 0:
        X_pad = np.pad(image, ((padding, padding), (padding, padding)), mode="constant", constant_values=0.0)
    else:
        X_pad = image

    H_pad, W_pad = X_pad.shape
    H_out = (H_pad - K_h) // stride + 1
    W_out = (W_pad - K_w) // stride + 1

    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"Invalid dimensions: padded shape {(H_pad, W_pad)} smaller than kernel {(K_h, K_w)}")

    output = np.zeros((H_out, W_out), dtype=float)

    for i in range(H_out):
        for j in range(W_out):
            r_start = i * stride
            c_start = j * stride
            window = X_pad[r_start : r_start + K_h, c_start : c_start + K_w]
            output[i, j] = np.sum(window * kernel)

    return output


def get_classical_kernels() -> Dict[str, np.ndarray]:
    """Return dictionary of canonical 2D spatial convolution kernels."""
    return {
        "Sobel X (Vertical Veins)": np.array([
            [-1.0, 0.0, +1.0],
            [-2.0, 0.0, +2.0],
            [-1.0, 0.0, +1.0],
        ]),
        "Sobel Y (Horizontal Edges)": np.array([
            [-1.0, -2.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [+1.0, +2.0, +1.0],
        ]),
        "Laplacian (Second Derivative)": np.array([
            [ 0.0, -1.0,  0.0],
            [-1.0, +4.0, -1.0],
            [ 0.0, -1.0,  0.0],
        ]),
        "Sharpening Filter": np.array([
            [ 0.0, -1.0,  0.0],
            [-1.0, +5.0, -1.0],
            [ 0.0, -1.0,  0.0],
        ]),
        "Gaussian Blur 3x3 (Smoothing)": (1.0 / 16.0) * np.array([
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ]),
        "Ridge / Spot Detector": np.array([
            [-1.0, -1.0, -1.0],
            [-1.0, +8.0, -1.0],
            [-1.0, -1.0, -1.0],
        ]),
    }


def visualize_classical_filters(
    image_gray: np.ndarray,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Demonstrate discrete 2D spatial convolution across canonical edge/texture filters.

    Args:
        image_gray: 2D uint8 or float grayscale image of shape (H, W).
        save_path: Optional path to save figure.

    Returns:
        matplotlib Figure object.
    """
    kernels = get_classical_kernels()
    n_kernels = len(kernels)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300)
    axes = axes.flatten()

    # Panel 0: Input Image
    axes[0].imshow(image_gray, cmap="gray")
    axes[0].set_title(f"Input Leaf (Grayscale)\nShape: {image_gray.shape}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    img_norm = image_gray.astype(float) / 255.0 if image_gray.max() > 1.0 else image_gray.astype(float)

    for idx, (name, k) in enumerate(kernels.items(), start=1):
        filtered = convolve2d_numpy(img_norm, k, stride=1, padding=1)

        # Normalize filtered map for visual clarity
        vmin, vmax = np.percentile(filtered, [1, 99])
        im = axes[idx].imshow(filtered, cmap="viridis", vmin=vmin, vmax=vmax)
        k_str = f"K:\n" + "\n".join(["[" + " ".join([f"{v:+.1f}" for v in row]) + "]" for row in k])
        axes[idx].set_title(f"{name}\nStride=1, Pad=1", fontsize=10, fontweight="bold")
        axes[idx].axis("off")
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    # Empty 8th subplot used for mathematical formula annotation
    formula_text = (
        "Discrete 2D Spatial Convolution:\n\n"
        "Y(i, j) = ∑ₘ ∑ₙ X_pad(i·s + m, j·s + n) · K(m, n)\n\n"
        "Output Dimensions:\n"
        "  H_out = ⌊(H + 2p - K_h) / s⌋ + 1\n"
        "  W_out = ⌊(W + 2p - K_w) / s⌋ + 1\n\n"
        "Properties:\n"
        "• Sobel: Computes directional image gradients ∇f.\n"
        "• Laplacian: Computes isotropic divergence ∇²f.\n"
        "• Gaussian: Suppresses high-frequency sensor noise.\n"
        "• Ridge: Isolates high-curvature spot lesions."
    )
    axes[7].text(0.05, 0.5, formula_text, fontsize=9.5, family="monospace", va="center")
    axes[7].axis("off")

    fig.suptitle(
        "Educational Demonstration: 2D Spatial Convolution & Classical Feature Kernels",
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

