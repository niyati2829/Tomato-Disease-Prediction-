"""Singular Value Decomposition (SVD) Image Compression Module.

Demonstrates low-rank approximation of image matrices via SVD:
    A = U * Sigma * V^T
    A_k = sum_{i=1}^k sigma_i * u_i * v_i^T

Computes:
1. Low-rank rank-k matrix approximation.
2. Frobenius reconstruction error ||A - A_k||_F.
3. Energy retention ratio (cumulative singular value variance).
4. Compression ratio (k*(m + n + 1) vs m*n).
5. Peak Signal-to-Noise Ratio (PSNR).
6. Multi-panel visual comparison across different k ranks.

Author: Crop Disease Detection Project Team
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def compress_matrix_svd(matrix: np.ndarray, k: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute rank-k truncated SVD reconstruction of a 2D matrix.

    Args:
        matrix: 2D float array of shape (m, n).
        k: Truncation rank (number of singular values to retain).

    Returns:
        Tuple of (reconstructed_matrix, diagnostics_dict).
    """
    m, n = matrix.shape
    k = min(k, m, n)

    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Low-rank reconstruction
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    Vtk = Vt[:k, :]
    approx = Uk @ Sk @ Vtk

    # Metrics
    frob_orig = np.linalg.norm(matrix, "fro")
    frob_err = np.linalg.norm(matrix - approx, "fro")
    rel_frob_err = frob_err / (frob_orig + 1e-12)

    total_energy = np.sum(S ** 2)
    retained_energy = np.sum(S[:k] ** 2) / (total_energy + 1e-12)

    original_params = m * n
    compressed_params = k * (m + n + 1)
    compression_ratio = compressed_params / original_params

    # PSNR calculation
    mse = np.mean((matrix - approx) ** 2)
    max_val = np.max(matrix) if np.max(matrix) > 0 else 1.0
    psnr = float(10 * np.log10((max_val ** 2) / (mse + 1e-12))) if mse > 0 else float("inf")

    diagnostics = {
        "rank_k": k,
        "singular_value_k": float(S[k - 1]) if k > 0 else 0.0,
        "frobenius_error": float(frob_err),
        "relative_frobenius_error": float(rel_frob_err),
        "retained_energy_percentage": float(retained_energy * 100.0),
        "compression_ratio": float(compression_ratio),
        "storage_savings_percentage": float((1.0 - compression_ratio) * 100.0),
        "psnr_db": float(psnr),
    }

    return approx, diagnostics


def compress_image_svd(
    image: np.ndarray,
    k_ranks: List[int],
) -> Dict[int, Tuple[np.ndarray, Dict[str, float]]]:
    """Compress multi-channel or grayscale image across multiple k ranks."""
    results = {}
    is_rgb = (image.ndim == 3 and image.shape[2] == 3)

    for k in k_ranks:
        if is_rgb:
            reconstructed_channels = []
            diag_accum = {}
            for c in range(3):
                channel = image[:, :, c].astype(float)
                recon_c, diag_c = compress_matrix_svd(channel, k)
                reconstructed_channels.append(np.clip(recon_c, 0, 255))
                for key, val in diag_c.items():
                    diag_accum[key] = diag_accum.get(key, 0.0) + val / 3.0
            reconstructed = np.stack(reconstructed_channels, axis=2).astype(np.uint8)
            results[k] = (reconstructed, diag_accum)
        else:
            gray = image.astype(float)
            recon, diag = compress_matrix_svd(gray, k)
            results[k] = (np.clip(recon, 0, 255).astype(np.uint8), diag)

    return results


def compress_rgb_svd(image: np.ndarray, k: int) -> Dict[str, Any]:
    """Convenience helper to compress RGB image at a single rank k."""
    res = compress_image_svd(image, [k])
    recon, diag = res[k]
    return {
        "reconstructed_rgb": recon,
        "mean_energy_retention": diag["retained_energy_percentage"] / 100.0,
        "compression_ratio": diag["compression_ratio"],
        "psnr_db": diag["psnr_db"],
        "diagnostics": diag,
    }


def plot_svd_compression_comparison(
    image_path: Path | str,
    output_path: Path | str,
    k_ranks: List[int] = [5, 20, 50, 100],
) -> Path:
    """Generate high-resolution SVD compression comparison figure."""
    img_file = Path(image_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if img_file.exists():
        with Image.open(img_file) as img:
            image_np = np.array(img.convert("RGB"))
    else:
        # Fallback synthetic pattern for robust sandbox execution
        x = np.linspace(-3, 3, 200)
        y = np.linspace(-3, 3, 200)
        xx, yy = np.meshgrid(x, y)
        z = np.sin(xx ** 2 + yy ** 2) * 127 + 128
        image_np = np.stack([z, np.roll(z, 20, axis=0), np.roll(z, 40, axis=1)], axis=2).astype(np.uint8)

    results = compress_image_svd(image_np, k_ranks)

    # 1 panel for original + N panels for k ranks + 1 panel for singular value decay curve
    ncols = len(k_ranks) + 1
    fig = plt.figure(figsize=(18, 7), dpi=150)

    # Top row: images
    gs = fig.add_gridspec(2, ncols, height_ratios=[1.2, 0.8])

    # Original Image
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(image_np)
    ax_orig.set_title(f"Original Image\n{image_np.shape[0]}×{image_np.shape[1]} (Rank {min(image_np.shape[:2])})", fontsize=9, fontweight="bold")
    ax_orig.axis("off")

    # Compressed images
    for idx, k in enumerate(k_ranks):
        ax_k = fig.add_subplot(gs[0, idx + 1])
        recon_img, diag = results[k]
        ax_k.imshow(recon_img)
        title_k = (
            f"k = {k}\n"
            f"PSNR: {diag['psnr_db']:.1f} dB\n"
            f"Energy: {diag['retained_energy_percentage']:.1f}%\n"
            f"Params: {diag['compression_ratio']*100:.1f}%"
        )
        ax_k.set_title(title_k, fontsize=8.5, fontweight="bold")
        ax_k.axis("off")

    # Bottom row: Energy retention & Singular value spectrum
    ax_spectrum = fig.add_subplot(gs[1, :ncols // 2])
    # Compute SVD of grayscale version for spectrum plot
    gray = np.mean(image_np, axis=2)
    _, S, _ = np.linalg.svd(gray, full_matrices=False)
    ax_spectrum.semilogy(range(1, len(S) + 1), S, color="#1976d2", linewidth=2)
    for k in k_ranks:
        if k <= len(S):
            ax_spectrum.axvline(k, color="#d32f2f", linestyle="--", alpha=0.6)
            ax_spectrum.scatter([k], [S[k - 1]], color="#d32f2f", zorder=4)
    ax_spectrum.set_title("Singular Value Energy Spectrum (Log Scale)", fontsize=10, fontweight="bold")
    ax_spectrum.set_xlabel("Singular Value Index i", fontsize=9)
    ax_spectrum.set_ylabel(r"Singular Value $\sigma_i$", fontsize=9)
    ax_spectrum.grid(True, linestyle="--", alpha=0.5)

    # Bottom right: Cumulative Energy Retention Curve
    ax_energy = fig.add_subplot(gs[1, ncols // 2:])
    cum_energy = np.cumsum(S ** 2) / np.sum(S ** 2) * 100.0
    ax_energy.plot(range(1, len(cum_energy) + 1), cum_energy, color="#388e3c", linewidth=2)
    for k in k_ranks:
        if k <= len(cum_energy):
            e_val = cum_energy[k - 1]
            ax_energy.scatter([k], [e_val], color="#f57c00", zorder=4)
            ax_energy.annotate(f"{e_val:.1f}%", xy=(k, e_val), xytext=(0, 6), textcoords="offset points",
                               fontsize=7.5, fontweight="bold", ha="center")
    ax_energy.set_title("Cumulative Variance Energy Retained (%)", fontsize=10, fontweight="bold")
    ax_energy.set_xlabel("Rank k", fontsize=9)
    ax_energy.set_ylabel("Variance Retained (%)", fontsize=9)
    ax_energy.set_ylim([0, 105])
    ax_energy.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("SVD Image Compression & Low-Rank Approximation Analysis (REQ-01)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
