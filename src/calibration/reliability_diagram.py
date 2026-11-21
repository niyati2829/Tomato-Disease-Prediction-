"""Reliability diagrams and confidence distribution visualizations."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from .calibration_metrics import compute_reliability_bins, compute_ece, compute_mce


def plot_reliability_diagram(
    uncalibrated_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    n_bins: int = 15,
    title: str = "Reliability Diagram (Holdout Test Set)",
) -> plt.Figure:
    """Plot side-by-side reliability diagrams before and after temperature scaling."""
    uncal_stats = compute_reliability_bins(uncalibrated_probs, labels, n_bins=n_bins)
    cal_stats = compute_reliability_bins(calibrated_probs, labels, n_bins=n_bins)

    uncal_ece = compute_ece(uncalibrated_probs, labels, n_bins=n_bins)
    uncal_mce = compute_mce(uncalibrated_probs, labels, n_bins=n_bins)
    cal_ece = compute_ece(calibrated_probs, labels, n_bins=n_bins)
    cal_mce = compute_mce(calibrated_probs, labels, n_bins=n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    configs = [
        (axes[0], uncal_stats, uncal_ece, uncal_mce, "Uncalibrated Model", "#e74c3c"),
        (axes[1], cal_stats, cal_ece, cal_mce, "Temperature Scaled Model", "#2ecc71"),
    ]

    for ax, stats, ece, mce, model_label, bar_color in configs:
        centers = stats["bin_centers"]
        accs = stats["bin_accuracies"]
        confs = stats["bin_confidences"]
        counts = stats["bin_counts"]
        width = 1.0 / n_bins

        # Plot bars for empirical accuracy
        non_empty = counts > 0
        ax.bar(
            centers[non_empty],
            accs[non_empty],
            width=width * 0.9,
            color=bar_color,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.0,
            label="Outputs (Empirical Accuracy)",
        )

        # Plot calibration gap
        gaps = np.abs(accs - confs)
        for i in range(n_bins):
            if counts[i] > 0:
                bottom = min(accs[i], confs[i])
                ax.bar(
                    centers[i],
                    gaps[i],
                    bottom=bottom,
                    width=width * 0.9,
                    color="#f39c12",
                    alpha=0.5,
                    hatch="//",
                    edgecolor="#d35400",
                    linewidth=0.8,
                    label="Calibration Gap" if i == 0 else "",
                )

        # Diagonal identity line
        ax.plot([0, 1], [0, 1], "--", color="navy", linewidth=2.0, label="Perfect Calibration")

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Confidence", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"{model_label}\nECE: {ece * 100:.2f}% | MCE: {mce * 100:.2f}%", fontsize=13, weight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", framealpha=0.9)

    plt.suptitle(title, fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_confidence_histograms(
    uncalibrated_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    save_path: Optional[str] = None,
    bins: int = 30,
    title: str = "Prediction Confidence Distribution",
) -> plt.Figure:
    """Plot confidence distribution comparing uncalibrated vs calibrated probabilities."""
    uncal_conf = np.max(uncalibrated_probs, axis=1) if uncalibrated_probs.ndim == 2 else uncalibrated_probs
    cal_conf = np.max(calibrated_probs, axis=1) if calibrated_probs.ndim == 2 else calibrated_probs

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axes[0].hist(uncal_conf, bins=bins, range=(0.0, 1.0), color="#3498db", edgecolor="black", alpha=0.8)
    axes[0].set_title(f"Uncalibrated Confidences\nMean: {np.mean(uncal_conf):.4f} | Median: {np.median(uncal_conf):.4f}", fontsize=12, weight="bold")
    axes[0].set_xlabel("Confidence", fontsize=11)
    axes[0].set_ylabel("Sample Count", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].hist(cal_conf, bins=bins, range=(0.0, 1.0), color="#2ecc71", edgecolor="black", alpha=0.8)
    axes[1].set_title(f"Calibrated Confidences\nMean: {np.mean(cal_conf):.4f} | Median: {np.median(cal_conf):.4f}", fontsize=12, weight="bold")
    axes[1].set_xlabel("Confidence", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(title, fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig
