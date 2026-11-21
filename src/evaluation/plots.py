"""Visualization & Performance Plotting Module.

Provides functions to:
1. Plot and save training vs. validation loss and accuracy curves.
2. Plot and save raw and normalized confusion matrices.
3. Plot per-class metric comparison bar charts (F1, Recall).
4. Plot model benchmarking comparison bar charts.
5. Export high-resolution PNG figures directly into `artifacts/figures/`.

Author: Crop Disease Detection Project Team
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure writable matplotlib cache directory in sandboxed/restricted environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(
    history: Dict[str, List[float]],
    output_dir: Path | str,
    prefix: str = "baseline_cnn",
) -> Dict[str, Path]:
    """Plot and save training and validation loss and accuracy curves across epochs."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    saved_paths = {}

    # 1. Loss Curve
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(epochs, history["train_loss"], "o-", color="#1976d2", label="Training Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], "s--", color="#d32f2f", label="Validation Loss", linewidth=2)
    ax.set_title(f"{prefix.replace('_', ' ').title()}: Training vs Validation Loss", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax.set_xticks(list(epochs))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    loss_path = out_dir / f"{prefix}_loss.png"
    plt.tight_layout()
    fig.savefig(loss_path, bbox_inches="tight")
    plt.close(fig)
    saved_paths["loss"] = loss_path

    # 2. Accuracy Curve
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(epochs, history["train_acc"], "o-", color="#388e3c", label="Training Accuracy", linewidth=2)
    ax.plot(epochs, history["val_acc"], "s--", color="#f57c00", label="Validation Accuracy", linewidth=2)
    ax.set_title(f"{prefix.replace('_', ' ').title()}: Training vs Validation Accuracy", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_xticks(list(epochs))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    acc_path = out_dir / f"{prefix}_accuracy.png"
    plt.tight_layout()
    fig.savefig(acc_path, bbox_inches="tight")
    plt.close(fig)
    saved_paths["accuracy"] = acc_path

    # 3. Combined Learning Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    ax1.plot(epochs, history["train_loss"], "o-", color="#1976d2", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", color="#d32f2f", label="Val Loss", linewidth=2)
    ax1.set_title("Cross-Entropy Loss vs Epoch", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_xticks(list(epochs))
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(epochs, history["train_acc"], "o-", color="#388e3c", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "s--", color="#f57c00", label="Val Accuracy", linewidth=2)
    ax2.set_title("Classification Accuracy (%) vs Epoch", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_xticks(list(epochs))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle(f"{prefix.replace('_', ' ').title()}: Training & Validation Dynamics", fontsize=13, fontweight="bold", y=1.02)
    combined_path = out_dir / f"{prefix}_learning_curves.png"
    plt.tight_layout()
    fig.savefig(combined_path, bbox_inches="tight")
    plt.close(fig)
    saved_paths["combined"] = combined_path

    return saved_paths


def plot_confusion_matrix_heatmap(
    cm: np.ndarray | List[List[int]],
    class_names: List[str],
    output_path: Path | str,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    cmap: str = "Blues",
) -> Path:
    """Plot and save a publication-quality confusion matrix heatmap.

    Args:
        cm: Confusion matrix array of shape (C, C).
        class_names: List of class name strings sorted by index.
        output_path: Destination file path.
        title: Title of plot.
        normalize: If True, row-normalizes so each cell represents recall (0 to 1).
        cmap: Matplotlib colormap name.

    Returns:
        Path of saved figure.
    """
    cm_arr = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        display_mat = cm_arr / row_sums
        fmt = ".2f"
        vmax = 1.0
        cbar_label = "Recall (Row-Normalized)"
    else:
        display_mat = cm_arr
        fmt = "d"
        vmax = None
        cbar_label = "Count"

    short_names = [
        name.replace("Tomato___", "").replace("_", " ").replace(" Two-spotted spider mite", "\n(Spider mite)")
        for name in class_names
    ]

    fig, ax = plt.subplots(figsize=(11, 9.5), dpi=150)
    im = ax.imshow(display_mat, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)

    n_classes = len(class_names)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)

    thresh = display_mat.max() / 2.0 if display_mat.max() > 0 else 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            val = display_mat[i, j]
            raw_cnt = int(cm_arr[i, j])
            color = "white" if val > thresh else "black"
            if normalize:
                cell_text = f"{val:.2f}\n({raw_cnt})"
            else:
                cell_text = f"{raw_cnt}"
            ax.text(j, i, cell_text, ha="center", va="center", fontsize=7.5, color=color)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_per_class_metric_comparison(
    candidates_dict: Dict[str, Dict[str, Any]],
    class_names: List[str],
    output_path: Path | str,
    metric_key: str = "f1_score",
    title: str = "Per-Class F1-Score Comparison",
) -> Path:
    """Plot grouped bar chart comparing a per-class metric across candidate models.

    Args:
        candidates_dict: Dictionary mapping model display name -> dict with key 'per_class'.
        class_names: List of class names.
        output_path: Path to save figure.
        metric_key: Metric key inside candidate's per_class dict ('f1_score', 'recall', etc.).
        title: Figure title.
    """
    short_names = [
        name.replace("Tomato___", "").replace("_", " ").replace(" Two-spotted spider mite", "\n(Spider mite)")
        for name in class_names
    ]
    num_classes = len(class_names)
    num_models = len(candidates_dict)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    x = np.arange(num_classes)
    width = 0.8 / num_models

    colors = ["#78909c", "#42a5f5", "#2e7d32", "#ab47bc"]

    for m_idx, (m_name, m_data) in enumerate(candidates_dict.items()):
        vals = []
        for c_name in class_names:
            c_info = m_data.get("per_class", {}).get(c_name, {})
            vals.append(c_info.get(metric_key, 0.0) * 100.0)
        pos = x - 0.4 + width * (m_idx + 0.5)
        ax.bar(pos, vals, width, label=m_name, color=colors[m_idx % len(colors)], alpha=0.9)

    ax.set_ylabel(f"{metric_key.replace('_', ' ').title()} (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8.5)
    ax.set_ylim([75, 102])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, loc="lower right", fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


# Convenient alias
plot_per_class_metric_bars = plot_per_class_metric_comparison


def plot_model_benchmarking_bars(
    benchmark_data: List[Dict[str, Any]],
    output_path: Path | str,
) -> Path:
    """Plot multi-metric benchmark comparison across all models on Holdout Test Set."""
    model_names = [b["model_display_name"] for b in benchmark_data]
    test_accs = [b["test_accuracy"] for b in benchmark_data]
    test_macro_f1s = [b["test_macro_f1"] * 100.0 for b in benchmark_data]
    test_weighted_f1s = [b["test_weighted_f1"] * 100.0 for b in benchmark_data]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    rects1 = ax.bar(x - width, test_accs, width, label="Test Accuracy (%)", color="#1976d2")
    rects2 = ax.bar(x, test_macro_f1s, width, label="Test Macro F1 (%)", color="#388e3c")
    rects3 = ax.bar(x + width, test_weighted_f1s, width, label="Test Weighted F1 (%)", color="#f57c00")

    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Model Benchmarking Comparison on Holdout Test Set (N=2,724)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9.5, fontweight="bold")
    ax.set_ylim([90, 101])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, loc="lower right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Attach value labels above bars
    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


# Convenient alias
plot_per_class_metric_bars = plot_per_class_metric_comparison
