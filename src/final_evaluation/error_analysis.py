"""In-depth classification error analysis, confusion metrics, and failure characterization."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn


def clean_class_name(name: str) -> str:
    """Format PlantVillage class names for readable display."""
    return name.replace("Tomato___", "").replace("_", " ")


def analyze_classification_errors(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
) -> Dict[str, any]:
    """Execute exhaustive error analysis over the dataset.

    Args:
        model: Production PyTorch model.
        dataloader: Holdout test DataLoader.
        class_names: List of class strings in index order.
        device: Evaluation device.

    Returns:
        Dictionary of confusion matrix, per-class metrics, error instances, and confidences.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_confs = []
    all_probs = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    confs_arr = np.array(all_confs)
    probs_arr = np.concatenate(all_probs, axis=0)

    # 1. Confusion Matrix
    cm = confusion_matrix(targets_arr, preds_arr, labels=list(range(len(class_names))))

    # 2. Per-class metrics
    report_dict = classification_report(
        targets_arr,
        preds_arr,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # 3. Identify all error cases
    error_indices = np.where(preds_arr != targets_arr)[0]
    correct_indices = np.where(preds_arr == targets_arr)[0]

    errors = []
    dataset = dataloader.dataset
    has_df = hasattr(dataset, "df")

    for idx in error_indices:
        t_cls = targets_arr[idx]
        p_cls = preds_arr[idx]
        conf = confs_arr[idx]
        img_rel_path = dataset.df.iloc[idx]["image_path"] if has_df else f"sample_{idx}"
        errors.append({
            "sample_index": int(idx),
            "image_path": str(img_rel_path),
            "true_class_idx": int(t_cls),
            "true_class_name": class_names[t_cls],
            "pred_class_idx": int(p_cls),
            "pred_class_name": class_names[p_cls],
            "confidence": float(conf),
            "margin": float(conf - probs_arr[idx, t_cls]),
        })

    # 4. Top Confused Pairs
    confusion_pairs = {}
    for err in errors:
        pair = (err["true_class_name"], err["pred_class_name"])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

    # 5. High-confidence errors vs low-confidence correct
    high_conf_errors = [e for e in errors if e["confidence"] >= 0.80]
    low_conf_correct = [int(idx) for idx in correct_indices if confs_arr[idx] < 0.70]

    return {
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "errors": errors,
        "top_confusions": sorted_confusions,
        "total_errors": len(errors),
        "total_samples": len(targets_arr),
        "accuracy": float(np.mean(preds_arr == targets_arr)),
        "correct_confs": confs_arr[correct_indices],
        "error_confs": confs_arr[error_indices],
        "high_conf_errors_count": len(high_conf_errors),
        "low_conf_correct_count": len(low_conf_correct),
        "class_names": class_names,
    }


def plot_detailed_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """Plot publication-quality confusion matrix heatmap."""
    clean_names = [clean_class_name(c) for c in class_names]

    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.nan_to_num(cm_display)
        fmt = ".2%"
        title = "Normalized Confusion Matrix (Holdout Test Set N=2,724)"
    else:
        cm_display = cm
        fmt = "d"
        title = "Raw Confusion Matrix (Holdout Test Set N=2,724)"

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm_display.max() / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            val = cm_display[i, j]
            txt = f"{val:.1%}" if normalize else str(val)
            color = "white" if val > thresh else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8, weight="bold")

    ax.set_xticks(np.arange(len(clean_names)))
    ax.set_yticks(np.arange(len(clean_names)))

    ax.set_xlabel("Predicted Pathology Class", fontsize=12, weight="bold", labelpad=10)
    ax.set_ylabel("True Ground-Truth Class", fontsize=12, weight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, weight="bold", pad=14)
    plt.xticks(rotation=40, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_error_confidence_distribution(
    correct_confs: np.ndarray,
    error_confs: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot comparative histogram of model confidence on correct vs erroneous classifications."""
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(0.0, 1.0, 25)
    ax.hist(correct_confs, bins=bins, color="#2ecc71", alpha=0.7, label=f"Correct (N={len(correct_confs)})", density=True, edgecolor="black")
    if len(error_confs) > 0:
        ax.hist(error_confs, bins=bins, color="#e74c3c", alpha=0.8, label=f"Errors (N={len(error_confs)})", density=True, edgecolor="black")

    ax.set_xlabel("Predicted Model Confidence", fontsize=11, weight="bold")
    ax.set_ylabel("Probability Density", fontsize=11, weight="bold")
    ax.set_title("Prediction Confidence Distribution: Correct vs Erroneous Predictions", fontsize=13, weight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(framealpha=0.9, fontsize=11)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_top_confusions(
    top_confusions: List[Tuple[Tuple[str, str], int]],
    save_path: Optional[str] = None,
    top_k: int = 8,
) -> plt.Figure:
    """Bar chart of the most frequent misclassification pairs."""
    top_k_items = top_confusions[:top_k]
    if not top_k_items:
        labels = ["None"]
        counts = [0]
    else:
        labels = [f"{clean_class_name(pair[0])} -> {clean_class_name(pair[1])}" for pair, _ in top_k_items]
        counts = [cnt for _, cnt in top_k_items]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, counts, color="#e67e22", edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Misclassified Samples", fontsize=11, weight="bold")
    ax.set_title(f"Top {len(labels)} Misclassification Pairs (Test Set Errors)", fontsize=13, weight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    for i, v in enumerate(counts):
        ax.text(v + 0.1, i, str(v), va="center", weight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig
