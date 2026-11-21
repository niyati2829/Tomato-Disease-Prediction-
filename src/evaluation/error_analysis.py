"""Qualitative Error Analysis Module.

Provides functions to:
1. Identify misclassified samples with ground truth, prediction, and confidence score.
2. Calculate per-class false positives and false negatives.
3. Identify top confused class pairs.
4. Render and save a publication-quality misclassification gallery.

Author: Crop Disease Detection Project Team
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def find_misclassifications(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    file_paths: List[str],
    class_names: List[str],
) -> List[Dict[str, Any]]:
    """Identify and rank all misclassified instances.

    Args:
        y_true: True integer labels.
        y_pred: Predicted integer labels.
        y_probs: Softmax probabilities of shape (N, C).
        file_paths: Image file paths (relative or absolute).
        class_names: List of class name strings sorted by index.

    Returns:
        List of error dicts sorted by prediction confidence descending.
    """
    errors = []
    for i in range(len(y_true)):
        true_idx = int(y_true[i])
        pred_idx = int(y_pred[i])
        if true_idx != pred_idx:
            pred_conf = float(y_probs[i, pred_idx])
            true_conf = float(y_probs[i, true_idx])
            errors.append({
                "sample_index": i,
                "file_path": file_paths[i],
                "true_class_index": true_idx,
                "true_class_name": class_names[true_idx],
                "pred_class_index": pred_idx,
                "pred_class_name": class_names[pred_idx],
                "pred_confidence": pred_conf,
                "true_confidence": true_conf,
            })

    # Sort by predicted confidence descending (highest confidence mistakes first)
    errors.sort(key=lambda x: x["pred_confidence"], reverse=True)
    return errors


def analyze_confusion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Identify most frequently confused class pairs and compute FP/FN per class.

    Returns:
        Tuple of (confused_pairs_list, per_class_fp_fn_dict).
    """
    pair_counts = {}
    fp_counts = {name: 0 for name in class_names}
    fn_counts = {name: 0 for name in class_names}

    for true_idx, pred_idx in zip(y_true, y_pred):
        if true_idx != pred_idx:
            true_name = class_names[true_idx]
            pred_name = class_names[pred_idx]
            pair = (true_name, pred_name)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            fn_counts[true_name] += 1
            fp_counts[pred_name] += 1

    sorted_pairs = [
        {"true_class": pair[0], "pred_class": pair[1], "count": count}
        for pair, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    per_class_fp_fn = {}
    for name in class_names:
        per_class_fp_fn[name] = {
            "false_positives": fp_counts[name],
            "false_negatives": fn_counts[name],
        }

    return sorted_pairs, per_class_fp_fn


def plot_misclassification_gallery(
    errors: List[Dict[str, Any]],
    output_path: Path | str,
    root_dir: Path | str = ".",
    max_images: int = 12,
) -> Path:
    """Render a visual gallery of top misclassified leaf images.

    Args:
        errors: List of error dictionaries from find_misclassifications.
        output_path: Where to save the output PNG.
        root_dir: Root directory prepended to relative image paths.
        max_images: Maximum number of images to plot (e.g. 12 for 3x4 grid).

    Returns:
        Saved file Path.
    """
    root_path = Path(root_dir)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not errors:
        # Perfect model fallback
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        ax.text(0.5, 0.5, "Zero misclassifications detected!", ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.savefig(out_file, bbox_inches="tight")
        plt.close(fig)
        return out_file

    n_images = min(len(errors), max_images)
    ncols = 4
    nrows = int(np.ceil(n_images / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), dpi=150)
    axes_flat = np.array(axes).flatten()

    for idx in range(nrows * ncols):
        ax = axes_flat[idx]
        if idx < n_images:
            err = errors[idx]
            img_path = root_path / err["file_path"]
            if img_path.exists():
                try:
                    with Image.open(img_path) as img:
                        ax.imshow(img.convert("RGB"))
                except Exception:
                    ax.text(0.5, 0.5, "Image Load Error", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "File Missing", ha="center", va="center")

            true_short = err["true_class_name"].replace("Tomato___", "").replace("_", " ")
            pred_short = err["pred_class_name"].replace("Tomato___", "").replace("_", " ")

            title_text = (
                f"True: {true_short}\n"
                f"Pred: {pred_short}\n"
                f"Conf: {err['pred_confidence']*100:.1f}% (True: {err['true_confidence']*100:.1f}%)"
            )
            ax.set_title(title_text, fontsize=9, fontweight="bold", color="#d32f2f", pad=6)
        ax.axis("off")

    plt.suptitle("Top Representative Misclassifications on Holdout Test Set", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
