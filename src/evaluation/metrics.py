"""Classification Metrics Computation Module.

Provides standard evaluation metrics using scikit-learn:
- Accuracy
- Macro and Weighted Precision, Recall, and F1-score
- Per-class precision, recall, F1-score, and support
- Confusion matrix computation

Author: Crop Disease Detection Project Team
"""

from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics for multi-class evaluation.

    Args:
        y_true: Ground truth integer class labels of shape (N,).
        y_pred: Predicted integer class labels of shape (N,).
        class_names: List of class name strings sorted by class index (0 to C-1).

    Returns:
        Dictionary containing overall, macro, weighted, and per-class metrics.
    """
    overall_acc = float(accuracy_score(y_true, y_pred) * 100.0)

    # Macro metrics (unweighted arithmetic mean across classes)
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Weighted metrics (weighted by support/frequency of each class)
    weighted_p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Per-class support (count of true occurrences)
    unique_labels, counts = np.unique(y_true, return_counts=True)
    support_dict = dict(zip(unique_labels.tolist(), counts.tolist()))

    per_class_dict = {}
    for idx, name in enumerate(class_names):
        support = support_dict.get(idx, 0)
        per_class_dict[name] = {
            "class_index": idx,
            "precision": float(per_class_p[idx]),
            "recall": float(per_class_r[idx]),
            "f1_score": float(per_class_f1[idx]),
            "support": int(support),
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return {
        "overall_accuracy": overall_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "total_samples": int(len(y_true)),
    }


def format_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> str:
    """Generate human-readable scikit-learn text classification report."""
    short_names = [
        name.replace("Tomato___", "").replace("_", " ") for name in class_names
    ]
    return classification_report(
        y_true,
        y_pred,
        target_names=short_names,
        digits=4,
        zero_division=0,
    )
