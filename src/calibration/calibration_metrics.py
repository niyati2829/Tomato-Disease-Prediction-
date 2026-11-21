"""Calibration evaluation metrics: ECE, MCE, Brier Score, and NLL."""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def compute_reliability_bins(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, np.ndarray]:
    """Compute bin-level statistics for reliability diagrams and ECE/MCE.

    Args:
        probs: Predicted probability distributions of shape (N, C) or top-1 confidences (N,).
        labels: True integer labels of shape (N,).
        n_bins: Number of equal-width confidence bins between 0 and 1.

    Returns:
        dict containing:
            - bin_edges: (n_bins + 1,)
            - bin_centers: (n_bins,)
            - bin_accuracies: (n_bins,) empirical accuracy per bin (0.0 if empty)
            - bin_confidences: (n_bins,) mean predicted confidence per bin (0.0 if empty)
            - bin_counts: (n_bins,) number of samples in each bin
            - bin_gaps: (n_bins,) |acc - conf| per bin
    """
    if probs.ndim == 2:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
    else:
        confidences = probs
        predictions = labels  # caller already filtered or passed accuracy mask

    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_accuracies = np.zeros(n_bins, dtype=np.float64)
    bin_confidences = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    bin_gaps = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences >= low) & (confidences < high)

        count = np.sum(in_bin)
        bin_counts[i] = count
        if count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            bin_accuracies[i] = bin_acc
            bin_confidences[i] = bin_conf
            bin_gaps[i] = np.abs(bin_acc - bin_conf)

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "bin_gaps": bin_gaps,
    }


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE).

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    bin_stats = compute_reliability_bins(probs, labels, n_bins=n_bins)
    total_samples = len(labels)
    if total_samples == 0:
        return 0.0

    counts = bin_stats["bin_counts"]
    gaps = bin_stats["bin_gaps"]
    ece = np.sum((counts / total_samples) * gaps)
    return float(ece)


def compute_mce(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Maximum Calibration Error (MCE).

    MCE = max_{m=1..M, |B_m| > 0} |acc(B_m) - conf(B_m)|
    """
    bin_stats = compute_reliability_bins(probs, labels, n_bins=n_bins)
    counts = bin_stats["bin_counts"]
    gaps = bin_stats["bin_gaps"]
    non_empty = counts > 0
    if not np.any(non_empty):
        return 0.0
    return float(np.max(gaps[non_empty]))


def compute_brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
    n_classes: int = 10,
) -> float:
    """Multi-class Brier score.

    BS = (1 / N) * sum_{i=1}^N sum_{c=1}^C (p_{ic} - y_{ic})^2
    Lower is better (0 is perfect forecast).
    """
    n_samples = len(labels)
    if n_samples == 0:
        return 0.0

    if probs.ndim == 1:
        # Binary or single-prob case
        y_true = labels.astype(np.float64)
        return float(np.mean((probs - y_true) ** 2))

    y_one_hot = np.zeros_like(probs)
    y_one_hot[np.arange(n_samples), labels] = 1.0
    brier = np.mean(np.sum((probs - y_one_hot) ** 2, axis=1))
    return float(brier)


def compute_nll(
    probs: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Negative Log-Likelihood (cross-entropy).

    NLL = - (1 / N) * sum_{i=1}^N log(p_{i, y_i} + eps)
    """
    n_samples = len(labels)
    if n_samples == 0:
        return 0.0

    clipped_probs = np.clip(probs, eps, 1.0)
    if clipped_probs.ndim == 2:
        true_class_probs = clipped_probs[np.arange(n_samples), labels]
    else:
        true_class_probs = clipped_probs
    nll = -np.mean(np.log(true_class_probs))
    return float(nll)


def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Compute full suite of calibration and performance metrics."""
    preds = np.argmax(probs, axis=1) if probs.ndim == 2 else (probs >= 0.5).astype(int)
    acc = float(np.mean(preds == labels))
    ece = compute_ece(probs, labels, n_bins=n_bins)
    mce = compute_mce(probs, labels, n_bins=n_bins)
    brier = compute_brier_score(probs, labels)
    nll = compute_nll(probs, labels)
    top1_conf = float(np.mean(np.max(probs, axis=1))) if probs.ndim == 2 else float(np.mean(probs))

    return {
        "accuracy": acc,
        "mean_confidence": top1_conf,
        "ece": ece,
        "mce": mce,
        "brier_score": brier,
        "nll": nll,
    }
