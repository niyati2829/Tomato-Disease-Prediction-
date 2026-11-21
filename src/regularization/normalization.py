"""Normalization Techniques: Batch Normalization vs Layer Normalization (REQ-05).

Implements from scratch using pure NumPy:
1. BatchNorm1d (Normalizes across batch dimension N per feature channel)
2. LayerNorm1d (Normalizes across feature dimension D per individual sample)

Compares:
- Normalized dimensions
- Batch statistics vs per-sample statistics
- Train vs Eval mode in BatchNorm (running stats vs batch stats)
- Sensitivity to batch size (e.g., small batch size degradation in BatchNorm vs stability in LayerNorm)

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


class BatchNorm1dNumPy:
    """Batch Normalization for 2D inputs (N, D).

    Normalizes along the batch dimension axis=0:
        mu_c = (1 / N) * sum_{i=1}^N x_{i, c}
        sigma_c^2 = (1 / N) * sum_{i=1}^N (x_{i, c} - mu_c)^2
        x_hat_{i, c} = (x_{i, c} - mu_c) / sqrt(sigma_c^2 + eps)
        y_{i, c} = gamma_c * x_hat_{i, c} + beta_c
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones((1, num_features), dtype=float)
        self.beta = np.zeros((1, num_features), dtype=float)
        self.running_mean = np.zeros((1, num_features), dtype=float)
        self.running_var = np.ones((1, num_features), dtype=float)
        self.training = True

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.training:
            # Batch statistics
            mean = np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)

            # Update running statistics (exponential moving average)
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var

            X_norm = (X - mean) / np.sqrt(var + self.eps)
        else:
            # Deterministic inference using frozen running statistics
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * X_norm + self.beta


class LayerNorm1dNumPy:
    """Layer Normalization for 2D inputs (N, D).

    Normalizes along the feature dimension axis=1 for each sample independently:
        mu_i = (1 / D) * sum_{j=1}^D x_{i, j}
        sigma_i^2 = (1 / D) * sum_{j=1}^D (x_{i, j} - mu_i)^2
        x_hat_{i, j} = (x_{i, j} - mu_i) / sqrt(sigma_i^2 + eps)
        y_{i, j} = gamma_j * x_hat_{i, j} + beta_j
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        self.num_features = num_features
        self.eps = eps
        self.gamma = np.ones((1, num_features), dtype=float)
        self.beta = np.zeros((1, num_features), dtype=float)
        self.training = True

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        # Compute mean and variance along feature dimension for EACH sample independently
        mean = np.mean(X, axis=1, keepdims=True)
        var = np.var(X, axis=1, keepdims=True)
        X_norm = (X - mean) / np.sqrt(var + self.eps)
        return self.gamma * X_norm + self.beta


def compare_batch_and_layer_norm(
    batch_sizes: List[int] = [32, 4, 2],
    num_features: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate BatchNorm and LayerNorm under standard and severely constrained batch sizes."""
    rng = np.random.RandomState(seed)
    results = {}

    for B in batch_sizes:
        # Synthetic activation with internal covariate shift (different channel means and variances)
        true_means = rng.uniform(-4.0, 4.0, size=(1, num_features))
        true_stds = rng.uniform(0.5, 3.5, size=(1, num_features))
        raw_X = rng.randn(B, num_features) * true_stds + true_means

        bn = BatchNorm1dNumPy(num_features=num_features)
        ln = LayerNorm1dNumPy(num_features=num_features)

        # Forward pass (Training mode)
        out_bn_train = bn.forward(raw_X)
        out_ln_train = ln.forward(raw_X)

        # Switch to eval mode
        bn.eval()
        ln.eval()
        out_bn_eval = bn.forward(raw_X)
        out_ln_eval = ln.forward(raw_X)

        results[f"B_{B}"] = {
            "batch_size": B,
            "raw_mean": float(np.mean(raw_X)),
            "raw_std": float(np.std(raw_X)),
            "bn_train_mean": float(np.mean(out_bn_train)),
            "bn_train_std": float(np.std(out_bn_train)),
            "bn_eval_diff": float(np.max(np.abs(out_bn_train - out_bn_eval))),
            "ln_train_mean": float(np.mean(out_ln_train)),
            "ln_train_std": float(np.std(out_ln_train)),
            "ln_eval_diff": float(np.max(np.abs(out_ln_train - out_ln_eval))),
            "raw_X": raw_X,
            "out_bn": out_bn_train,
            "out_ln": out_ln_train,
        }

    return results


def plot_normalization_comparison(
    results: Dict[str, Any],
    output_path: Path | str = "artifacts/figures/regularization/batchnorm_vs_layernorm_distributions.png",
) -> Path:
    """Plot comparative histograms and channel statistics of unnormalized, BatchNorm, and LayerNorm."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    data_b32 = results["B_32"]
    raw_vals = data_b32["raw_X"].ravel()
    bn_vals = data_b32["out_bn"].ravel()
    ln_vals = data_b32["out_ln"].ravel()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    # 1. Raw Activations (Covariate Shift)
    axes[0].hist(raw_vals, bins=35, color="#7f7f7f", alpha=0.75, edgecolor="black", linewidth=0.5)
    axes[0].set_title(rf"Unnormalized Activations\n$\mu={np.mean(raw_vals):.2f}, \sigma={np.std(raw_vals):.2f}$", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Activation Magnitude", fontsize=10)
    axes[0].set_ylabel("Density / Count", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # 2. Batch Normalization (Axis 0)
    axes[1].hist(bn_vals, bins=35, color="#1f77b4", alpha=0.75, edgecolor="black", linewidth=0.5)
    axes[1].set_title(rf"Batch Normalization (Axis 0)\n$\mu={np.mean(bn_vals):.2f}, \sigma={np.std(bn_vals):.2f}$", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Normalized Value", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    # 3. Layer Normalization (Axis 1)
    axes[2].hist(ln_vals, bins=35, color="#2ca02c", alpha=0.75, edgecolor="black", linewidth=0.5)
    axes[2].set_title(rf"Layer Normalization (Axis 1)\n$\mu={np.mean(ln_vals):.2f}, \sigma={np.std(ln_vals):.2f}$", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Normalized Value", fontsize=10)
    axes[2].grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Normalization Comparison: Stabilizing Hidden Feature Distributions ($B=32, D=50$)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
