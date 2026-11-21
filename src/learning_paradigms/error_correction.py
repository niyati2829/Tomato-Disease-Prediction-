"""Error-Correction Learning: Delta Rule & Least Mean Squares (LMS) (REQ-06).

Implements from scratch:
1. Widrow-Hoff Delta Rule / Least Mean Squares (LMS):
       y = w^T x + b
       error = t - y
       Delta w = eta * (t - y) * x
       Delta b = eta * (t - y)

Contrasts supervised error-driven learning Delta w ~ (t - y) * x against
unsupervised correlation-driven Hebbian learning Delta w ~ y * x.

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


class DeltaRuleLearner:
    """Adaptive Linear Neuron (Adaline) trained via the Delta Rule / LMS."""

    def __init__(self, n_features: int, lr: float = 0.05, seed: int = 42) -> None:
        self.lr = lr
        rng = np.random.RandomState(seed)
        self.w = rng.randn(n_features) * 0.2
        self.b = 0.0

    def predict_linear(self, x: np.ndarray) -> float:
        """Compute continuous linear activation y = w^T x + b."""
        return float(np.dot(self.w, x) + self.b)

    def predict_class(self, x: np.ndarray, threshold: float = 0.5) -> int:
        """Threshold continuous activation into binary class label {0, 1}."""
        return 1 if self.predict_linear(x) >= threshold else 0

    def train_step(self, x: np.ndarray, target: float) -> Tuple[float, float]:
        """Execute single LMS update step.

        Returns:
            Tuple of (output y, scalar error e = t - y).
        """
        y = self.predict_linear(x)
        error = target - y
        # Delta Rule parameter update
        self.w += self.lr * error * x
        self.b += self.lr * error
        return y, error

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 60, seed: int = 42) -> Dict[str, Any]:
        """Train Adaline over multiple epochs and track MSE loss."""
        rng = np.random.RandomState(seed)
        N = len(X)
        initial_w = self.w.copy()
        initial_b = self.b
        loss_history = []
        error_count_history = []

        for epoch in range(epochs):
            indices = rng.permutation(N)
            epoch_errors_sq = []
            misclassified = 0

            for idx in indices:
                xi = X[idx]
                target = y[idx]
                pred_lin, err = self.train_step(xi, target)
                epoch_errors_sq.append(err ** 2)

                pred_c = 1 if pred_lin >= 0.5 else 0
                if pred_c != int(target >= 0.5):
                    misclassified += 1

            mse = float(np.mean(epoch_errors_sq)) / 2.0
            loss_history.append(mse)
            error_count_history.append(misclassified)

        final_preds = np.array([self.predict_class(X[i]) for i in range(N)])

        return {
            "initial_w": initial_w,
            "initial_b": initial_b,
            "final_w": self.w.copy(),
            "final_b": self.b,
            "loss_history": np.array(loss_history),
            "error_count_history": np.array(error_count_history),
            "final_preds": final_preds,
        }


def generate_linearly_separable_data(
    n_samples: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 2D binary classification clusters separated by a linear margin."""
    rng = np.random.RandomState(seed)
    n_half = n_samples // 2

    # Class 0 centered at (-1.5, -1.0)
    X0 = rng.randn(n_half, 2) * 0.5 + np.array([-1.5, -1.0])
    y0 = np.zeros(n_half)

    # Class 1 centered at (1.5, 1.0)
    X1 = rng.randn(n_half, 2) * 0.5 + np.array([1.5, 1.0])
    y1 = np.ones(n_half)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    return X, y


def demonstrate_error_correction(seed: int = 42) -> Dict[str, Any]:
    """Demonstrate Delta Rule learning dynamics on linearly separable data."""
    X, y = generate_linearly_separable_data(n_samples=200, seed=seed)
    learner = DeltaRuleLearner(n_features=2, lr=0.04, seed=seed)
    history = learner.fit(X, y, epochs=50, seed=seed)

    return {
        "X": X,
        "y": y,
        "history": history,
    }


def plot_error_correction_figures(
    results: Dict[str, Any],
    out_dir: Path | str = "artifacts/figures/learning_paradigms/",
) -> Path:
    """Generate plots showing MSE convergence and decision boundary movement before and after learning."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_file = out_path / "error_correction_delta_rule_training.png"

    X = results["X"]
    y = results["y"]
    hist = results["history"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Loss and Classification Error Curves
    epochs = range(1, len(hist["loss_history"]) + 1)
    ax1.plot(epochs, hist["loss_history"], color="#d62728", linewidth=2.2, label=r"MSE Loss: $\frac{1}{2N}\sum(t - y)^2$")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Mean Squared Error", fontsize=10, color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, hist["error_count_history"], color="#1f77b4", linewidth=2.0, linestyle="--", label="Misclassified Samples")
    ax1_twin.set_ylabel("Misclassified Count", fontsize=10, color="#1f77b4")
    ax1_twin.tick_params(axis="y", labelcolor="#1f77b4")

    ax1.set_title("Delta Rule / LMS Convergence Dynamics", fontsize=11, fontweight="bold")

    # 2. Decision Boundary: Before vs After Learning
    ax2.scatter(X[y == 0, 0], X[y == 0, 1], color="#377eb8", alpha=0.7, label="Class 0 ($t=0$)", s=35)
    ax2.scatter(X[y == 1, 0], X[y == 1, 1], color="#e41a1c", alpha=0.7, label="Class 1 ($t=1$)", s=35)

    x_vals = np.linspace(-3.5, 3.5, 100)

    # Initial boundary: w1 * x1 + w2 * x2 + b = 0.5 (threshold) -> x2 = (0.5 - b - w1*x1) / w2
    w_init, b_init = hist["initial_w"], hist["initial_b"]
    if abs(w_init[1]) > 1e-5:
        y_init = (0.5 - b_init - w_init[0] * x_vals) / w_init[1]
        ax2.plot(x_vals, y_init, color="black", linestyle=":", linewidth=1.8, label="Initial Boundary (Untrained)")

    # Final boundary
    w_fin, b_fin = hist["final_w"], hist["final_b"]
    if abs(w_fin[1]) > 1e-5:
        y_fin = (0.5 - b_fin - w_fin[0] * x_vals) / w_fin[1]
        ax2.plot(x_vals, y_fin, color="#2ca02c", linestyle="-", linewidth=2.4, label="Converged Boundary (Delta Rule)")

    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(-3.5, 3.5)
    ax2.set_title("Decision Boundary Before vs After Supervised Learning", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Feature $x_1$", fontsize=10)
    ax2.set_ylabel("Feature $x_2$", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(fontsize=9, loc="upper left", framealpha=0.92)

    plt.suptitle("Error-Correction Learning: Target Feedback Driving Parameter Alignment", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(fig_file, bbox_inches="tight")
    plt.close(fig)

    return fig_file
