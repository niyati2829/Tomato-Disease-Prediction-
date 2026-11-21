"""Rosenblatt's Perceptron Algorithm from Scratch.

Implements:
1. Classical linear threshold classifier with Heaviside step function.
2. Perceptron learning rule:
       w <- w + eta * (y - y_hat) * x
       b <- b + eta * (y - y_hat)
3. Convergence proof demonstration on linearly separable datasets (AND, OR, 2D blobs).
4. Demonstration of the fundamental linear separability limitation on XOR.
5. Decision boundary visualization.

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


class Perceptron:
    """Single-layer Perceptron with Heaviside step activation."""

    def __init__(
        self,
        n_features: int | None = None,
        n_inputs: int | None = None,
        lr: float | None = None,
        learning_rate: float | None = None,
        max_epochs: int = 100,
        seed: int = 42,
    ) -> None:
        num_inputs = n_features if n_features is not None else n_inputs
        if num_inputs is None:
            raise ValueError("Must provide either n_features or n_inputs.")
        self.lr = learning_rate if learning_rate is not None else (lr if lr is not None else 0.1)
        self.learning_rate = self.lr
        self.max_epochs = max_epochs
        rng = np.random.RandomState(seed)
        self.weights = rng.randn(num_inputs) * 0.05
        self.bias = 0.0
        self.converged = False
        self.convergence_epoch: int | None = None
        self.epochs_to_converge = max_epochs
        self.error_history: List[int] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels {0, 1}."""
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.0, 1, 0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit Perceptron weights using the online learning rule."""
        N = X.shape[0]
        self.error_history = []
        self.converged = False

        for epoch in range(1, self.max_epochs + 1):
            misclassified = 0
            for i in range(N):
                xi = X[i]
                target = y[i]
                pred = 1 if (np.dot(xi, self.weights) + self.bias) >= 0.0 else 0
                error = target - pred

                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    misclassified += 1

            self.error_history.append(misclassified)
            if misclassified == 0:
                self.converged = True
                self.convergence_epoch = epoch
                self.epochs_to_converge = epoch
                break

        return {
            "converged": self.converged,
            "epochs_trained": len(self.error_history),
            "final_weights": self.weights.tolist(),
            "final_bias": float(self.bias),
            "total_errors_final_epoch": self.error_history[-1],
        }


def plot_perceptron_decision_boundaries(output_path: Path | str) -> Path:
    """Plot Perceptron decision boundaries on linearly separable (OR/AND) vs non-separable (XOR)."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

    # 1. AND Gate (Linearly Separable)
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_and = np.array([0, 0, 0, 1])
    p_and = Perceptron(n_features=2, lr=0.1, max_epochs=50)
    p_and.fit(X_and, y_and)

    # 2. OR Gate (Linearly Separable)
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_or = np.array([0, 1, 1, 1])
    p_or = Perceptron(n_features=2, lr=0.1, max_epochs=50)
    p_or.fit(X_or, y_or)

    # 3. XOR Gate (Non-Linearly Separable)
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([0, 1, 1, 0])
    p_xor = Perceptron(n_features=2, lr=0.1, max_epochs=50)
    p_xor.fit(X_xor, y_xor)

    tasks = [
        ("AND Gate", X_and, y_and, p_and, axes[0]),
        ("OR Gate", X_or, y_or, p_or, axes[1]),
        ("XOR Gate (Minsky Limitation)", X_xor, y_xor, p_xor, axes[2]),
    ]

    # Coordinate meshgrid for decision region background
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for title, X, y, model, ax in tasks:
        # Decision boundary contours
        preds_grid = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, preds_grid, alpha=0.25, cmap="coolwarm")

        # Plot boundary line w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b) / w2
        if abs(model.weights[1]) > 1e-6:
            line_x = np.linspace(x_min, x_max, 100)
            line_y = -(model.weights[0] * line_x + model.bias) / model.weights[1]
            ax.plot(line_x, line_y, "k--", linewidth=2, label="Decision Boundary")

        # Plot data points
        for cl, marker, color, label in [(0, "o", "#1976d2", "Class 0"), (1, "s", "#d32f2f", "Class 1")]:
            mask = (y == cl)
            ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker, s=120, edgecolors="k", linewidth=1.5, label=label)

        status = f"Converged (Epoch {model.epochs_to_converge})" if model.converged else "Failed to Converge (Oscillation)"
        ax.set_title(f"{title}\n{status}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("$x_1$", fontsize=10)
        ax.set_ylabel("$x_2$", fontsize=10)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=True, fontsize=8, loc="upper left")

    plt.suptitle("Perceptron Decision Boundaries: Linear Separability vs. XOR Limitation", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
