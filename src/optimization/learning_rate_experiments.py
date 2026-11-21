"""Learning Rate Dynamics & Stability Experiments (REQ-04).

Demonstrates:
1. Too small learning rate (crawling / slow convergence)
2. Appropriate learning rate (rapid, stable monotonic convergence)
3. Too large learning rate (oscillations, overshoot, or explosive divergence)

Author: Crop Disease Detection Project Team
"""

import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.optimization.gradient_descent import ConvexRegressionObjective, generate_synthetic_regression_data


def run_learning_rate_experiments(
    learning_rates: List[float] = [0.01, 0.5, 1.4, 2.1],
    n_steps: int = 35,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute gradient descent under varied learning rates to evaluate convergence stability."""
    X, y, _ = generate_synthetic_regression_data(n_samples=200, n_features=4, seed=seed)
    obj = ConvexRegressionObjective(X, y)

    # Initial weights away from optimum
    rng = np.random.RandomState(seed)
    w_init = rng.randn(4) * 2.5

    results: Dict[str, Any] = {}

    for lr in learning_rates:
        w = w_init.copy()
        losses = [obj.compute_loss(w)]
        grad_norms = [float(np.linalg.norm(obj.compute_gradient(w)))]
        diverged = False

        for step in range(n_steps):
            g = obj.compute_gradient(w)
            w = w - lr * g
            loss = obj.compute_loss(w)

            if np.isnan(loss) or np.isinf(loss) or loss > 1e5:
                diverged = True
                losses.append(float("inf"))
                grad_norms.append(float("inf"))
                break

            losses.append(loss)
            grad_norms.append(float(np.linalg.norm(g)))

        # Classified by spectral limit: lambda_max ~ 1.13, 2/lambda_max ~ 1.77
        if lr <= 0.05:
            regime = "Under-damped / Too Slow"
        elif lr <= 0.8:
            regime = "Optimal / Stable"
        elif lr <= 1.7:
            regime = "Oscillatory / Sub-optimal"
        else:
            regime = "Excessive / Divergent"

        results[f"lr_{lr}"] = {
            "lr": lr,
            "regime": regime,
            "losses": losses,
            "grad_norms": grad_norms,
            "final_loss": losses[-1],
            "diverged": diverged,
        }

    return results


def plot_learning_rate_comparison(
    results: Dict[str, Any],
    output_path: Path | str = "artifacts/figures/optimization/learning_rate_comparison.png",
) -> Path:
    """Plot loss progression curves for small, optimal, and divergent learning rates."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    palette = {
        "Under-damped / Too Slow": "#1f77b4",
        "Optimal / Stable": "#2ca02c",
        "Oscillatory / Sub-optimal": "#ff7f0e",
        "Excessive / Divergent": "#d62728",
    }

    for key, data in results.items():
        lr = data["lr"]
        regime = data["regime"]
        losses = np.array(data["losses"])
        color = palette.get(regime, "#333333")

        # Clip excessive values for visualization sanity
        clipped_losses = np.clip(losses, 1e-4, 1e4)
        ax1.plot(
            range(len(clipped_losses)),
            clipped_losses,
            label=rf"$\eta = {lr}$ ({regime})",
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=3.5,
        )

        grad_norms = np.array(data["grad_norms"])
        clipped_grads = np.clip(grad_norms, 1e-4, 1e4)
        ax2.plot(
            range(len(clipped_grads)),
            clipped_grads,
            label=rf"$\eta = {lr}$",
            color=color,
            linewidth=2.0,
        )

    ax1.set_yscale("log")
    ax1.set_title(r"Loss Trajectory Under Varied Learning Rates ($\eta$)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Optimization Step", fontsize=11)
    ax1.set_ylabel("Mean Squared Error Loss (Log Scale)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    ax2.set_yscale("log")
    ax2.set_title(r"Gradient Norm ($\|\nabla L\|_2$) vs Steps", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Optimization Step", fontsize=11)
    ax2.set_ylabel(r"$\|\nabla L\|_2$ (Log Scale)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    plt.suptitle("Learning Rate Dynamics: Undershooting vs Stable vs Divergent Regimes", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
