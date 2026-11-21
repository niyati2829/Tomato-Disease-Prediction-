"""Loss Landscape & Optimization Trajectory Visualizations (REQ-04).

Generates:
1. 2D Contour visualization of an anisotropic loss valley with overlaid optimizer trajectories.
2. 3D Surface visualization of the identical landscape with 3D trajectories.
3. Multi-optimizer convergence comparison plot (Loss vs Steps).

Author: Crop Disease Detection Project Team
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from src.optimization.optimizers import (
    AdamOptimizer,
    GradientDescentOptimizer,
    MomentumOptimizer,
    NesterovOptimizer,
    optimize_arbitrary_function,
)


def anisotropic_valley_loss(w: np.ndarray, condition_number: float = 10.0) -> float:
    """Anisotropic convex quadratic valley: f(x, y) = 0.5 * (x^2 + condition_number * y^2).

    Simulates pathological curvature where steepness along y is much greater than x,
    inducing zig-zag oscillations in vanilla gradient descent.
    """
    x, y = w[0], w[1]
    return float(0.5 * (x**2 + condition_number * (y**2)))


def anisotropic_valley_gradient(w: np.ndarray, condition_number: float = 10.0) -> np.ndarray:
    """Analytical gradient of anisotropic valley: grad(x, y) = [x, condition_number * y]."""
    x, y = w[0], w[1]
    return np.array([x, condition_number * y], dtype=float)


def compute_optimizer_trajectories(
    w_start: Tuple[float, float] = (-4.0, 2.5),
    n_steps: int = 50,
    lr: float = 0.15,
    condition_number: float = 10.0,
) -> Dict[str, Dict[str, Any]]:
    """Compute trajectory for GD, Momentum, Nesterov, and Adam on the anisotropic valley."""
    f = lambda w: anisotropic_valley_loss(w, condition_number=condition_number)
    grad_f = lambda w: anisotropic_valley_gradient(w, condition_number=condition_number)
    w0 = np.array(w_start, dtype=float)

    optimizers = {
        "Gradient Descent": (GradientDescentOptimizer(lr=lr), "#e41a1c", "o"),
        "Momentum": (MomentumOptimizer(lr=lr * 0.7, momentum=0.85), "#377eb8", "s"),
        "Nesterov": (NesterovOptimizer(lr=lr * 0.7, momentum=0.85), "#4daf4a", "^"),
        "Adam": (AdamOptimizer(lr=0.35, beta1=0.9, beta2=0.999), "#984ea3", "D"),
    }

    results = {}
    for name, (opt, color, marker) in optimizers.items():
        res = optimize_arbitrary_function(f, grad_f, opt, w0, n_steps=n_steps)
        res["color"] = color
        res["marker"] = marker
        results[name] = res

    return results


def plot_2d_contour_trajectories(
    trajectories: Dict[str, Dict[str, Any]],
    output_path: Path | str = "artifacts/figures/optimization/loss_landscape_2d_contours.png",
    condition_number: float = 10.0,
) -> Path:
    """Generate 2D contour plot overlaid with optimizer trajectories."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Grid for contour
    x = np.linspace(-5.0, 5.0, 400)
    y = np.linspace(-3.5, 3.5, 400)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (X**2 + condition_number * (Y**2))

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

    # Log-spaced contours to highlight valley structure
    levels = np.geomspace(0.05, 75.0, 30)
    cp = ax.contour(X, Y, Z, levels=levels, cmap="Blues_r", alpha=0.6, linewidths=0.9)
    ax.clabel(cp, inline=True, fontsize=7, fmt="%.1f")

    # Global minimum
    ax.scatter([0.0], [0.0], color="gold", s=180, marker="*", edgecolors="black", zorder=10, label="Global Min $(0, 0)$")

    # Overlay optimizer paths
    for name, data in trajectories.items():
        path = data["trajectory"]
        color = data["color"]
        marker = data["marker"]
        ax.plot(path[:, 0], path[:, 1], label=name, color=color, linewidth=2.0, alpha=0.9)
        # Mark subsampled points for clarity
        sub = np.linspace(0, len(path) - 1, 10, dtype=int)
        ax.scatter(path[sub, 0], path[sub, 1], color=color, s=28, marker=marker, zorder=5)

    ax.scatter([-4.0], [2.5], color="black", s=100, marker="X", zorder=10, label="Start $(-4.0, 2.5)$")

    ax.set_title(
        rf"2D Loss Contours & Optimizer Trajectories (Anisotropic Valley, $\kappa = {condition_number:.0f}$)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Parameter $w_1$ (Gentle Curvature)", fontsize=11)
    ax.set_ylabel("Parameter $w_2$ (Steep Curvature)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_3d_surface_trajectories(
    trajectories: Dict[str, Dict[str, Any]],
    output_path: Path | str = "artifacts/figures/optimization/loss_landscape_3d_surface.png",
    condition_number: float = 10.0,
) -> Path:
    """Generate 3D surface plot overlaid with 3D optimizer trajectories."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-4.5, 4.5, 120)
    y = np.linspace(-3.0, 3.0, 120)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (X**2 + condition_number * (Y**2))

    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    # Wireframe / surface
    surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", alpha=0.45, edgecolor="none", antialiased=True)

    # Plot paths on surface
    for name, data in trajectories.items():
        path = data["trajectory"]
        color = data["color"]
        z_path = [anisotropic_valley_loss(p, condition_number=condition_number) for p in path]
        ax.plot(path[:, 0], path[:, 1], z_path, label=name, color=color, linewidth=2.5, alpha=0.95)

    # Global min & start
    ax.scatter([0], [0], [0], color="gold", s=180, marker="*", edgecolors="black", label="Optimum $(0, 0, 0)$")
    z_start = anisotropic_valley_loss(np.array([-4.0, 2.5]), condition_number=condition_number)
    ax.scatter([-4.0], [2.5], [z_start], color="black", s=90, marker="X", label="Start $(-4, 2.5)$")

    ax.set_title("3D Loss Surface with Optimization Paths", fontsize=13, fontweight="bold", pad=20)
    ax.set_xlabel("$w_1$", fontsize=10)
    ax.set_ylabel("$w_2$", fontsize=10)
    ax.set_zlabel("Loss $f(w_1, w_2)$", fontsize=10)
    ax.view_init(elev=35, azim=-125)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.95), framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_optimizer_convergence_comparison(
    trajectories: Dict[str, Dict[str, Any]],
    output_path: Path | str = "artifacts/figures/optimization/optimizer_convergence_comparison.png",
) -> Path:
    """Generate convergence curves (loss and gradient norm vs step count)."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    for name, data in trajectories.items():
        losses = data["losses"]
        grad_norms = data["grad_norms"]
        color = data["color"]

        ax1.plot(range(len(losses)), losses, label=name, color=color, linewidth=2.2)
        ax2.plot(range(len(grad_norms)), grad_norms, label=name, color=color, linewidth=2.2)

    ax1.set_yscale("log")
    ax1.set_title("Optimizer Convergence Speed (Loss vs Steps)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Iteration Step", fontsize=11)
    ax1.set_ylabel("Loss (Log Scale)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9.5)

    ax2.set_yscale("log")
    ax2.set_title(r"Gradient Norm ($\|\nabla f\|_2$) vs Steps", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Iteration Step", fontsize=11)
    ax2.set_ylabel(r"$\|\nabla f\|_2$ (Log Scale)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9.5)

    plt.suptitle("First-Order Optimizers Benchmark on Pathological Curvature", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
