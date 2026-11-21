"""Hebbian Learning & Oja's Rule Implementations (REQ-06).

Implements:
1. Basic Hebbian Learning:
       Delta w = eta * y * x   (y = w^T x)
   Demonstrates unsupervised correlation-based association and catastrophic unbounded weight growth.

2. Oja's Modified Rule (1982):
       Delta w = eta * y * (x - y * w)
   Derived as a Taylor-series approximation to normalized Hebbian learning.
   Demonstrates weight stabilization (||w||_2 -> 1) and exact extraction of the first principal component (PCA).

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


class BasicHebbianLearner:
    """Basic unsupervised Hebbian learning neuron.

    Weight update equation:
        Delta w = eta * y * x   where y = w^T x
    """

    def __init__(self, n_features: int, lr: float = 0.005, seed: int = 42) -> None:
        self.lr = lr
        rng = np.random.RandomState(seed)
        self.w = rng.randn(n_features) * 0.1
        # Normalize initially for fair start
        self.w /= np.linalg.norm(self.w) + 1e-12

    def predict(self, x: np.ndarray) -> float:
        """Compute linear activation y = w^T x."""
        return float(np.dot(self.w, x))

    def train_step(self, x: np.ndarray) -> Tuple[float, float]:
        """Execute single Hebbian update step.

        Returns:
            Tuple of (output y, updated weight norm ||w||_2).
        """
        y = self.predict(x)
        delta_w = self.lr * y * x
        self.w += delta_w
        norm = float(np.linalg.norm(self.w))
        return y, norm


class OjaLearner:
    """Oja's modified Hebbian learning neuron with weight normalization feedback.

    Weight update equation:
        Delta w = eta * y * (x - y * w)   where y = w^T x
    """

    def __init__(self, n_features: int, lr: float = 0.01, seed: int = 42) -> None:
        self.lr = lr
        rng = np.random.RandomState(seed)
        self.w = rng.randn(n_features) * 0.1
        self.w /= np.linalg.norm(self.w) + 1e-12

    def predict(self, x: np.ndarray) -> float:
        """Compute linear activation y = w^T x."""
        return float(np.dot(self.w, x))

    def train_step(self, x: np.ndarray) -> Tuple[float, float]:
        """Execute single Oja update step.

        Returns:
            Tuple of (output y, updated weight norm ||w||_2).
        """
        y = self.predict(x)
        # Oja's weight stabilization term: - eta * y^2 * w
        delta_w = self.lr * y * (x - y * self.w)
        self.w += delta_w
        norm = float(np.linalg.norm(self.w))
        return y, norm


def generate_correlated_gaussian_data(
    n_samples: int = 400,
    angle_deg: float = 35.0,
    var_major: float = 4.0,
    var_minor: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 2D zero-centered correlated data along a known principal axis."""
    rng = np.random.RandomState(seed)
    theta = np.radians(angle_deg)
    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    # Uncorrelated scales
    scales = np.diag([np.sqrt(var_major), np.sqrt(var_minor)])
    Z = rng.randn(n_samples, 2)
    X = Z @ scales @ R.T
    # Center explicitly
    X -= np.mean(X, axis=0)

    # True principal component from covariance matrix
    C = (X.T @ X) / n_samples
    eigvals, eigvecs = np.linalg.eigh(C)
    idx_max = int(np.argmax(eigvals))
    true_pc1 = eigvecs[:, idx_max]
    # Ensure consistent sign
    if true_pc1[0] < 0:
        true_pc1 = -true_pc1

    return X, true_pc1


def compare_hebbian_and_oja(
    n_iterations: int = 1500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare Basic Hebbian vs Oja's rule on identical correlated data."""
    X, true_pc1 = generate_correlated_gaussian_data(n_samples=500, angle_deg=35.0, seed=seed)
    n_samples = len(X)

    hebb = BasicHebbianLearner(n_features=2, lr=0.005, seed=seed)
    oja = OjaLearner(n_features=2, lr=0.008, seed=seed)

    hebb_norms = [float(np.linalg.norm(hebb.w))]
    oja_norms = [float(np.linalg.norm(oja.w))]
    oja_cos_sims = []

    rng = np.random.RandomState(seed)
    for step in range(n_iterations):
        sample = X[rng.randint(0, n_samples)]
        _, h_norm = hebb.train_step(sample)
        _, o_norm = oja.train_step(sample)

        hebb_norms.append(h_norm)
        oja_norms.append(o_norm)

        # Cosine similarity with true PC1
        oja_dir = oja.w / (np.linalg.norm(oja.w) + 1e-12)
        cos_sim = abs(float(np.dot(oja_dir, true_pc1)))
        oja_cos_sims.append(cos_sim)

    # Final alignments
    final_oja_dir = oja.w / np.linalg.norm(oja.w)
    final_cos_sim = abs(float(np.dot(final_oja_dir, true_pc1)))

    return {
        "X": X,
        "true_pc1": true_pc1,
        "final_hebb_w": hebb.w.copy(),
        "final_hebb_norm": hebb_norms[-1],
        "final_oja_w": oja.w.copy(),
        "final_oja_norm": oja_norms[-1],
        "final_oja_dir": final_oja_dir,
        "final_cosine_similarity": final_cos_sim,
        "hebb_norms": np.array(hebb_norms),
        "oja_norms": np.array(oja_norms),
        "oja_cos_sims": np.array(oja_cos_sims),
    }


def plot_hebbian_and_oja_figures(
    results: Dict[str, Any],
    out_dir: Path | str = "artifacts/figures/learning_paradigms/",
) -> Tuple[Path, Path]:
    """Generate plots showing weight explosion vs Oja stabilization, and PCA direction alignment."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig1_file = out_path / "hebbian_vs_oja_weight_growth.png"
    fig2_file = out_path / "oja_pca_direction_alignment.png"

    # Figure 1: Weight norm evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    hebb_norms = results["hebb_norms"]
    oja_norms = results["oja_norms"]
    oja_sims = results["oja_cos_sims"]

    ax1.plot(hebb_norms, color="#d62728", linewidth=2.2, label=r"Basic Hebbian ($\Delta w = \eta y x$)")
    ax1.plot(oja_norms, color="#2ca02c", linewidth=2.2, label=r"Oja's Rule ($\Delta w = \eta y(x - yw)$)")
    ax1.axhline(1.0, color="black", linestyle="--", alpha=0.6, label=r"Theoretical Unit Norm ($\|w\|_2 = 1.0$)")
    ax1.set_title("Weight Magnitude Trajectory: Explosion vs Stability", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Training Iteration", fontsize=10)
    ax1.set_ylabel(r"Weight Vector Norm $\|w\|_2$", fontsize=10)
    ax1.set_yscale("log")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(fontsize=9, loc="upper left")

    ax2.plot(oja_sims, color="#1f77b4", linewidth=2.0, label=r"Cosine Similarity $|\cos(\theta)|$ with True PC1")
    ax2.axhline(1.0, color="#2ca02c", linestyle="--", alpha=0.7, label="Perfect PCA Alignment (1.0)")
    ax2.set_title("Oja's Rule Convergence to Principal Component (PCA)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Training Iteration", fontsize=10)
    ax2.set_ylabel("Absolute Cosine Similarity", fontsize=10)
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(fontsize=9, loc="lower right")

    plt.suptitle("Hebbian Learning Mechanics: Mathematical Unbounded Growth vs Oja Normalization", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(fig1_file, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: 2D Data Scatter with True PC1 vs Learned Oja vector
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    X = results["X"]
    true_pc = results["true_pc1"]
    oja_dir = results["final_oja_dir"]

    ax.scatter(X[:, 0], X[:, 1], color="#a6cee3", alpha=0.55, edgecolors="none", s=22, label="Correlated Input Samples")

    # Scale vectors for visual clarity
    scale = 3.5
    ax.arrow(0, 0, true_pc[0] * scale, true_pc[1] * scale, color="#e41a1c", width=0.08, head_width=0.25,
             length_includes_head=True, label=f"True Principal Component 1 (Eigendecomposition)", zorder=6)
    ax.arrow(0, 0, oja_dir[0] * scale * 0.95, oja_dir[1] * scale * 0.95, color="#33a02c", width=0.05, head_width=0.22,
             linestyle="--", length_includes_head=True, label=f"Oja Learned Weight Vector (Sim: {results['final_cosine_similarity']:.4f})", zorder=7)

    ax.set_title(rf"Oja's Rule Principal Component Extraction ($|\cos(\theta)|={results['final_cosine_similarity']:.4f}$)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature $x_1$", fontsize=11)
    ax.set_ylabel("Feature $x_2$", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.axis("equal")
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.92)

    plt.tight_layout()
    plt.savefig(fig2_file, bbox_inches="tight")
    plt.close(fig)

    return fig1_file, fig2_file

