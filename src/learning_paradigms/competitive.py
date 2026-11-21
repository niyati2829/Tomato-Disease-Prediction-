"""Competitive Learning & Kohonen Self-Organizing Maps (REQ-06).

Implements from scratch:
1. Winner-Takes-All (WTA) Competitive Learning:
       k* = argmin_k ||x - w_k||_2
       Delta w_{k*} = eta * (x - w_{k*})
2. Kohonen Self-Organizing Map (SOM):
       Topology-preserving 2D grid with Gaussian neighborhood coupling:
       h_{c, i}(t) = exp(- ||r_c - r_i||^2 / (2 * sigma(t)^2))
       Delta w_i = eta(t) * h_{c, i}(t) * (x - w_i)

Demonstrates unsupervised clustering and topological manifold mapping on synthetic multi-cluster data.

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


class CompetitiveLearningWTA:
    """Winner-Takes-All (WTA) Competitive Learning Clustering Network."""

    def __init__(self, n_features: int, n_clusters: int, lr: float = 0.05, seed: int = 42) -> None:
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.lr = lr
        rng = np.random.RandomState(seed)
        # Initialize prototypes from standard normal
        self.weights = rng.randn(n_clusters, n_features)

    def find_bmu(self, x: np.ndarray) -> int:
        """Find the Best Matching Unit (BMU) index with minimum Euclidean distance."""
        dists = np.linalg.norm(self.weights - x, axis=1)
        return int(np.argmin(dists))

    def train(self, X: np.ndarray, epochs: int = 40, seed: int = 42) -> Dict[str, Any]:
        """Train competitive prototypes and record quantization error."""
        rng = np.random.RandomState(seed)
        N = len(X)
        history_error = []
        initial_weights = self.weights.copy()

        for epoch in range(epochs):
            indices = rng.permutation(N)
            for idx in indices:
                x = X[idx]
                bmu = self.find_bmu(x)
                # Winner takes all update
                self.weights[bmu] += self.lr * (x - self.weights[bmu])

            # Quantization error across dataset
            bmu_indices = [self.find_bmu(X[i]) for i in range(N)]
            assigned_prototypes = self.weights[bmu_indices]
            mse = float(np.mean(np.sum((X - assigned_prototypes) ** 2, axis=1)))
            history_error.append(mse)

        final_clusters = [self.find_bmu(X[i]) for i in range(N)]

        return {
            "initial_weights": initial_weights,
            "final_weights": self.weights.copy(),
            "history_error": history_error,
            "cluster_assignments": np.array(final_clusters),
        }


class KohonenSOM:
    """Kohonen Self-Organizing Map with 2D Grid Topology."""

    def __init__(
        self,
        grid_rows: int = 6,
        grid_cols: int = 6,
        n_features: int = 2,
        lr_initial: float = 0.3,
        sigma_initial: float | None = None,
        seed: int = 42,
    ) -> None:
        self.rows = grid_rows
        self.cols = grid_cols
        self.n_neurons = grid_rows * grid_cols
        self.n_features = n_features
        self.lr_0 = lr_initial
        self.sigma_0 = sigma_initial if sigma_initial is not None else max(grid_rows, grid_cols) / 2.0

        rng = np.random.RandomState(seed)
        self.weights = rng.uniform(-2.0, 2.0, size=(self.n_neurons, n_features))

        # 2D coordinates of neurons on the grid
        self.grid_coords = np.zeros((self.n_neurons, 2))
        for r in range(grid_rows):
            for c in range(grid_cols):
                idx = r * grid_cols + c
                self.grid_coords[idx] = [r, c]

    def find_bmu(self, x: np.ndarray) -> int:
        """Find index of neuron with minimum Euclidean distance to input x."""
        dists = np.linalg.norm(self.weights - x, axis=1)
        return int(np.argmin(dists))

    def train(self, X: np.ndarray, n_iterations: int = 800, seed: int = 42) -> Dict[str, Any]:
        """Train SOM with exponentially decaying learning rate and Gaussian neighborhood radius."""
        rng = np.random.RandomState(seed)
        N = len(X)
        initial_weights = self.weights.copy()
        time_constant = n_iterations / np.log(self.sigma_0 + 1e-5)

        for t in range(n_iterations):
            # Sample random data point
            x = X[rng.randint(0, N)]
            bmu = self.find_bmu(x)
            bmu_coord = self.grid_coords[bmu]

            # Decaying learning rate and neighborhood radius
            eta_t = self.lr_0 * np.exp(-t / n_iterations)
            sigma_t = self.sigma_0 * np.exp(-t / time_constant)
            sigma_t_sq = max(sigma_t ** 2, 1e-4)

            # Distances on the 2D grid from BMU to all neurons
            grid_dists_sq = np.sum((self.grid_coords - bmu_coord) ** 2, axis=1)

            # Gaussian neighborhood influence h_{c, i}
            h_ci = np.exp(-grid_dists_sq / (2.0 * sigma_t_sq))  # Shape (n_neurons,)

            # Vectorized neighborhood update
            delta = eta_t * h_ci[:, np.newaxis] * (x - self.weights)
            self.weights += delta

        return {
            "initial_weights": initial_weights,
            "final_weights": self.weights.copy(),
            "grid_coords": self.grid_coords,
            "rows": self.rows,
            "cols": self.cols,
        }


def generate_multicluster_data(
    n_samples_per_cluster: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 4 distinct 2D Gaussian clusters."""
    rng = np.random.RandomState(seed)
    centers = [
        [-2.5, 2.5],
        [2.5, 2.5],
        [-2.5, -2.5],
        [2.5, -2.5],
    ]
    clusters = []
    labels = []
    for idx, c in enumerate(centers):
        pts = rng.randn(n_samples_per_cluster, 2) * 0.45 + np.array(c)
        clusters.append(pts)
        labels.extend([idx] * n_samples_per_cluster)

    X = np.vstack(clusters)
    return X, np.array(labels)


def demonstrate_competitive_and_som(seed: int = 42) -> Dict[str, Any]:
    """Run both Competitive Learning (WTA) and Kohonen SOM on multi-cluster data."""
    X, labels = generate_multicluster_data(n_samples_per_cluster=100, seed=seed)

    # 1. Competitive WTA
    wta = CompetitiveLearningWTA(n_features=2, n_clusters=4, lr=0.08, seed=seed)
    wta_res = wta.train(X, epochs=40, seed=seed)

    # 2. Kohonen SOM (5x5 grid)
    som = KohonenSOM(grid_rows=5, grid_cols=5, n_features=2, lr_initial=0.4, seed=seed)
    som_res = som.train(X, n_iterations=1000, seed=seed)

    return {
        "X": X,
        "labels": labels,
        "wta": wta_res,
        "som": som_res,
    }


def plot_competitive_and_som_figures(
    results: Dict[str, Any],
    out_dir: Path | str = "artifacts/figures/learning_paradigms/",
) -> Tuple[Path, Path]:
    """Generate figures showing WTA cluster convergence and SOM topological grid organization."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig1_file = out_path / "competitive_learning_clusters.png"
    fig2_file = out_path / "som_topological_organization.png"

    X = results["X"]
    wta = results["wta"]
    som = results["som"]

    # Figure 1: Competitive Learning WTA
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Left: initial vs final prototypes
    ax1.scatter(X[:, 0], X[:, 1], c=wta["cluster_assignments"], cmap="tab10", alpha=0.4, s=25, label="Data Samples")
    init_w = wta["initial_weights"]
    final_w = wta["final_weights"]
    ax1.scatter(init_w[:, 0], init_w[:, 1], color="black", marker="x", s=90, linewidth=2.0, label="Initial Prototypes")
    ax1.scatter(final_w[:, 0], final_w[:, 1], color="red", marker="*", s=220, edgecolors="black", label="Converged Prototypes", zorder=10)

    # Connect initial to final
    for i in range(len(init_w)):
        ax1.annotate("", xy=final_w[i], xytext=init_w[i], arrowprops=dict(arrowstyle="->", color="black", lw=1.5, ls="--"))

    ax1.set_title("Winner-Takes-All Prototype Movement", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Feature $x_1$", fontsize=10)
    ax1.set_ylabel("Feature $x_2$", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(fontsize=9, loc="upper right")

    # Right: Quantization error curve
    ax2.plot(wta["history_error"], color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    ax2.set_title("WTA Mean Quantization Error (MSE vs Epoch)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Mean Squared Quantization Error", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Competitive Learning: Unsupervised Cluster Prototype Centroid Discovery", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(fig1_file, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Kohonen Self-Organizing Map (SOM) Lattice Topography
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    ax.scatter(X[:, 0], X[:, 1], color="#cccccc", alpha=0.6, s=20, label="Input Data Points")

    weights = som["final_weights"]
    rows, cols = som["rows"], som["cols"]

    # Draw grid mesh connections between adjacent neurons in the lattice
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            # Horizontal connection to (r, c+1)
            if c + 1 < cols:
                idx_right = r * cols + (c + 1)
                ax.plot([weights[idx, 0], weights[idx_right, 0]], [weights[idx, 1], weights[idx_right, 1]], color="#1f77b4", lw=1.2, alpha=0.85)
            # Vertical connection to (r+1, c)
            if r + 1 < rows:
                idx_down = (r + 1) * cols + c
                ax.plot([weights[idx, 0], weights[idx_down, 0]], [weights[idx, 1], weights[idx_down, 1]], color="#1f77b4", lw=1.2, alpha=0.85)

    ax.scatter(weights[:, 0], weights[:, 1], color="#d62728", s=65, edgecolors="black", zorder=5, label="SOM Neurons (5x5 Lattice)")
    ax.set_title("Kohonen Self-Organizing Map: Topology-Preserving Manifold Mesh", fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature $x_1$", fontsize=11)
    ax.set_ylabel("Feature $x_2$", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.92)

    plt.tight_layout()
    plt.savefig(fig2_file, bbox_inches="tight")
    plt.close(fig)

    return fig1_file, fig2_file

