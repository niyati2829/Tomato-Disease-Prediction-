"""Energy-Based Probabilistic Learning: Restricted Boltzmann Machine (RBM) (REQ-06).

Implements from scratch:
1. Binary stochastic units: visible v in {0, 1}^V, hidden h in {0, 1}^H
2. Joint energy function:
       E(v, h) = - v^T W h - a^T v - b^T h
3. Conditional sigmoid probabilities:
       P(h_j = 1 | v) = sigmoid(sum_i v_i W_ij + b_j)
       P(v_i = 1 | h) = sigmoid(sum_j W_ij h_j + a_i)
4. 1-step Contrastive Divergence (CD-1) learning:
       Delta W = eta * (v_0 * h_0^T - v_1 * h_1^T)
5. Free energy calculation:
       F(v) = - a^T v - sum_j ln(1 + exp(v^T W_{.j} + b_j))

Contrasts deterministic energy minimization (Hopfield) with stochastic thermal sampling (Boltzmann).

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


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically clipped sigmoid activation."""
    z_clipped = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


class RestrictedBoltzmannMachine:
    """Restricted Boltzmann Machine with binary visible and hidden stochastic units."""

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        lr: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.n_v = n_visible
        self.n_h = n_hidden
        self.lr = lr
        self.rng = np.random.RandomState(seed)

        # Small Gaussian weights
        self.W = self.rng.randn(n_visible, n_hidden) * 0.1
        self.a = np.zeros(n_visible, dtype=float)  # Visible biases
        self.b = np.zeros(n_hidden, dtype=float)   # Hidden biases

    def sample_hidden(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute P(h=1|v) and sample stochastic binary hidden states."""
        p_h = sigmoid(v @ self.W + self.b)
        h_samples = (self.rng.rand(*p_h.shape) < p_h).astype(float)
        return p_h, h_samples

    def sample_visible(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute P(v=1|h) and sample stochastic binary visible states."""
        p_v = sigmoid(h @ self.W.T + self.a)
        v_samples = (self.rng.rand(*p_v.shape) < p_v).astype(float)
        return p_v, v_samples

    def compute_energy(self, v: np.ndarray, h: np.ndarray) -> float:
        """Compute joint scalar energy: E(v, h) = -v^T W h - a^T v - b^T h."""
        v_flat = np.asarray(v, dtype=float).ravel()
        h_flat = np.asarray(h, dtype=float).ravel()
        term_inter = float(v_flat @ self.W @ h_flat)
        term_vis = float(np.dot(self.a, v_flat))
        term_hid = float(np.dot(self.b, h_flat))
        return -(term_inter + term_vis + term_hid)

    def compute_free_energy(self, V: np.ndarray) -> np.ndarray:
        """Compute analytical free energy: F(v) = -a^T v - sum_j ln(1 + exp(v^T W_j + b_j))."""
        V = np.atleast_2d(V)
        linear_hid = V @ self.W + self.b
        vis_term = V @ self.a
        hid_term = np.sum(np.log(1.0 + np.exp(np.clip(linear_hid, -40.0, 40.0))), axis=1)
        return -(vis_term + hid_term)

    def train_step_cd1(self, v0: np.ndarray) -> Tuple[float, float]:
        """Execute single batch CD-1 (Contrastive Divergence) update."""
        N = len(v0)

        # 1. Positive Phase (data-driven)
        p_h0, h0 = self.sample_hidden(v0)

        # 2. Negative Phase (model reconstruction via 1-step Gibbs sampling)
        p_v1, v1 = self.sample_visible(h0)
        p_h1, _ = self.sample_hidden(v1)

        # 3. Gradients
        pos_grad = (v0.T @ p_h0) / N
        neg_grad = (v1.T @ p_h1) / N

        dW = pos_grad - neg_grad
        da = np.mean(v0 - v1, axis=0)
        db = np.mean(p_h0 - p_h1, axis=0)

        self.W += self.lr * dW
        self.a += self.lr * da
        self.b += self.lr * db

        reconstruction_error = float(np.mean((v0 - p_v1) ** 2))
        avg_free_energy = float(np.mean(self.compute_free_energy(v0)))

        return reconstruction_error, avg_free_energy

    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 16) -> Dict[str, Any]:
        """Train RBM over multiple epochs and track reconstruction error."""
        N = len(X)
        history_recon_error = []
        history_free_energy = []

        for epoch in range(epochs):
            indices = self.rng.permutation(N)
            epoch_errors = []
            epoch_energies = []

            for start in range(0, N, batch_size):
                batch_idx = indices[start : start + batch_size]
                v_batch = X[batch_idx]
                err, fe = self.train_step_cd1(v_batch)
                epoch_errors.append(err)
                epoch_energies.append(fe)

            history_recon_error.append(float(np.mean(epoch_errors)))
            history_free_energy.append(float(np.mean(epoch_energies)))

        return {
            "reconstruction_error": np.array(history_recon_error),
            "free_energy": np.array(history_free_energy),
        }


def generate_structured_binary_patterns(n_samples: int = 300, seed: int = 42) -> np.ndarray:
    """Generate synthetic 4x4 (16-bit) structured binary patterns: horizontal & vertical bars."""
    rng = np.random.RandomState(seed)
    dataset = []

    for _ in range(n_samples):
        grid = np.zeros((4, 4), dtype=float)
        # Randomly choose horizontal bar or vertical bar
        if rng.rand() > 0.5:
            # Horizontal bar at row 1 or 2
            row = rng.choice([1, 2])
            grid[row, :] = 1.0
        else:
            # Vertical bar at col 1 or 2
            col = rng.choice([1, 2])
            grid[:, col] = 1.0

        # Add 5% noise bit flips
        noise_mask = rng.rand(4, 4) < 0.05
        grid = np.logical_xor(grid, noise_mask).astype(float)
        dataset.append(grid.ravel())

    return np.array(dataset)


def demonstrate_boltzmann_sampling(seed: int = 42) -> Dict[str, Any]:
    """Train RBM on structured 16-bit binary patterns and sample reconstructions."""
    X = generate_structured_binary_patterns(n_samples=300, seed=seed)
    rbm = RestrictedBoltzmannMachine(n_visible=16, n_hidden=8, lr=0.08, seed=seed)
    training_res = rbm.fit(X, epochs=80, batch_size=20)

    # Reconstruct test sample
    test_v = X[0:1]
    p_h, h = rbm.sample_hidden(test_v)
    p_v, v_recon = rbm.sample_visible(h)

    # Gibbs sampling chain for free energy exploration
    v_gibbs = test_v.copy()
    gibbs_energies = []
    for _ in range(30):
        _, h_s = rbm.sample_hidden(v_gibbs)
        _, v_gibbs = rbm.sample_visible(h_s)
        e = rbm.compute_energy(v_gibbs[0], h_s[0])
        gibbs_energies.append(e)

    return {
        "rbm": rbm,
        "training_history": training_res,
        "original_sample": test_v.reshape(4, 4),
        "reconstructed_sample": p_v.reshape(4, 4),
        "gibbs_energies": np.array(gibbs_energies),
    }


def plot_boltzmann_figures(
    results: Dict[str, Any],
    out_dir: Path | str = "artifacts/figures/learning_paradigms/",
) -> Path:
    """Generate figure displaying RBM reconstruction error, free energy descent, and pattern generation."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_file = out_path / "boltzmann_rbm_sampling_energy.png"

    hist = results["training_history"]
    orig = results["original_sample"]
    recon = results["reconstructed_sample"]
    gibbs_e = results["gibbs_energies"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)

    # 1. Reconstruction MSE
    axes[0].plot(hist["reconstruction_error"], color="#1f77b4", linewidth=2.0)
    axes[0].set_title("CD-1 Reconstruction Error", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Reconstruction MSE", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # 2. Free Energy Progression
    axes[1].plot(hist["free_energy"], color="#2ca02c", linewidth=2.0)
    axes[1].set_title("Average Free Energy Progression", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel(r"Free Energy $\mathcal{F}(v)$", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    # 3. Original vs Reconstructed Visual Pattern
    axes[2].imshow(orig, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("Input Binary Sample (4x4)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(recon, cmap="Blues", vmin=0, vmax=1)
    axes[3].set_title("RBM Reconstructed Probs", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    plt.suptitle("Restricted Boltzmann Machine: Contrastive Divergence Energy Learning & Reconstruction", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(fig_file, bbox_inches="tight")
    plt.close(fig)

    return fig_file

