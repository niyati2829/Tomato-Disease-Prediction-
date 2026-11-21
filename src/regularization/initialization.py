"""Weight Initialization Schemes & Variance Propagation (REQ-05).

Implements and benchmarks:
1. Zero Initialization (Demonstrating symmetry failure)
2. Random Initialization (Demonstrating variance collapse/explosion)
3. Xavier / Glorot Initialization (Optimized for Tanh/Sigmoid)
4. He / Kaiming Initialization (Optimized for ReLU)

Tracks layer-by-layer activation distributions, gradient norms, and symmetry breaking.

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


def zero_initialization(shape: Tuple[int, int]) -> np.ndarray:
    """Zero initialization: W_ij = 0.

    Fails symmetry breaking: all hidden units compute identical outputs
    and receive identical gradients, preventing the network from learning diverse features.
    """
    return np.zeros(shape, dtype=float)


def random_initialization(shape: Tuple[int, int], scale: float = 1.0, seed: int = 42) -> np.ndarray:
    """Naive Gaussian initialization: W_ij ~ N(0, scale^2).

    If scale is too large (e.g., 1.0), deep activations explode or saturate;
    if scale is too small (e.g., 0.01), signals vanish rapidly across depth.
    """
    rng = np.random.RandomState(seed)
    return rng.randn(*shape) * scale


def xavier_initialization(shape: Tuple[int, int], seed: int = 42) -> np.ndarray:
    """Xavier / Glorot Normal initialization: W_ij ~ N(0, 2 / (n_in + n_out)).

    Preserves activation and gradient variance across linear and Tanh/Sigmoid layers.
    Derived under the assumption that activation function has derivative ~ 1 at zero.
    """
    n_in, n_out = shape
    std = np.sqrt(2.0 / (n_in + n_out))
    rng = np.random.RandomState(seed)
    return rng.randn(n_in, n_out) * std


def he_initialization(shape: Tuple[int, int], seed: int = 42) -> np.ndarray:
    """He / Kaiming Normal initialization: W_ij ~ N(0, 2 / n_in).

    Accounts for the rectifying property of ReLU, which sets ~50% of activations to zero,
    halving the output variance. Compensates with an extra factor of 2.
    """
    n_in, _ = shape
    std = np.sqrt(2.0 / n_in)
    rng = np.random.RandomState(seed)
    return rng.randn(*shape) * std


def simulate_deep_network_flow(
    init_scheme: str,
    activation_name: str = "relu",
    n_layers: int = 6,
    layer_dim: int = 120,
    n_samples: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Pass Gaussian noise through a deep linear-activation chain and record variance across depth."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, layer_dim)

    weights: List[np.ndarray] = []
    activations: List[np.ndarray] = [X]
    act_means: List[float] = [float(np.mean(X))]
    act_stds: List[float] = [float(np.std(X))]

    # Build and pass forward
    curr_A = X
    for l in range(n_layers):
        shape = (layer_dim, layer_dim)
        if init_scheme == "zero":
            W = zero_initialization(shape)
        elif init_scheme == "random_large":
            W = random_initialization(shape, scale=1.0, seed=seed + l)
        elif init_scheme == "random_small":
            W = random_initialization(shape, scale=0.01, seed=seed + l)
        elif init_scheme == "xavier":
            W = xavier_initialization(shape, seed=seed + l)
        elif init_scheme == "he":
            W = he_initialization(shape, seed=seed + l)
        else:
            raise ValueError(f"Unknown init scheme: {init_scheme}")

        weights.append(W)
        Z = curr_A @ W

        if activation_name == "relu":
            curr_A = np.maximum(0.0, Z)
        elif activation_name == "tanh":
            curr_A = np.tanh(Z)
        elif activation_name == "linear":
            curr_A = Z
        else:
            raise ValueError(f"Unknown activation: {activation_name}")

        activations.append(curr_A)
        act_means.append(float(np.mean(curr_A)))
        act_stds.append(float(np.std(curr_A)))

    # Simulate backward gradient flow
    grad_stds: List[float] = []
    dA = rng.randn(n_samples, layer_dim)
    grad_stds.append(float(np.std(dA)))

    for l in reversed(range(n_layers)):
        W = weights[l]
        if activation_name == "relu":
            dZ = dA * (activations[l + 1] > 0.0)
        elif activation_name == "tanh":
            dZ = dA * (1.0 - activations[l + 1] ** 2)
        else:
            dZ = dA

        dA = dZ @ W.T
        grad_stds.append(float(np.std(dA)))

    grad_stds.reverse()

    # Check for symmetry failure in zero init
    final_layer = activations[-1]
    is_symmetric = bool(np.allclose(final_layer, final_layer[:, [0]]))

    return {
        "scheme": init_scheme,
        "activation": activation_name,
        "act_means": act_means,
        "act_stds": act_stds,
        "grad_stds": grad_stds,
        "weights": weights,
        "activations": activations,
        "is_symmetric": is_symmetric,
    }


def compare_initialization_schemes(seed: int = 42) -> Dict[str, Any]:
    """Run comparative suite on deep MLP across Zero, Random Small, Random Large, Xavier, and He."""
    schemes = ["zero", "random_small", "random_large", "xavier", "he"]
    results_relu = {}
    results_tanh = {}

    for s in schemes:
        results_relu[s] = simulate_deep_network_flow(s, activation_name="relu", seed=seed)
        results_tanh[s] = simulate_deep_network_flow(s, activation_name="tanh", seed=seed)

    return {"relu": results_relu, "tanh": results_tanh}


def plot_initialization_diagnostics(
    sim_data: Dict[str, Any],
    out_dir: Path | str = "artifacts/figures/regularization/",
) -> Tuple[Path, Path]:
    """Plot weight histograms and layer activation/gradient standard deviation curves."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig1_file = out_path / "weight_initialization_histograms.png"
    fig2_file = out_path / "layer_activation_gradient_distributions.png"

    # Figure 1: Weight distribution histograms
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
    shape = (200, 200)

    w_zero = zero_initialization(shape).ravel()
    w_rand = random_initialization(shape, scale=0.5, seed=42).ravel()
    w_xav = xavier_initialization(shape, seed=42).ravel()
    w_he = he_initialization(shape, seed=42).ravel()

    plots = [
        ("Zero Init", w_zero, "#d62728", axes[0]),
        (r"Random Init ($\sigma=0.5$)", w_rand, "#ff7f0e", axes[1]),
        ("Xavier Init (Glorot)", w_xav, "#1f77b4", axes[2]),
        ("He Init (Kaiming)", w_he, "#2ca02c", axes[3]),
    ]

    for title, w_arr, color, ax in plots:
        ax.hist(w_arr, bins=40, color=color, alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.set_title(rf"{title}" + "\n" + rf"$\mu={np.mean(w_arr):.4f}, \sigma={np.std(w_arr):.4f}$", fontsize=10, fontweight="bold")
        ax.set_xlabel("Weight Value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle(r"Weight Initialization Density Distributions ($200 \times 200$ Matrix)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(fig1_file, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Layer-wise activation standard deviation decay/explosion across depth
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    relu_data = sim_data["relu"]
    tanh_data = sim_data["tanh"]

    schemes_meta = [
        ("zero", "Zero Init", "#d62728", "o"),
        ("random_small", r"Random ($\sigma=0.01$)", "#8c564b", "x"),
        ("random_large", r"Random ($\sigma=1.0$)", "#e377c2", "^"),
        ("xavier", "Xavier (Glorot)", "#1f77b4", "s"),
        ("he", "He (Kaiming)", "#2ca02c", "D"),
    ]

    # Panel 1: ReLU Network
    for key, label, color, marker in schemes_meta:
        stds = relu_data[key]["act_stds"]
        ax1.plot(range(len(stds)), stds, label=label, color=color, marker=marker, linewidth=2.0)

    ax1.set_yscale("log")
    ax1.set_title("Activation Variance Propagation with ReLU", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Layer Depth ($0 = Input$)", fontsize=10)
    ax1.set_ylabel(r"Standard Deviation $\sigma(A^{[l]})$ [Log Scale]", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8.5, loc="lower left")

    # Panel 2: Tanh Network
    for key, label, color, marker in schemes_meta:
        stds = tanh_data[key]["act_stds"]
        ax2.plot(range(len(stds)), stds, label=label, color=color, marker=marker, linewidth=2.0)

    ax2.set_yscale("log")
    ax2.set_title("Activation Variance Propagation with Tanh", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Layer Depth ($0 = Input$)", fontsize=10)
    ax2.set_ylabel(r"Standard Deviation $\sigma(A^{[l]})$ [Log Scale]", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8.5, loc="lower left")

    plt.suptitle("Signal Propagation Stability Across 6 Hidden Layers", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(fig2_file, bbox_inches="tight")
    plt.close(fig)

    return fig1_file, fig2_file
