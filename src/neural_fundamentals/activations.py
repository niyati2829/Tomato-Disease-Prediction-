"""Activation Functions & Derivatives Comparative Analysis Module.

Implements and compares:
1. Heaviside Step Function
2. Sigmoid (Logistic) Function
3. Hyperbolic Tangent (Tanh)
4. Rectified Linear Unit (ReLU)
5. Leaky ReLU

Features:
- Exact mathematical forward formulas.
- Exact analytical first derivatives.
- Output range, saturation analysis, and vanishing gradient comparisons.
- High-resolution multi-panel visualization of functions and gradients.

Author: Crop Disease Detection Project Team
"""

import os
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Forward & Derivative Implementations
# ---------------------------------------------------------------------------

def step(z: np.ndarray) -> np.ndarray:
    """Heaviside step activation: 1 if z >= 0 else 0."""
    return np.where(z >= 0.0, 1.0, 0.0)


def step_derivative(z: np.ndarray) -> np.ndarray:
    """Step derivative: 0 everywhere (dirac delta at 0 approximated as 0 for optimization)."""
    return np.zeros_like(z)


def sigmoid_act(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation: 1 / (1 + exp(-z))."""
    z_clipped = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def sigmoid_act_derivative(z: np.ndarray) -> np.ndarray:
    """Sigmoid derivative: sigma(z) * (1 - sigma(z)). Max derivative = 0.25 at z=0."""
    s = sigmoid_act(z)
    return s * (1.0 - s)


def tanh_act(z: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation: (exp(z) - exp(-z)) / (exp(z) + exp(-z))."""
    return np.tanh(z)


def tanh_act_derivative(z: np.ndarray) -> np.ndarray:
    """Tanh derivative: 1 - tanh^2(z). Max derivative = 1.0 at z=0."""
    t = np.tanh(z)
    return 1.0 - t ** 2


def relu_act(z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation: max(0, z)."""
    return np.maximum(0.0, z)


def relu_act_derivative(z: np.ndarray) -> np.ndarray:
    """ReLU sub-gradient: 1.0 if z > 0 else 0.0 (chosen 0 at z=0)."""
    return np.where(z > 0.0, 1.0, 0.0)


def leaky_relu_act(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU activation: z if z > 0 else alpha * z."""
    return np.where(z > 0.0, z, alpha * z)


def leaky_relu_act_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU derivative: 1.0 if z > 0 else alpha."""
    return np.where(z > 0.0, 1.0, alpha)


# Convenience aliases
step_function = step
sigmoid = sigmoid_act
sigmoid_derivative = sigmoid_act_derivative
tanh = tanh_act
tanh_derivative = tanh_act_derivative
relu = relu_act
relu_derivative = relu_act_derivative
leaky_relu = leaky_relu_act
leaky_relu_derivative = leaky_relu_act_derivative


def get_activation_specifications() -> Dict[str, Dict[str, Any]]:
    """Return theoretical properties and trade-offs of each activation function."""
    return {
        "Step": {
            "output_range": "{0, 1}",
            "zero_centered": False,
            "max_derivative": 0.0,
            "vanishing_gradient_risk": "Catastrophic (zero gradient everywhere prevents backprop)",
            "primary_use_case": "Historical McCulloch-Pitts and single-layer Perceptrons only",
        },
        "Sigmoid": {
            "output_range": "(0, 1)",
            "zero_centered": False,
            "max_derivative": 0.25,
            "vanishing_gradient_risk": "Severe for |z| > 4 (derivative < 0.018)",
            "primary_use_case": "Binary classification output probability layer",
        },
        "Tanh": {
            "output_range": "(-1, 1)",
            "zero_centered": True,
            "max_derivative": 1.0,
            "vanishing_gradient_risk": "Moderate (saturates for |z| > 3)",
            "primary_use_case": "Recurrent neural networks (RNN / LSTM cell state gates)",
        },
        "ReLU": {
            "output_range": "[0, inf)",
            "zero_centered": False,
            "max_derivative": 1.0,
            "vanishing_gradient_risk": "None for z > 0; Dying ReLU problem for z < 0",
            "primary_use_case": "Standard default hidden layers for modern feedforward and CNNs",
        },
        "Leaky ReLU": {
            "output_range": "(-inf, inf)",
            "zero_centered": "Approximately",
            "max_derivative": 1.0,
            "vanishing_gradient_risk": "Mitigated (gradient = alpha for z < 0)",
            "primary_use_case": "Generative adversarial networks (GANs) and deep residual nets",
        },
    }


def plot_activation_comparison(output_path: Path | str) -> Path:
    """Generate high-resolution comparative visualization of activation functions and derivatives."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    z = np.linspace(-4.0, 4.0, 400)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Left: Activations g(z)
    ax1.plot(z, step(z), label="Step (Heaviside)", color="#757575", linestyle=":", linewidth=2)
    ax1.plot(z, sigmoid_act(z), label="Sigmoid $\\sigma(z)$", color="#1976d2", linewidth=2.5)
    ax1.plot(z, tanh_act(z), label="Tanh $\\tanh(z)$", color="#388e3c", linewidth=2.5)
    ax1.plot(z, relu_act(z), label="ReLU $\\max(0, z)$", color="#d32f2f", linewidth=2.5)
    ax1.plot(z, leaky_relu_act(z, alpha=0.1), label="Leaky ReLU ($\\alpha=0.1$)", color="#f57c00", linestyle="--", linewidth=2)

    ax1.set_title("Activation Functions $g(z)$", fontsize=11, fontweight="bold", pad=12)
    ax1.set_xlabel("Input Activation $z$", fontsize=10)
    ax1.set_ylabel("Output $g(z)$", fontsize=10)
    ax1.set_ylim([-1.5, 3.5])
    ax1.axhline(0, color="black", linestyle="-", alpha=0.3, linewidth=0.8)
    ax1.axvline(0, color="black", linestyle="-", alpha=0.3, linewidth=0.8)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, fontsize=9, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: Derivatives g'(z)
    ax2.plot(z, step_derivative(z), label="Step' $(= 0)$", color="#757575", linestyle=":", linewidth=2)
    ax2.plot(z, sigmoid_act_derivative(z), label="Sigmoid' $(\\max = 0.25)$", color="#1976d2", linewidth=2.5)
    ax2.plot(z, tanh_act_derivative(z), label="Tanh' $(\\max = 1.0)$", color="#388e3c", linewidth=2.5)
    ax2.plot(z, relu_act_derivative(z), label="ReLU' $(\\in \\{0, 1\\})$", color="#d32f2f", linewidth=2.5)
    ax2.plot(z, leaky_relu_act_derivative(z, alpha=0.1), label="Leaky ReLU' $(\\in \\{0.1, 1\\})$", color="#f57c00", linestyle="--", linewidth=2)

    ax2.set_title("Activation Derivatives $g'(z)$ (Gradient Flow)", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xlabel("Input Activation $z$", fontsize=10)
    ax2.set_ylabel("First Derivative $g'(z)$", fontsize=10)
    ax2.set_ylim([-0.1, 1.2])
    ax2.axhline(0, color="black", linestyle="-", alpha=0.3, linewidth=0.8)
    ax2.axvline(0, color="black", linestyle="-", alpha=0.3, linewidth=0.8)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=9, loc="upper left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle("Activation Function Landscape & Derivative Dynamics (REQ-02)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
