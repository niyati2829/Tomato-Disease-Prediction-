"""Regularization Techniques: L1, L2, and Inverted Dropout (REQ-05).

Implements from scratch using pure NumPy:
1. L1 Regularization (Lasso penalty, subgradient, sparsity induction)
2. L2 Regularization (Ridge / Weight Decay, smooth quadratic shrinkage)
3. Inverted Dropout (Bernoulli masking, 1/(1-p) scaling, train vs inference behavior)

Trains an over-parameterized neural classifier on small-sample data to benchmark
overfitting mitigation across Unregularized, L1, L2, and Dropout.

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


class L1Regularizer:
    """L1 Regularization (Lasso): Penalty = lambda * sum(|W|)."""

    def __init__(self, lmbda: float = 0.01) -> None:
        self.lmbda = lmbda

    def compute_penalty(self, weights: List[np.ndarray]) -> float:
        total = 0.0
        for W in weights:
            total += float(np.sum(np.abs(W)))
        return self.lmbda * total

    def compute_gradient(self, W: np.ndarray) -> np.ndarray:
        """Subgradient of L1: lambda * sign(W)."""
        return self.lmbda * np.sign(W)


class L2Regularizer:
    """L2 Regularization (Weight Decay / Ridge): Penalty = 0.5 * lambda * sum(W^2)."""

    def __init__(self, lmbda: float = 0.01) -> None:
        self.lmbda = lmbda

    def compute_penalty(self, weights: List[np.ndarray]) -> float:
        total = 0.0
        for W in weights:
            total += float(np.sum(W**2))
        return 0.5 * self.lmbda * total

    def compute_gradient(self, W: np.ndarray) -> np.ndarray:
        """Gradient of L2: lambda * W."""
        return self.lmbda * W


class DropoutLayer:
    """Inverted Dropout Layer.

    During training:
        mask ~ Bernoulli(1 - p)
        output = (X * mask) / (1 - p)
    During inference:
        output = X
    """

    def __init__(self, p_drop: float = 0.3, seed: int = 42) -> None:
        self.p_drop = p_drop
        self.rng = np.random.RandomState(seed)
        self.training = True
        self.mask: np.ndarray | None = None

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def forward(self, X: np.ndarray) -> np.ndarray:
        if not self.training or self.p_drop == 0.0:
            return X

        # Keep probability
        p_keep = 1.0 - self.p_drop
        self.mask = (self.rng.rand(*X.shape) < p_keep).astype(float)
        # Inverted scaling ensures expectation E[output] == X
        return (X * self.mask) / p_keep

    def backward(self, dY: np.ndarray) -> np.ndarray:
        if not self.training or self.p_drop == 0.0 or self.mask is None:
            return dY
        p_keep = 1.0 - self.p_drop
        return (dY * self.mask) / p_keep


class RegularizedMLP:
    """2-layer MLP classifier with configurable L1, L2, and Dropout regularizers."""

    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        n_out: int,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        dropout_p: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        # He initialization for hidden, Xavier for output
        self.W1 = self.rng.randn(n_in, n_hidden) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = self.rng.randn(n_hidden, n_out) * np.sqrt(2.0 / (n_hidden + n_out))
        self.b2 = np.zeros((1, n_out))

        self.l1 = L1Regularizer(l1_reg)
        self.l2 = L2Regularizer(l2_reg)
        self.dropout = DropoutLayer(p_drop=dropout_p, seed=seed)
        self.training = True

    def train(self) -> None:
        self.training = True
        self.dropout.train()

    def eval(self) -> None:
        self.training = False
        self.dropout.eval()

    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0.0, Z1)  # ReLU
        A1_drop = self.dropout.forward(A1)
        Z2 = A1_drop @ self.W2 + self.b2
        A2 = self._softmax(Z2)
        cache = {"X": X, "Z1": Z1, "A1": A1, "A1_drop": A1_drop, "Z2": Z2, "A2": A2}
        return A2, cache

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        N = y_true.shape[0]
        y_clipped = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        ce_loss = -np.sum(y_true * np.log(y_clipped)) / N
        reg_loss = self.l1.compute_penalty([self.W1, self.W2]) + self.l2.compute_penalty([self.W1, self.W2])
        return float(ce_loss + reg_loss)

    def backward(self, y_true: np.ndarray, cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        N = y_true.shape[0]
        A2 = cache["A2"]
        dZ2 = (A2 - y_true) / N

        dW2 = cache["A1_drop"].T @ dZ2 + self.l1.compute_gradient(self.W2) + self.l2.compute_gradient(self.W2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1_drop = dZ2 @ self.W2.T
        dA1 = self.dropout.backward(dA1_drop)
        dZ1 = dA1 * (cache["Z1"] > 0.0)

        dW1 = cache["X"].T @ dZ1 + self.l1.compute_gradient(self.W1) + self.l2.compute_gradient(self.W1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_parameters(self, grads: Dict[str, np.ndarray], lr: float) -> None:
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]


def generate_overfitting_dataset(
    n_train: int = 50,
    n_val: int = 150,
    n_features: int = 30,
    n_classes: int = 2,
    noise: float = 0.25,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate high-dimensional, small-sample dataset prone to empirical overfitting."""
    rng = np.random.RandomState(seed)
    n_total = n_train + n_val

    # Only 5 features carry true predictive signal; 25 features are pure noise
    X_informative = rng.randn(n_total, 5)
    w_true = rng.uniform(-1.5, 1.5, size=(5, 1))
    logits = X_informative @ w_true + rng.randn(n_total, 1) * noise
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (probs > 0.5).astype(int).ravel()

    # Noise features
    X_noise = rng.randn(n_total, n_features - 5)
    X = np.hstack([X_informative, X_noise])
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # One-hot
    Y_onehot = np.zeros((n_total, n_classes))
    Y_onehot[np.arange(n_total), labels] = 1.0

    X_train, y_train = X[:n_train], Y_onehot[:n_train]
    X_val, y_val = X[n_train:], Y_onehot[n_train:]
    return X_train, y_train, X_val, y_val


def train_regularized_model(
    model: RegularizedMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    lr: float = 0.15,
) -> Dict[str, Any]:
    """Train model and record training vs validation loss and accuracy curves."""
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        # Training pass
        model.train()
        probs_train, cache = model.forward(X_train)
        loss_train = model.compute_loss(y_train, probs_train)
        grads = model.backward(y_train, cache)
        model.update_parameters(grads, lr=lr)

        # Accuracy
        preds_train = np.argmax(probs_train, axis=1)
        acc_train = float(np.mean(preds_train == np.argmax(y_train, axis=1)))

        # Validation pass
        model.eval()
        probs_val, _ = model.forward(X_val)
        loss_val = model.compute_loss(y_val, probs_val)
        preds_val = np.argmax(probs_val, axis=1)
        acc_val = float(np.mean(preds_val == np.argmax(y_val, axis=1)))

        history["train_loss"].append(loss_train)
        history["val_loss"].append(loss_val)
        history["train_acc"].append(acc_train)
        history["val_acc"].append(acc_val)

    # Calculate weight metrics
    all_w = np.concatenate([model.W1.ravel(), model.W2.ravel()])
    w_norm_l2 = float(np.linalg.norm(all_w))
    w_norm_l1 = float(np.sum(np.abs(all_w)))
    sparsity = float(np.mean(np.abs(all_w) < 1e-3)) * 100.0

    return {
        "history": history,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "w_norm_l2": w_norm_l2,
        "w_norm_l1": w_norm_l1,
        "sparsity_percentage": sparsity,
        "weights_w1": model.W1.copy(),
        "weights_w2": model.W2.copy(),
    }


def compare_regularization_schemes(seed: int = 42) -> Dict[str, Any]:
    """Run controlled experiment comparing Unregularized, L1, L2, and Dropout."""
    X_train, y_train, X_val, y_val = generate_overfitting_dataset(seed=seed)
    n_in = X_train.shape[1]
    n_hidden = 40
    n_out = 2

    experiments = {
        "Unregularized (Baseline)": RegularizedMLP(n_in, n_hidden, n_out, l1_reg=0.0, l2_reg=0.0, dropout_p=0.0, seed=seed),
        r"L1 Regularization ($\lambda=0.015$)": RegularizedMLP(n_in, n_hidden, n_out, l1_reg=0.015, l2_reg=0.0, dropout_p=0.0, seed=seed),
        r"L2 Regularization ($\lambda=0.02$)": RegularizedMLP(n_in, n_hidden, n_out, l1_reg=0.0, l2_reg=0.02, dropout_p=0.0, seed=seed),
        "Dropout ($p=0.35$)": RegularizedMLP(n_in, n_hidden, n_out, l1_reg=0.0, l2_reg=0.0, dropout_p=0.35, seed=seed),
    }

    results = {}
    for name, model in experiments.items():
        results[name] = train_regularized_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.12)

    return results


def plot_regularization_figures(
    results: Dict[str, Any],
    out_dir: Path | str = "artifacts/figures/regularization/",
) -> Tuple[Path, Path, Path]:
    """Generate 3 regularization figures: curves, weight distributions, and dropout diagram."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig1_file = out_path / "regularization_loss_accuracy_curves.png"
    fig2_file = out_path / "l1_vs_l2_weight_magnitudes.png"
    fig3_file = out_path / "dropout_training_inference_diagram.png"

    # Figure 1: Train vs Validation Loss & Accuracy curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), dpi=300)
    colors = {
        "Unregularized (Baseline)": "#d62728",
        r"L1 Regularization ($\lambda=0.015$)": "#2ca02c",
        r"L2 Regularization ($\lambda=0.02$)": "#1f77b4",
        "Dropout ($p=0.35$)": "#9467bd",
    }

    for name, data in results.items():
        c = colors.get(name, "#333333")
        hist = data["history"]
        ax1.plot(hist["train_loss"], label=f"{name} (Train)", color=c, linestyle="--", alpha=0.7)
        ax1.plot(hist["val_loss"], label=f"{name} (Val)", color=c, linewidth=2.0)

        ax2.plot(hist["train_acc"], label=f"{name} (Train)", color=c, linestyle="--", alpha=0.7)
        ax2.plot(hist["val_acc"], label=f"{name} (Val)", color=c, linewidth=2.0)

    ax1.set_title("Cross-Entropy Loss Progression", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_yscale("log")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(fontsize=8, loc="upper right")

    ax2.set_title("Classification Accuracy Progression", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epochs", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(fontsize=8, loc="lower right")

    plt.suptitle("Combating Overfitting: Regularization Generalization Impact", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(fig1_file, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: L1 vs L2 Weight magnitude distribution (Sparsity vs Shrinkage)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    w_unreg = results["Unregularized (Baseline)"]["weights_w1"].ravel()
    w_l1 = results[r"L1 Regularization ($\lambda=0.015$)"]["weights_w1"].ravel()
    w_l2 = results[r"L2 Regularization ($\lambda=0.02$)"]["weights_w1"].ravel()

    plots = [
        ("Unregularized Weights", w_unreg, "#d62728", axes[0]),
        ("L1 Regularized (Sparsity / Zeros)", w_l1, "#2ca02c", axes[1]),
        ("L2 Regularized (Weight Shrinkage)", w_l2, "#1f77b4", axes[2]),
    ]

    for title, w_arr, c, ax in plots:
        ax.hist(w_arr, bins=45, color=c, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_title(rf"{title}" + "\n" + rf"$\|W\|_2={np.linalg.norm(w_arr):.2f}, Zeros(<10^{{-3}})={np.mean(np.abs(w_arr)<1e-3)*100:.1f}\%$", fontsize=10, fontweight="bold")
        ax.set_xlabel("Weight Value", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Parameter Distributions: L1 Feature Selection vs L2 Weight Shrinkage", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(fig2_file, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Dropout training vs inference schematic illustration
    fig, (ax_train, ax_eval) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Diagrammatic neuron layers
    layer_in = [0.2, 0.4, 0.6, 0.8]
    layer_hidden = [0.15, 0.35, 0.55, 0.75, 0.95]
    layer_out = [0.4, 0.6]

    # Training mode: with dropped units (marked with X)
    ax_train.set_title("Training Mode ($p_{drop}=0.4$)\nRandom Masking + Inverted Scaling $1/(1-p)$", fontsize=11, fontweight="bold")
    for y_in in layer_in:
        ax_train.scatter(0.2, y_in, color="#1f77b4", s=300, zorder=5)
    for idx, y_h in enumerate(layer_hidden):
        if idx in [1, 3]:  # Dropped neurons
            ax_train.scatter(0.5, y_h, color="#e0e0e0", edgecolors="red", linewidth=2.0, s=300, zorder=5)
            ax_train.text(0.5, y_h, "X", color="red", fontsize=14, fontweight="bold", ha="center", va="center")
        else:
            ax_train.scatter(0.5, y_h, color="#2ca02c", s=300, zorder=5)
    for y_o in layer_out:
        ax_train.scatter(0.8, y_o, color="#ff7f0e", s=300, zorder=5)

    # Draw active connections
    for y_in in layer_in:
        for idx, y_h in enumerate(layer_hidden):
            if idx not in [1, 3]:
                ax_train.plot([0.2, 0.5], [y_in, y_h], color="#888888", alpha=0.4, linewidth=1.0)
    for idx, y_h in enumerate(layer_hidden):
        if idx not in [1, 3]:
            for y_o in layer_out:
                ax_train.plot([0.5, 0.8], [y_h, y_o], color="#888888", alpha=0.4, linewidth=1.0)

    ax_train.set_xlim(0.0, 1.0)
    ax_train.set_ylim(0.0, 1.1)
    ax_train.axis("off")

    # Inference mode: full ensemble active
    ax_eval.set_title("Inference / Evaluation Mode ($p_{drop}=0$)\nDeterministic Identity Pass-through (All Neurons Active)", fontsize=11, fontweight="bold")
    for y_in in layer_in:
        ax_eval.scatter(0.2, y_in, color="#1f77b4", s=300, zorder=5)
    for y_h in layer_hidden:
        ax_eval.scatter(0.5, y_h, color="#2ca02c", s=300, zorder=5)
    for y_o in layer_out:
        ax_eval.scatter(0.8, y_o, color="#ff7f0e", s=300, zorder=5)

    for y_in in layer_in:
        for y_h in layer_hidden:
            ax_eval.plot([0.2, 0.5], [y_in, y_h], color="#888888", alpha=0.4, linewidth=1.0)
    for y_h in layer_hidden:
        for y_o in layer_out:
            ax_eval.plot([0.5, 0.8], [y_h, y_o], color="#888888", alpha=0.4, linewidth=1.0)

    ax_eval.set_xlim(0.0, 1.0)
    ax_eval.set_ylim(0.0, 1.1)
    ax_eval.axis("off")

    plt.suptitle("Dropout Architecture: Sub-network Sampling during Training vs Ensemble Inference", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig3_file, bbox_inches="tight")
    plt.close(fig)

    return fig1_file, fig2_file, fig3_file
