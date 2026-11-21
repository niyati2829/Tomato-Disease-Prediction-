"""Multilayer Perceptron (MLP) from Scratch with Manual Analytical Backpropagation.

A pure NumPy implementation of a feedforward neural network:
1. Vectorized forward propagation with explicit activation caching.
2. Configurable layer architectures (e.g., [2, 16, 8, 3]).
3. Configurable hidden activation (ReLU or Sigmoid) with Softmax output head.
4. Multiclass categorical Cross-Entropy loss.
5. Exact analytical backpropagation derived via the multivariable chain rule:
       dZ^[L] = (1/N) * (Y_hat - Y)
       dW^[l] = (A^[l-1])^T @ dZ^[l]
       db^[l] = sum_rows(dZ^[l])
       dA^[l-1] = dZ^[l] @ (W^[l])^T
       dZ^[l-1] = dA^[l-1] * g'(Z^[l-1])
6. Mini-batch gradient descent optimizer.
7. Explicit separation of parameters (W, b) from hyperparameters (lr, layers, epochs).
8. End-to-end two-sided numerical gradient checking verification.

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


class MultilayerPerceptron:
    """Fully vectorized Multilayer Perceptron built without autograd."""

    def __init__(
        self,
        layer_sizes: List[int],
        hidden_activation: str = "relu",
        seed: int = 42,
    ) -> None:
        """Initialize weights, biases, and hyperparameters.

        Args:
            layer_sizes: List of integers representing neuron counts [n_in, h1, h2, ..., n_out].
            hidden_activation: 'relu' or 'sigmoid'.
            seed: Random seed for reproducible initialization.
        """
        assert len(layer_sizes) >= 2, "Architecture must contain at least input and output layers."
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.hidden_activation = hidden_activation.lower()
        self.seed = seed

        # Track hyperparameters separately
        self.hyperparameters = {
            "layer_sizes": layer_sizes,
            "hidden_activation": hidden_activation,
            "seed": seed,
        }

        # Initialize parameters: He initialization for ReLU, Xavier for Sigmoid
        self.parameters: Dict[str, np.ndarray] = {}
        rng = np.random.RandomState(seed)

        for l in range(1, self.num_layers + 1):
            n_in = layer_sizes[l - 1]
            n_out = layer_sizes[l]

            if self.hidden_activation == "relu" and l < self.num_layers:
                # He/Kaiming normal initialization: std = sqrt(2 / n_in)
                std = np.sqrt(2.0 / n_in)
            else:
                # Xavier/Glorot normal initialization: std = sqrt(2 / (n_in + n_out))
                std = np.sqrt(2.0 / (n_in + n_out))

            self.parameters[f"W{l}"] = rng.randn(n_in, n_out) * std
            self.parameters[f"b{l}"] = np.zeros((1, n_out))

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Return parameter count breakdown and parameter vs hyperparameter metadata."""
        layer_shapes = {}
        total_params = 0
        for name, param in self.parameters.items():
            layer_shapes[name] = {"shape": param.shape, "numel": param.size}
            total_params += param.size

        return {
            "total_parameters": total_params,
            "total_learnable_tensors": len(self.parameters),
            "layer_shapes": layer_shapes,
            "learnable_parameters": list(self.parameters.keys()),
            "hyperparameters": {
                "layer_sizes": self.layer_sizes,
                "hidden_activation": self.hidden_activation,
                "num_layers": self.num_layers,
                "seed": self.seed,
            },
        }

    # -----------------------------------------------------------------------
    # Activation Utilities
    # -----------------------------------------------------------------------

    def _activate(self, Z: np.ndarray, act_name: str) -> np.ndarray:
        if act_name == "relu":
            return np.maximum(0.0, Z)
        elif act_name == "sigmoid":
            clipped_Z = np.clip(Z, -50.0, 50.0)
            return 1.0 / (1.0 + np.exp(-clipped_Z))
        else:
            raise ValueError(f"Unknown activation: {act_name}")

    def _activate_derivative(self, Z: np.ndarray, act_name: str) -> np.ndarray:
        if act_name == "relu":
            return np.where(Z > 0.0, 1.0, 0.0)
        elif act_name == "sigmoid":
            s = self._activate(Z, "sigmoid")
            return s * (1.0 - s)
        else:
            raise ValueError(f"Unknown activation: {act_name}")

    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax along rows."""
        # Subtract max for numerical stability (prevents overflow in exp)
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    # -----------------------------------------------------------------------
    # Forward Propagation
    # -----------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Execute vectorized forward pass and cache linear and activation tensors.

        Args:
            X: Input matrix of shape (N, n_features).

        Returns:
            Tuple of (output_probabilities, cache_dict).
        """
        cache: Dict[str, np.ndarray] = {"A0": X}
        A = X

        # Hidden layers
        for l in range(1, self.num_layers):
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]
            Z = np.dot(A, W) + b
            A = self._activate(Z, self.hidden_activation)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        # Output layer with Softmax
        W_last = self.parameters[f"W{self.num_layers}"]
        b_last = self.parameters[f"b{self.num_layers}"]
        Z_last = np.dot(A, W_last) + b_last
        A_last = self._softmax(Z_last)

        cache[f"Z{self.num_layers}"] = Z_last
        cache[f"A{self.num_layers}"] = A_last

        return A_last, cache

    # -----------------------------------------------------------------------
    # Loss Calculation
    # -----------------------------------------------------------------------

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
        """Compute multiclass Categorical Cross-Entropy loss.

        Args:
            y_true: One-hot encoded labels of shape (N, n_classes).
            y_pred: Softmax probabilities of shape (N, n_classes).

        Returns:
            Scalar average cross-entropy loss.
        """
        N = y_true.shape[0]
        y_clipped = np.clip(y_pred, eps, 1.0 - eps)
        # Sum across classes, mean across batch samples
        loss = -np.sum(y_true * np.log(y_clipped)) / N
        return float(loss)

    # -----------------------------------------------------------------------
    # Analytical Backward Propagation
    # -----------------------------------------------------------------------

    def backward(self, y_true: np.ndarray, cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute manual analytical backpropagation using the chain rule.

        Returns:
            Dictionary containing analytical parameter gradients: dW1, db1, dW2, db2, ...
        """
        grads: Dict[str, np.ndarray] = {}
        N = y_true.shape[0]
        L = self.num_layers

        # Output layer gradient: Softmax + CrossEntropy combination simplifies cleanly to:
        # dL / dZ^[L] = (1 / N) * (A^[L] - Y)
        Y_hat = cache[f"A{L}"]
        dZ = (Y_hat - y_true) / N

        A_prev = cache[f"A{L-1}"]
        grads[f"dW{L}"] = np.dot(A_prev.T, dZ)
        grads[f"db{L}"] = np.sum(dZ, axis=0, keepdims=True)

        # Backpropagate through hidden layers
        for l in range(L - 1, 0, -1):
            W_next = self.parameters[f"W{l+1}"]
            dA = np.dot(dZ, W_next.T)
            dZ = dA * self._activate_derivative(cache[f"Z{l}"], self.hidden_activation)

            A_prev = cache[f"A{l-1}"]
            grads[f"dW{l}"] = np.dot(A_prev.T, dZ)
            grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True)

        return grads

    # -----------------------------------------------------------------------
    # Parameter Updates
    # -----------------------------------------------------------------------

    def update_parameters(self, grads: Dict[str, np.ndarray], lr: float) -> None:
        """Apply gradient descent parameter updates."""
        for l in range(1, self.num_layers + 1):
            self.parameters[f"W{l}"] -= lr * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= lr * grads[f"db{l}"]

    # -----------------------------------------------------------------------
    # Numerical Gradient Verification
    # -----------------------------------------------------------------------

    def verify_gradients_numerically(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        eps: float = 1e-7,
    ) -> Dict[str, Any]:
        """Verify analytical gradients against two-sided central finite differences.

        Returns:
            Dictionary containing relative errors per parameter tensor.
        """
        # Compute analytical gradients
        _, cache = self.forward(X)
        analytical_grads = self.backward(y_true, cache)

        results: Dict[str, Any] = {}
        max_overall_error = 0.0

        for key in list(self.parameters.keys()):
            param = self.parameters[key]
            grad_ana = analytical_grads[f"d{key}"]
            grad_num = np.zeros_like(param)

            # Iterate over all elements in the parameter tensor
            it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                idx = it.multi_index
                orig_val = param[idx]

                # Perturb +eps
                param[idx] = orig_val + eps
                probs_pos, _ = self.forward(X)
                loss_pos = self.compute_loss(y_true, probs_pos)

                # Perturb -eps
                param[idx] = orig_val - eps
                probs_neg, _ = self.forward(X)
                loss_neg = self.compute_loss(y_true, probs_neg)

                # Restore
                param[idx] = orig_val

                # Central difference
                grad_num[idx] = (loss_pos - loss_neg) / (2.0 * eps)
                it.iternext()

            diff_norm = np.linalg.norm(grad_num - grad_ana)
            denom = np.linalg.norm(grad_num) + np.linalg.norm(grad_ana) + 1e-12
            rel_error = float(diff_norm / denom)

            max_overall_error = max(max_overall_error, rel_error)
            results[key] = {
                "relative_error": rel_error,
                "analytical_norm": float(np.linalg.norm(grad_ana)),
                "numerical_norm": float(np.linalg.norm(grad_num)),
                "passed": bool(rel_error < 1e-5),
            }

        results["max_relative_error"] = max_overall_error
        results["all_passed"] = bool(max_overall_error < 1e-5)
        return results

    # -----------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.05,
    ) -> Dict[str, List[float]]:
        """Train MLP using mini-batch gradient descent."""
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        N = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            # Shuffle indices each epoch
            indices = np.random.permutation(N)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch gradient updates
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                _, cache = self.forward(X_batch)
                grads = self.backward(y_batch, cache)
                self.update_parameters(grads, lr)

            # Epoch evaluation
            train_preds, _ = self.forward(X_train)
            train_loss = self.compute_loss(y_train, train_preds)
            train_acc = float(np.mean(np.argmax(train_preds, axis=1) == np.argmax(y_train, axis=1)) * 100.0)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if X_val is not None and y_val is not None:
                val_preds, _ = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_preds)
                val_acc = float(np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val, axis=1)) * 100.0)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

        return history


def plot_mlp_training_results(
    history: Dict[str, List[float]],
    output_path: Path | str,
) -> Path:
    """Generate high-resolution training loss and accuracy curves."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1976d2", linewidth=2)
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#d32f2f", linestyle="--", linewidth=2)
    ax1.set_title("Cross-Entropy Loss vs Epoch", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="#388e3c", linewidth=2)
    if "val_acc" in history and history["val_acc"]:
        ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="#f57c00", linestyle="--", linewidth=2)
    ax2.set_title("Classification Accuracy (%) vs Epoch", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_ylim([0, 105])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, loc="lower right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle("Multilayer Perceptron from Scratch: Manual Backpropagation Training Dynamics (REQ-03)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
