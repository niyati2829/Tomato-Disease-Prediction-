"""Sigmoid Neuron Implementation.

Provides continuous, differentiable non-linear activation:
    z = w^T x + b
    y_hat = sigma(z) = 1 / (1 + exp(-z))
    d sigma(z) / dz = sigma(z) * (1 - sigma(z))

Features:
1. Numerically stable sigmoid function (prevents floating-point overflow for large negative z).
2. Exact analytical derivative computation.
3. Vectorized forward pass across batches of inputs.
4. Binary cross-entropy loss and analytical gradient computation:
       dL / dw = (y_hat - y) * x
       dL / db = (y_hat - y)
5. Gradient descent parameter updates.

Author: Crop Disease Detection Project Team
"""

import numpy as np
from typing import Any, Dict, Tuple


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid activation function."""
    z_arr = np.asarray(z, dtype=float)
    # Use standard formulation for positive z, exp(z)/(1+exp(z)) for negative z
    pos_mask = (z_arr >= 0)
    neg_mask = ~pos_mask

    res = np.empty_like(z_arr)
    res[pos_mask] = 1.0 / (1.0 + np.exp(-z_arr[pos_mask]))
    res[neg_mask] = np.exp(z_arr[neg_mask]) / (1.0 + np.exp(z_arr[neg_mask]))

    return float(res) if np.isscalar(z) else res


def sigmoid_derivative(z: np.ndarray | float) -> np.ndarray | float:
    """Analytical derivative of sigmoid: d sigma(z) / dz = sigma(z) * (1 - sigma(z))."""
    s = sigmoid(z)
    return s * (1.0 - s)


class SigmoidNeuron:
    """Single artificial neuron with continuous sigmoid activation."""

    def __init__(self, n_features: int | None = None, n_inputs: int | None = None, seed: int = 42) -> None:
        """Initialize weights and bias."""
        num_inputs = n_features if n_features is not None else n_inputs
        if num_inputs is None:
            raise ValueError("Must provide either n_features or n_inputs.")
        rng = np.random.RandomState(seed)
        self.weights = rng.randn(num_inputs) * 0.1
        self.bias = 0.0

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Vectorized forward pass.

        Args:
            X: Input array of shape (N, n_features).

        Returns:
            Probabilities of shape (N,).
        """
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
        """Compute Binary Cross-Entropy (BCE) loss."""
        y_hat = np.clip(y_pred, eps, 1.0 - eps)
        bce = -np.mean(y_true * np.log(y_hat) + (1.0 - y_true) * np.log(1.0 - y_hat))
        return float(bce)

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1) -> Tuple[float, float, float]:
        """Execute single gradient descent update step.

        Returns:
            Tuple of (loss, grad_w_norm, grad_b).
        """
        N = X.shape[0]
        y_hat = self.forward(X)
        loss = self.compute_loss(y, y_hat)

        # Analytical gradients under Binary Cross Entropy
        error = y_hat - y  # Shape (N,)
        grad_w = np.dot(X.T, error) / N  # Shape (n_features,)
        grad_b = float(np.mean(error))

        # Parameter update
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b

        return loss, float(np.linalg.norm(grad_w)), grad_b

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.1) -> Dict[str, Any]:
        """Fit Sigmoid neuron across multiple epochs using gradient descent."""
        history: Dict[str, list] = {"loss": [], "accuracy": []}
        for _ in range(epochs):
            loss, _, _ = self.train_step(X, y, lr=lr)
            preds = np.where(self.forward(X) >= 0.5, 1.0, 0.0)
            acc = float(np.mean(preds == y))
            history["loss"].append(loss)
            history["accuracy"].append(acc)
        return history
