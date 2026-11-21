"""Gradient Descent Variants Implementation (REQ-04).

Implements and benchmarks:
1. Batch Gradient Descent (full dataset per step)
2. Stochastic Gradient Descent (1 sample per step with random permutation)
3. Mini-batch Gradient Descent (B samples per step)

Applied to a reproducible convex quadratic / mean squared error objective:
    L(w) = (1 / 2N) * ||X w - y||_2^2
    grad_L(w) = (1 / N) * X^T (X w - y)

Author: Crop Disease Detection Project Team
"""

import os
from typing import Any, Dict, List, Tuple
import numpy as np


class ConvexRegressionObjective:
    """Represents a convex linear regression objective L(w) = 1/(2N) * ||X w - y||^2."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1, 1)
        self.N, self.D = self.X.shape

    def compute_loss(self, w: np.ndarray, indices: np.ndarray | None = None) -> float:
        """Compute MSE loss on specified sample subset (or entire dataset)."""
        w_col = w.reshape(-1, 1)
        if indices is None:
            err = self.X @ w_col - self.y
            n = self.N
        else:
            err = self.X[indices] @ w_col - self.y[indices]
            n = len(indices)
        return float(np.sum(err**2) / (2.0 * n))

    def compute_gradient(self, w: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
        """Compute analytical gradient on specified sample subset (or entire dataset)."""
        w_col = w.reshape(-1, 1)
        if indices is None:
            err = self.X @ w_col - self.y
            grad = (self.X.T @ err) / self.N
        else:
            X_sub = self.X[indices]
            err_sub = X_sub @ w_col - self.y[indices]
            grad = (X_sub.T @ err_sub) / len(indices)
        return grad.ravel()

    def get_analytical_optimum(self) -> np.ndarray:
        """Compute closed-form Ordinary Least Squares solution w* = (X^T X)^{-1} X^T y."""
        w_opt, _, _, _ = np.linalg.lstsq(self.X, self.y, rcond=None)
        return w_opt.ravel()


class BatchGradientDescent:
    """Batch Gradient Descent: Updates parameters using the exact full-dataset gradient."""

    def __init__(self, lr: float = 0.05, max_epochs: int = 100) -> None:
        self.lr = lr
        self.max_epochs = max_epochs

    def optimize(
        self,
        objective: ConvexRegressionObjective,
        w_init: np.ndarray,
    ) -> Dict[str, Any]:
        w = np.array(w_init, dtype=float).copy()
        loss_history: List[float] = []
        grad_norm_history: List[float] = []
        w_history: List[np.ndarray] = [w.copy()]
        eval_count = 0

        for epoch in range(self.max_epochs):
            loss = objective.compute_loss(w)
            grad = objective.compute_gradient(w)
            eval_count += objective.N

            loss_history.append(loss)
            grad_norm = float(np.linalg.norm(grad))
            grad_norm_history.append(grad_norm)

            w -= self.lr * grad
            w_history.append(w.copy())

        final_loss = objective.compute_loss(w)
        loss_history.append(final_loss)
        grad_norm_history.append(float(np.linalg.norm(objective.compute_gradient(w))))

        return {
            "name": "Batch Gradient Descent",
            "w_final": w,
            "w_history": np.array(w_history),
            "loss_history": np.array(loss_history),
            "grad_norm_history": np.array(grad_norm_history),
            "total_updates": self.max_epochs,
            "total_sample_evaluations": eval_count,
            "final_loss": final_loss,
            "converged": bool(grad_norm_history[-1] < 1e-4),
        }


class StochasticGradientDescent:
    """Stochastic Gradient Descent (SGD): Updates parameters 1 sample at a time."""

    def __init__(self, lr: float = 0.01, max_epochs: int = 100, seed: int = 42) -> None:
        self.lr = lr
        self.max_epochs = max_epochs
        self.seed = seed

    def optimize(
        self,
        objective: ConvexRegressionObjective,
        w_init: np.ndarray,
    ) -> Dict[str, Any]:
        rng = np.random.RandomState(self.seed)
        w = np.array(w_init, dtype=float).copy()
        N = objective.N

        loss_history: List[float] = [objective.compute_loss(w)]
        grad_norm_history: List[float] = [float(np.linalg.norm(objective.compute_gradient(w)))]
        w_history: List[np.ndarray] = [w.copy()]
        total_updates = 0
        eval_count = 0

        for epoch in range(self.max_epochs):
            # Shuffle indices each epoch
            indices = rng.permutation(N)
            for idx in indices:
                sub_idx = np.array([idx])
                grad = objective.compute_gradient(w, indices=sub_idx)
                w -= self.lr * grad
                total_updates += 1
                eval_count += 1

            # Record full-batch loss at end of epoch
            full_loss = objective.compute_loss(w)
            full_grad_norm = float(np.linalg.norm(objective.compute_gradient(w)))
            loss_history.append(full_loss)
            grad_norm_history.append(full_grad_norm)
            w_history.append(w.copy())

        return {
            "name": "Stochastic Gradient Descent (SGD)",
            "w_final": w,
            "w_history": np.array(w_history),
            "loss_history": np.array(loss_history),
            "grad_norm_history": np.array(grad_norm_history),
            "total_updates": total_updates,
            "total_sample_evaluations": eval_count,
            "final_loss": loss_history[-1],
            "converged": bool(grad_norm_history[-1] < 1e-3),
        }


class MiniBatchGradientDescent:
    """Mini-batch Gradient Descent: Updates parameters using small subsets of size B."""

    def __init__(self, lr: float = 0.02, batch_size: int = 16, max_epochs: int = 100, seed: int = 42) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.seed = seed

    def optimize(
        self,
        objective: ConvexRegressionObjective,
        w_init: np.ndarray,
    ) -> Dict[str, Any]:
        rng = np.random.RandomState(self.seed)
        w = np.array(w_init, dtype=float).copy()
        N = objective.N

        loss_history: List[float] = [objective.compute_loss(w)]
        grad_norm_history: List[float] = [float(np.linalg.norm(objective.compute_gradient(w)))]
        w_history: List[np.ndarray] = [w.copy()]
        total_updates = 0
        eval_count = 0

        for epoch in range(self.max_epochs):
            indices = rng.permutation(N)
            for start in range(0, N, self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                grad = objective.compute_gradient(w, indices=batch_indices)
                w -= self.lr * grad
                total_updates += 1
                eval_count += len(batch_indices)

            full_loss = objective.compute_loss(w)
            full_grad_norm = float(np.linalg.norm(objective.compute_gradient(w)))
            loss_history.append(full_loss)
            grad_norm_history.append(full_grad_norm)
            w_history.append(w.copy())

        return {
            "name": f"Mini-Batch GD (B={self.batch_size})",
            "w_final": w,
            "w_history": np.array(w_history),
            "loss_history": np.array(loss_history),
            "grad_norm_history": np.array(grad_norm_history),
            "total_updates": total_updates,
            "total_sample_evaluations": eval_count,
            "final_loss": loss_history[-1],
            "converged": bool(grad_norm_history[-1] < 1e-3),
        }


def generate_synthetic_regression_data(
    n_samples: int = 300,
    n_features: int = 5,
    noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate reproducible synthetic linear regression dataset with ground truth weights."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    # Normalize features for consistent conditioning
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    w_true = rng.uniform(-2.0, 2.0, size=n_features)
    noise = rng.randn(n_samples) * noise_std
    y = X @ w_true + noise
    return X, y, w_true


def compare_gd_variants(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 60,
    batch_size: int = 16,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Run Batch GD, SGD, and Mini-batch GD on identical problem with identical initial weights."""
    obj = ConvexRegressionObjective(X, y)
    rng = np.random.RandomState(seed)
    w_init = rng.randn(X.shape[1]) * 1.5

    bgd = BatchGradientDescent(lr=lr, max_epochs=epochs)
    sgd = StochasticGradientDescent(lr=lr * 0.1, max_epochs=epochs, seed=seed)
    mbgd = MiniBatchGradientDescent(lr=lr * 0.5, batch_size=batch_size, max_epochs=epochs, seed=seed)

    res_bgd = bgd.optimize(obj, w_init)
    res_sgd = sgd.optimize(obj, w_init)
    res_mbgd = mbgd.optimize(obj, w_init)

    w_opt = obj.get_analytical_optimum()

    return {
        "batch_gd": res_bgd,
        "sgd": res_sgd,
        "mini_batch_gd": res_mbgd,
        "optimal_loss": obj.compute_loss(w_opt),
        "w_optimal": w_opt,
    }

