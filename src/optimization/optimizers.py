"""Advanced First-Order Optimization Algorithms (REQ-04).

Implements from scratch using pure NumPy:
1. Standard Gradient Descent (GD)
2. Polyak Momentum
3. Nesterov Accelerated Gradient (NAG)
4. Adam (Adaptive Moment Estimation with bias corrections)

Author: Crop Disease Detection Project Team
"""

from typing import Any, Callable, Dict, List, Tuple
import numpy as np


class Optimizer:
    """Abstract base class for first-order optimizers."""

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
        self.t = 0

    def reset(self) -> None:
        self.t = 0

    def update(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GradientDescentOptimizer(Optimizer):
    """Standard Vanilla Gradient Descent."""

    def __init__(self, lr: float = 0.05) -> None:
        super().__init__(lr=lr)

    def update(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        return w - self.lr * grad


class MomentumOptimizer(Optimizer):
    """Polyak Classical Momentum.

    Update Equations:
        v_t = gamma * v_{t-1} + lr * g_t
        w_t = w_{t-1} - v_t
    """

    def __init__(self, lr: float = 0.05, momentum: float = 0.9) -> None:
        super().__init__(lr=lr)
        self.gamma = momentum
        self.v: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self.v = None

    def update(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.v is None:
            self.v = np.zeros_like(w)

        self.v = self.gamma * self.v + self.lr * grad
        return w - self.v


class NesterovOptimizer(Optimizer):
    """Nesterov Accelerated Gradient (NAG).

    Applies the momentum step first, then computes lookahead gradient:
        w_look = w_{t-1} - gamma * v_{t-1}
        g_look = grad(w_look)
        v_t = gamma * v_{t-1} + lr * g_look
        w_t = w_{t-1} - v_t
    """

    def __init__(self, lr: float = 0.05, momentum: float = 0.9) -> None:
        super().__init__(lr=lr)
        self.gamma = momentum
        self.v: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self.v = None

    def get_lookahead_position(self, w: np.ndarray) -> np.ndarray:
        """Compute virtual lookahead position where gradient should be evaluated."""
        if self.v is None:
            return w
        return w - self.gamma * self.v

    def update_with_lookahead_grad(self, w: np.ndarray, grad_lookahead: np.ndarray) -> np.ndarray:
        """Update parameter using gradient evaluated at the lookahead position."""
        self.t += 1
        if self.v is None:
            self.v = np.zeros_like(w)

        self.v = self.gamma * self.v + self.lr * grad_lookahead
        return w - self.v

    def update(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Fallback when lookahead gradient is approximated by current gradient."""
        return self.update_with_lookahead_grad(w, grad)


class AdamOptimizer(Optimizer):
    """Adam (Adaptive Moment Estimation with Bias Corrections).

    Update Equations:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat_t = m_t / (1 - beta1^t)        [First-moment bias correction]
        v_hat_t = v_t / (1 - beta2^t)        [Second-moment bias correction]
        w_t = w_{t-1} - (lr / (sqrt(v_hat_t) + eps)) * m_hat_t
    """

    def __init__(
        self,
        lr: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self.m = None
        self.v = None

    def update(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        # Biased first and second moment estimates
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad**2)

        # Bias corrections
        m_hat = self.m / (1.0 - (self.beta1**self.t))
        v_hat = self.v / (1.0 - (self.beta2**self.t))

        # Parameter update
        step = (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
        return w - step


def optimize_arbitrary_function(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    optimizer: Optimizer,
    w_start: np.ndarray,
    n_steps: int = 100,
) -> Dict[str, Any]:
    """Optimize an arbitrary mathematical function and record the trajectory."""
    optimizer.reset()
    w = np.array(w_start, dtype=float).copy()
    trajectory: List[np.ndarray] = [w.copy()]
    losses: List[float] = [f(w)]
    grad_norms: List[float] = [float(np.linalg.norm(grad_f(w)))]

    for _ in range(n_steps):
        if isinstance(optimizer, NesterovOptimizer):
            w_look = optimizer.get_lookahead_position(w)
            g = grad_f(w_look)
            w = optimizer.update_with_lookahead_grad(w, g)
        else:
            g = grad_f(w)
            w = optimizer.update(w, g)

        trajectory.append(w.copy())
        losses.append(f(w))
        grad_norms.append(float(np.linalg.norm(grad_f(w))))

    return {
        "trajectory": np.array(trajectory),
        "losses": np.array(losses),
        "grad_norms": np.array(grad_norms),
        "final_w": w,
        "final_loss": losses[-1],
    }


def compare_advanced_optimizers(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    w_start: np.ndarray,
    n_steps: int = 80,
    lr: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Compare GD, Momentum, Nesterov, and Adam from identical starting points."""
    opts = {
        "Gradient Descent": GradientDescentOptimizer(lr=lr),
        "Momentum": MomentumOptimizer(lr=lr, momentum=0.9),
        "Nesterov": NesterovOptimizer(lr=lr, momentum=0.9),
        "Adam": AdamOptimizer(lr=lr, beta1=0.9, beta2=0.999),
    }

    results = {}
    for name, opt in opts.items():
        results[name] = optimize_arbitrary_function(f, grad_f, opt, w_start, n_steps=n_steps)

    return results

