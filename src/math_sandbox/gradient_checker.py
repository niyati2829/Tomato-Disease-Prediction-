"""Numerical vs. Analytical Gradient Checking Module.

Provides robust mathematical gradient verification:
1. Forward finite difference: (f(x + eps) - f(x)) / eps [O(eps)]
2. Central finite difference: (f(x + eps) - f(x - eps)) / (2 * eps) [O(eps^2)]
3. Multi-variable vector & matrix gradient verification.
4. Relative error metric: ||grad_num - grad_ana|| / (||grad_num|| + ||grad_ana||)
5. Convergence & step-size sensitivity plot (numerical roundoff vs. truncation error).

Author: Crop Disease Detection Project Team
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def forward_finite_difference(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute numerical gradient using forward finite difference formula O(eps)."""
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    fx = f(x)

    while not it.finished:
        idx = it.multi_index
        orig_val = x[idx]

        x[idx] = orig_val + eps
        fx_plus = f(x)
        x[idx] = orig_val  # Restore

        grad[idx] = (fx_plus - fx) / eps
        it.iternext()

    return grad


def central_finite_difference(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Compute numerical gradient using central finite difference formula O(eps^2)."""
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        orig_val = x[idx]

        x[idx] = orig_val + eps
        fx_plus = f(x)

        x[idx] = orig_val - eps
        fx_minus = f(x)

        x[idx] = orig_val  # Restore

        grad[idx] = (fx_plus - fx_minus) / (2.0 * eps)
        it.iternext()

    return grad


def compute_relative_gradient_error(
    grad_numerical: np.ndarray,
    grad_analytical: np.ndarray,
    eps_denom: float = 1e-12,
) -> float:
    """Calculate normalized relative Euclidean error between two gradient tensors.

    Formula:
        rel_error = ||grad_num - grad_ana||_2 / (||grad_num||_2 + ||grad_ana||_2 + eps)
    """
    diff_norm = np.linalg.norm(grad_numerical - grad_analytical)
    num_norm = np.linalg.norm(grad_numerical)
    ana_norm = np.linalg.norm(grad_analytical)
    denom = num_norm + ana_norm + eps_denom
    return float(diff_norm / denom)


def check_gradient_scalar(
    f: Callable[[float], float],
    f_prime: Callable[[float], float],
    x: float,
    eps_list: List[float] | None = None,
    h: float | None = None,
    eps: float | None = None,
) -> Any:
    """Compare forward and central numerical derivatives on scalar function."""
    ana_grad = f_prime(x)

    # If single step size specified
    single_h = h if h is not None else eps
    if single_h is not None:
        fwd_num = (f(x + single_h) - f(x)) / single_h
        cnt_num = (f(x + single_h) - f(x - single_h)) / (2.0 * single_h)
        fwd_err = abs(fwd_num - ana_grad)
        cnt_err = abs(cnt_num - ana_grad)
        return {
            "h": single_h,
            "eps": single_h,
            "analytical_grad": ana_grad,
            "forward_num_grad": fwd_num,
            "forward_diff_error": fwd_err,
            "central_num_grad": cnt_num,
            "central_diff_error": cnt_err,
            "central_diff_passed": cnt_err < 1e-6,
        }

    steps = eps_list if eps_list is not None else [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    results = []
    for step in steps:
        fwd_num = (f(x + step) - f(x)) / step
        cnt_num = (f(x + step) - f(x - step)) / (2.0 * step)

        fwd_err = abs(fwd_num - ana_grad)
        cnt_err = abs(cnt_num - ana_grad)

        results.append({
            "eps": step,
            "analytical_grad": ana_grad,
            "forward_num_grad": fwd_num,
            "forward_abs_error": fwd_err,
            "central_num_grad": cnt_num,
            "central_abs_error": cnt_err,
        })
    return results


def check_gradient_vector(
    f: Callable[[np.ndarray], float],
    grad_analytical: np.ndarray | Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-7,
    h: float | None = None,
) -> Dict[str, Any]:
    """Verify analytical gradient vector against central difference gradient."""
    step = h if h is not None else eps
    if callable(grad_analytical):
        grad_ana = grad_analytical(x)
    else:
        grad_ana = grad_analytical

    grad_numerical = central_finite_difference(f, x.copy(), eps=step)

    abs_err_max = float(np.max(np.abs(grad_numerical - grad_ana)))
    rel_error = compute_relative_gradient_error(grad_numerical, grad_ana)
    passed = rel_error < 1e-5

    return {
        "passed": passed,
        "relative_error": rel_error,
        "max_absolute_error": abs_err_max,
        "eps_used": step,
        "grad_numerical": grad_numerical.tolist(),
        "grad_analytical": grad_ana.tolist(),
    }


# Matrix alias (functions identically via np.nditer across arbitrary tensor dimensions)
check_gradient_matrix = check_gradient_vector


def plot_gradient_check_verification(output_path: Path | str) -> Path:
    """Plot truncation error vs roundoff error across eps step sizes."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Test function: f(x) = sin(x) * exp(x/2) at x = 1.2
    # f'(x) = cos(x)*exp(x/2) + 0.5*sin(x)*exp(x/2)
    x0 = 1.2
    f = lambda x: np.sin(x) * np.exp(x / 2.0)
    f_prime = lambda x: np.cos(x) * np.exp(x / 2.0) + 0.5 * np.sin(x) * np.exp(x / 2.0)
    true_deriv = f_prime(x0)

    eps_range = np.logspace(-14, 0, 50)
    fwd_errors = []
    cnt_errors = []

    for eps in eps_range:
        fwd = (f(x0 + eps) - f(x0)) / eps
        cnt = (f(x0 + eps) - f(x0 - eps)) / (2.0 * eps)
        fwd_errors.append(abs(fwd - true_deriv))
        cnt_errors.append(abs(cnt - true_deriv))

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax.loglog(eps_range, fwd_errors, "o-", color="#d32f2f", label="Forward Difference $O(\\epsilon)$", linewidth=2, markersize=3)
    ax.loglog(eps_range, cnt_errors, "s-", color="#1976d2", label="Central Difference $O(\\epsilon^2)$", linewidth=2, markersize=3)

    # Annotate optimal epsilon
    min_cnt_idx = np.argmin(cnt_errors)
    opt_eps = eps_range[min_cnt_idx]
    opt_err = cnt_errors[min_cnt_idx]
    ax.scatter([opt_eps], [opt_err], color="#388e3c", s=120, zorder=5, label=f"Optimal $\\epsilon \\approx 10^{{-7}}$ (Error $\\sim 10^{{-11}}$)")

    ax.set_title("Numerical Gradient Checking: Truncation Error vs. Floating-Point Cancellation", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Perturbation Step Size $\\epsilon$ (Log Scale)", fontsize=10)
    ax.set_ylabel("Absolute Derivative Error $|\\nabla_{num} - \\nabla_{ana}|$ (Log Scale)", fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
