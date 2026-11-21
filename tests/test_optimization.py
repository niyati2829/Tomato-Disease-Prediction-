"""Unit Tests for Optimization Dynamics (REQ-04).

Tests:
1. ConvexRegressionObjective loss and analytical gradients.
2. Batch GD, SGD, and Mini-batch GD convergence and update mechanics.
3. Advanced optimizers (Momentum, Nesterov, Adam) update equations.
4. Adam bias-correction exactness on step 1.
5. Learning rate stability, undershooting, and divergence behavior.
6. Optimization trajectory convergence on anisotropic valley.

Author: Crop Disease Detection Project Team
"""

import numpy as np
import pytest

from src.optimization.gradient_descent import (
    BatchGradientDescent,
    ConvexRegressionObjective,
    MiniBatchGradientDescent,
    StochasticGradientDescent,
    generate_synthetic_regression_data,
)
from src.optimization.learning_rate_experiments import run_learning_rate_experiments
from src.optimization.loss_landscape import (
    anisotropic_valley_gradient,
    anisotropic_valley_loss,
    compute_optimizer_trajectories,
)
from src.optimization.optimizers import (
    AdamOptimizer,
    GradientDescentOptimizer,
    MomentumOptimizer,
    NesterovOptimizer,
    optimize_arbitrary_function,
)


def test_convex_regression_objective_gradient():
    """Verify analytical gradient matches central finite difference on regression objective."""
    X, y, _ = generate_synthetic_regression_data(n_samples=50, n_features=3, seed=42)
    obj = ConvexRegressionObjective(X, y)
    w = np.array([0.5, -1.0, 2.0])

    analytical_grad = obj.compute_gradient(w)

    # Numerical gradient via central differences
    eps = 1e-7
    numerical_grad = np.zeros_like(w)
    for i in range(len(w)):
        w_plus = w.copy()
        w_plus[i] += eps
        w_minus = w.copy()
        w_minus[i] -= eps
        numerical_grad[i] = (obj.compute_loss(w_plus) - obj.compute_loss(w_minus)) / (2.0 * eps)

    diff = np.linalg.norm(analytical_grad - numerical_grad)
    assert diff < 1e-6


def test_batch_gradient_descent_convergence():
    """Verify Batch GD decreases loss monotonically on convex objective."""
    X, y, w_true = generate_synthetic_regression_data(n_samples=100, n_features=3, seed=42)
    obj = ConvexRegressionObjective(X, y)
    w_init = np.array([5.0, -5.0, 5.0])

    bgd = BatchGradientDescent(lr=0.08, max_epochs=60)
    res = bgd.optimize(obj, w_init)

    assert res["final_loss"] < res["loss_history"][0]
    assert res["total_updates"] == 60
    assert res["total_sample_evaluations"] == 60 * 100
    # Loss should decrease monotonically for reasonable lr
    for i in range(len(res["loss_history"]) - 1):
        assert res["loss_history"][i] >= res["loss_history"][i + 1] or res["loss_history"][i + 1] < 1e-3


def test_sgd_and_minibatch_convergence():
    """Verify SGD and Mini-batch GD converge and perform correct update counts."""
    X, y, _ = generate_synthetic_regression_data(n_samples=64, n_features=3, seed=42)
    obj = ConvexRegressionObjective(X, y)
    w_init = np.array([2.0, -2.0, 2.0])

    # SGD
    sgd = StochasticGradientDescent(lr=0.01, max_epochs=30, seed=42)
    res_sgd = sgd.optimize(obj, w_init)
    assert res_sgd["total_updates"] == 30 * 64
    assert res_sgd["final_loss"] < res_sgd["loss_history"][0]

    # Mini-batch GD (B=16 -> 4 batches per epoch)
    mbgd = MiniBatchGradientDescent(lr=0.03, batch_size=16, max_epochs=30, seed=42)
    res_mbgd = mbgd.optimize(obj, w_init)
    assert res_mbgd["total_updates"] == 30 * 4
    assert res_mbgd["final_loss"] < res_mbgd["loss_history"][0]


def test_momentum_and_nesterov_updates():
    """Verify momentum velocity accumulation and lookahead mechanics."""
    w = np.array([1.0, 1.0])
    grad = np.array([0.5, 0.5])

    # Momentum step 1: v_1 = 0.9 * 0 + 0.1 * [0.5, 0.5] = [0.05, 0.05]
    # w_1 = [1.0, 1.0] - [0.05, 0.05] = [0.95, 0.95]
    opt_m = MomentumOptimizer(lr=0.1, momentum=0.9)
    w_next = opt_m.update(w.copy(), grad)
    np.testing.assert_allclose(w_next, np.array([0.95, 0.95]))
    assert opt_m.v is not None
    np.testing.assert_allclose(opt_m.v, np.array([0.05, 0.05]))

    # Nesterov lookahead computation
    opt_nag = NesterovOptimizer(lr=0.1, momentum=0.9)
    opt_nag.v = np.array([0.1, -0.1])
    lookahead = opt_nag.get_lookahead_position(w)
    # w_look = w - gamma * v = [1.0, 1.0] - 0.9 * [0.1, -0.1] = [0.91, 1.09]
    np.testing.assert_allclose(lookahead, np.array([0.91, 1.09]))


def test_adam_bias_correction_step_one():
    """Verify Adam bias correction matches uncentered moment on step 1."""
    # On step t=1:
    # m_1 = (1 - beta1) * g_1
    # m_hat_1 = m_1 / (1 - beta1^1) = g_1!
    # v_1 = (1 - beta2) * g_1^2
    # v_hat_1 = v_1 / (1 - beta2^1) = g_1^2!
    # Step = lr / (sqrt(g_1^2) + eps) * g_1 = lr * sign(g_1) (approx)
    opt = AdamOptimizer(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
    w = np.array([2.0, -3.0])
    g = np.array([4.0, -9.0])

    w_new = opt.update(w.copy(), g)
    assert opt.t == 1
    # Since m_hat_1 = g and sqrt(v_hat_1) = |g|, step size is approximately lr * sign(g)
    expected_w = w - 0.1 * np.sign(g)
    np.testing.assert_allclose(w_new, expected_w, atol=1e-5)


def test_learning_rate_stability_and_divergence():
    """Verify small learning rate is slow, while excessive learning rate diverges."""
    res = run_learning_rate_experiments(learning_rates=[0.01, 0.5, 2.1], n_steps=35, seed=42)

    # Too small
    assert res["lr_0.01"]["final_loss"] > res["lr_0.5"]["final_loss"]
    assert res["lr_0.01"]["diverged"] is False

    # Optimal
    assert res["lr_0.5"]["diverged"] is False
    assert res["lr_0.5"]["final_loss"] < 0.1

    # Divergent
    assert res["lr_2.1"]["diverged"] is True


def test_anisotropic_valley_optimization():
    """Verify all 4 optimizers reach near-zero loss on anisotropic valley."""
    trajectories = compute_optimizer_trajectories(w_start=(-3.0, 2.0), n_steps=60, condition_number=8.0)

    for name in ["Gradient Descent", "Momentum", "Nesterov", "Adam"]:
        final_loss = trajectories[name]["final_loss"]
        # All should minimize the valley substantially from initial loss ~ 18.5
        assert final_loss < 1.0, f"{name} failed to reach low loss: {final_loss}"
