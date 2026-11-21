"""Unit Tests for Regularization, Initialization & Normalization (REQ-05).

Tests:
1. Weight initialization dimensions, statistical moments (Xavier, He), and zero-symmetry.
2. Signal variance preservation across depth (He on ReLU, Xavier on Tanh).
3. BatchNorm1dNumPy batch normalization properties and train vs eval behavior.
4. LayerNorm1dNumPy per-sample normalization and batch-size independence (including B=1).
5. L1, L2 penalty and gradient formulas.
6. Inverted Dropout mask distribution, inverted scaling, and train vs eval modes.
7. Controlled ablation study execution and schema integrity.

Author: Crop Disease Detection Project Team
"""

import numpy as np
import pytest

from src.regularization.ablation_studies import run_ablation_benchmark
from src.regularization.initialization import (
    he_initialization,
    random_initialization,
    simulate_deep_network_flow,
    xavier_initialization,
    zero_initialization,
)
from src.regularization.normalization import BatchNorm1dNumPy, LayerNorm1dNumPy
from src.regularization.regularizers import (
    DropoutLayer,
    L1Regularizer,
    L2Regularizer,
    RegularizedMLP,
    generate_overfitting_dataset,
    train_regularized_model,
)


# ---------------------------------------------------------------------------
# Weight Initialization Tests
# ---------------------------------------------------------------------------


def test_zero_initialization_and_symmetry():
    """Verify zero initialization causes identical hidden activations (symmetry collapse)."""
    W = zero_initialization((5, 10))
    assert W.shape == (5, 10)
    assert np.all(W == 0.0)

    # In a network, zero weights mean all neurons compute identical outputs
    sim = simulate_deep_network_flow("zero", activation_name="relu", n_layers=4, seed=42)
    assert sim["is_symmetric"] is True


def test_xavier_and_he_statistical_properties():
    """Verify Xavier and He empirical variances match theoretical formulas."""
    n_in, n_out = 1000, 1000
    shape = (n_in, n_out)

    # Xavier: Var = 2 / (n_in + n_out) = 2 / 2000 = 0.001
    W_xav = xavier_initialization(shape, seed=42)
    assert np.abs(np.mean(W_xav)) < 0.005
    expected_xav_var = 2.0 / (n_in + n_out)
    assert np.var(W_xav) == pytest.approx(expected_xav_var, rel=0.1)

    # He: Var = 2 / n_in = 2 / 1000 = 0.002
    W_he = he_initialization(shape, seed=42)
    assert np.abs(np.mean(W_he)) < 0.005
    expected_he_var = 2.0 / n_in
    assert np.var(W_he) == pytest.approx(expected_he_var, rel=0.1)


def test_deep_network_variance_propagation():
    """Verify He initialization prevents activation variance collapse on deep ReLU networks."""
    # Under small random init (sigma=0.01), activations vanish by layer 6
    sim_small = simulate_deep_network_flow("random_small", activation_name="relu", n_layers=6, seed=42)
    assert sim_small["act_stds"][-1] < 1e-6

    # Under He init, standard deviation remains stable (order of magnitude ~ 1)
    sim_he = simulate_deep_network_flow("he", activation_name="relu", n_layers=6, seed=42)
    assert sim_he["act_stds"][-1] > 0.1
    assert sim_he["act_stds"][-1] < 10.0


# ---------------------------------------------------------------------------
# Normalization Tests
# ---------------------------------------------------------------------------


def test_batch_norm_forward_and_eval():
    """Verify BatchNorm normalizes across axis 0 and uses running stats during evaluation."""
    N, D = 64, 10
    rng = np.random.RandomState(42)
    X = rng.randn(N, D) * 5.0 + 3.0

    bn = BatchNorm1dNumPy(num_features=D)
    bn.train()
    out_train = bn.forward(X)

    # In train mode, channel means must be ~0 and variances ~1
    np.testing.assert_allclose(np.mean(out_train, axis=0), np.zeros(D), atol=1e-5)
    np.testing.assert_allclose(np.var(out_train, axis=0), np.ones(D), atol=1e-3)

    # In eval mode with a single sample, evaluation must use running stats (no NaN or zero div)
    bn.eval()
    X_single = rng.randn(1, D)
    out_eval = bn.forward(X_single)
    assert out_eval.shape == (1, D)
    assert not np.isnan(out_eval).any()


def test_layer_norm_forward_and_batch_independence():
    """Verify LayerNorm normalizes across axis 1 per sample and works for B=1."""
    N, D = 16, 20
    rng = np.random.RandomState(42)
    X = rng.randn(N, D) * 4.0 - 2.0

    ln = LayerNorm1dNumPy(num_features=D)
    out = ln.forward(X)

    # In LayerNorm, each sample must have mean 0 and variance 1 across features
    np.testing.assert_allclose(np.mean(out, axis=1), np.zeros(N), atol=1e-5)
    np.testing.assert_allclose(np.var(out, axis=1), np.ones(N), atol=1e-3)

    # Works with batch size 1
    X_one = rng.randn(1, D)
    out_one = ln.forward(X_one)
    assert out_one.shape == (1, D)
    assert np.mean(out_one) == pytest.approx(0.0, abs=1e-5)
    assert np.var(out_one) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Regularizer Tests
# ---------------------------------------------------------------------------


def test_l1_and_l2_penalties_and_gradients():
    """Verify L1 and L2 penalty calculations and exact subgradient/gradient formulations."""
    W = np.array([[2.0, -3.0], [0.0, 4.0]])

    # L1: penalty = 0.02 * (2 + 3 + 0 + 4) = 0.02 * 9 = 0.18
    # subgrad = 0.02 * sign(W) = 0.02 * [[1, -1], [0, 1]]
    l1 = L1Regularizer(lmbda=0.02)
    assert l1.compute_penalty([W]) == pytest.approx(0.18)
    expected_l1_grad = np.array([[0.02, -0.02], [0.0, 0.02]])
    np.testing.assert_allclose(l1.compute_gradient(W), expected_l1_grad)

    # L2: penalty = 0.5 * 0.05 * (4 + 9 + 0 + 16) = 0.025 * 29 = 0.725
    # grad = 0.05 * W = [[0.1, -0.15], [0.0, 0.2]]
    l2 = L2Regularizer(lmbda=0.05)
    assert l2.compute_penalty([W]) == pytest.approx(0.725)
    expected_l2_grad = 0.05 * W
    np.testing.assert_allclose(l2.compute_gradient(W), expected_l2_grad)


def test_inverted_dropout_behavior():
    """Verify inverted dropout scaling 1/(1-p) during training and pass-through at inference."""
    X = np.ones((100, 100))
    p_drop = 0.4
    dropout = DropoutLayer(p_drop=p_drop, seed=42)

    # Train mode
    dropout.train()
    out_train = dropout.forward(X)
    # Expected mean remains 1.0 due to inverted scaling
    assert np.mean(out_train) == pytest.approx(1.0, rel=0.1)
    # Fraction of zeroed entries should be close to p_drop
    zero_ratio = np.mean(out_train == 0.0)
    assert zero_ratio == pytest.approx(p_drop, abs=0.05)

    # Eval mode
    dropout.eval()
    out_eval = dropout.forward(X)
    np.testing.assert_allclose(out_eval, X)
    assert not np.any(out_eval == 0.0)


def test_ablation_benchmark_execution():
    """Verify ablation benchmark executes all 7 controlled experiments and generates records."""
    res = run_ablation_benchmark(seed=42)
    records = res["records"]
    assert len(records) == 7

    for r in records:
        assert "experiment" in r
        assert "train_accuracy_pct" in r
        assert "val_accuracy_pct" in r
        assert "generalization_gap_pct" in r
        assert 0.0 <= r["train_accuracy_pct"] <= 100.0
        assert 0.0 <= r["val_accuracy_pct"] <= 100.0

