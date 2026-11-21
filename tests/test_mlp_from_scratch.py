"""Unit Tests for Multilayer Perceptron From Scratch (REQ-03).

Tests:
1. Vectorized forward pass output dimensions and Softmax normalization.
2. Cross-Entropy loss computation.
3. Rigorous numerical gradient checking comparing analytical backprop against central differences.
4. Parameter vs Hyperparameter separation.
5. Mini-batch gradient descent training convergence on synthetic data.

Author: Crop Disease Detection Project Team
"""

import numpy as np
import pytest

from src.neural_fundamentals.mlp_from_scratch import MultilayerPerceptron


def test_mlp_forward_shapes_and_softmax():
    """Verify forward propagation tensor dimensions and Softmax partition function sum."""
    mlp = MultilayerPerceptron(layer_sizes=[4, 8, 6, 3], hidden_activation="relu", seed=42)
    N = 10
    X = np.random.randn(N, 4)

    probs, cache = mlp.forward(X)

    # Output probabilities must have shape (N, n_classes)
    assert probs.shape == (N, 3)
    # Probabilities must be non-negative and sum to 1 across rows
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    row_sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(N), atol=1e-12)

    # Check cached tensors
    assert cache["A0"].shape == (N, 4)
    assert cache["Z1"].shape == (N, 8)
    assert cache["A1"].shape == (N, 8)
    assert cache["Z2"].shape == (N, 6)
    assert cache["A2"].shape == (N, 6)
    assert cache["Z3"].shape == (N, 3)
    assert cache["A3"].shape == (N, 3)


def test_mlp_cross_entropy_loss():
    """Verify categorical cross entropy loss is positive and penalizes incorrect predictions."""
    mlp = MultilayerPerceptron(layer_sizes=[2, 4, 2], hidden_activation="relu", seed=42)
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])

    # Near-perfect predictions -> near-zero loss
    probs_good = np.array([[0.999, 0.001], [0.001, 0.999]])
    loss_good = mlp.compute_loss(y_true, probs_good)
    assert 0.0 < loss_good < 0.01

    # Confident wrong predictions -> high loss
    probs_bad = np.array([[0.001, 0.999], [0.999, 0.001]])
    loss_bad = mlp.compute_loss(y_true, probs_bad)
    assert loss_bad > 5.0


def test_mlp_gradient_checking_relu():
    """Verify analytical backpropagation gradients match central finite differences (< 1e-5) for ReLU."""
    np.random.seed(123)
    mlp = MultilayerPerceptron(layer_sizes=[2, 5, 4, 3], hidden_activation="relu", seed=123)

    # Use small batch with non-zero inputs to avoid ReLU kink at 0
    X = np.random.randn(8, 2) + 0.5
    y = np.zeros((8, 3))
    y[np.arange(8), np.random.randint(0, 3, 8)] = 1.0

    res = mlp.verify_gradients_numerically(X, y, eps=1e-7)

    assert res["all_passed"] is True
    assert res["max_relative_error"] < 1e-5

    # Check all parameter tensors individually
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        assert res[key]["passed"] is True
        assert res[key]["relative_error"] < 1e-5


def test_mlp_gradient_checking_sigmoid():
    """Verify analytical backpropagation gradients match central finite differences (< 1e-5) for Sigmoid."""
    np.random.seed(456)
    mlp = MultilayerPerceptron(layer_sizes=[3, 6, 2], hidden_activation="sigmoid", seed=456)

    X = np.random.randn(6, 3)
    y = np.zeros((6, 2))
    y[np.arange(6), np.random.randint(0, 2, 6)] = 1.0

    res = mlp.verify_gradients_numerically(X, y, eps=1e-7)

    assert res["all_passed"] is True
    assert res["max_relative_error"] < 1e-5
    for key in ["W1", "b1", "W2", "b2"]:
        assert res[key]["passed"] is True
        assert res[key]["relative_error"] < 1e-5


def test_mlp_training_convergence():
    """Verify mini-batch gradient descent successfully minimizes loss and trains MLP."""
    np.random.seed(42)
    # Generate simple 2-class linearly separable dataset
    X_pos = np.random.randn(50, 2) + np.array([2.0, 2.0])
    X_neg = np.random.randn(50, 2) + np.array([-2.0, -2.0])
    X = np.vstack([X_pos, X_neg])

    y_onehot = np.zeros((100, 2))
    y_onehot[:50, 0] = 1.0
    y_onehot[50:, 1] = 1.0

    mlp = MultilayerPerceptron(layer_sizes=[2, 8, 2], hidden_activation="relu", seed=42)
    history = mlp.fit(X, y_onehot, epochs=80, lr=0.1, batch_size=16)

    # Loss must substantially decrease
    assert history["train_loss"][-1] < history["train_loss"][0]
    # Training accuracy on linearly separable clusters should reach > 95%
    assert history["train_acc"][-1] >= 0.95


def test_parameter_hyperparameter_separation():
    """Verify parameter counts and parameter vs hyperparameter metadata."""
    mlp = MultilayerPerceptron(layer_sizes=[10, 20, 5], hidden_activation="relu", seed=42)
    summary = mlp.get_parameter_summary()

    # W1: 10*20 = 200, b1: 1*20 = 20
    # W2: 20*5  = 100, b2: 1*5  = 5
    # Total learnable parameters = 325
    assert summary["total_parameters"] == 325
    assert summary["total_learnable_tensors"] == 4

    # Hyperparameters must not contain weight matrices
    hypers = summary["hyperparameters"]
    assert hypers["layer_sizes"] == [10, 20, 5]
    assert hypers["hidden_activation"] == "relu"
    assert "W1" not in hypers

