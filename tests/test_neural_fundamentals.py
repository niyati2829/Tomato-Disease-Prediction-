"""Unit Tests for Neural Network Fundamentals (REQ-02).

Tests:
1. McCulloch-Pitts threshold neuron logic gates (AND, OR, NOT, NAND, NOR).
2. Mathematical impossibility of single-layer XOR classification.
3. Sigmoid neuron forward activation, derivative, and gradient descent convergence.
4. Perceptron learning algorithm convergence on linearly separable gates and failure on XOR.
5. Activation functions (Step, Sigmoid, Tanh, ReLU, LeakyReLU) and analytical derivative checks.

Author: Crop Disease Detection Project Team
"""

import numpy as np
import pytest

from src.neural_fundamentals.activations import (
    leaky_relu,
    leaky_relu_derivative,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    step_function,
    tanh,
    tanh_derivative,
)
from src.neural_fundamentals.mcculloch_pitts import (
    build_and_gate,
    build_nand_gate,
    build_nor_gate,
    build_not_gate,
    build_or_gate,
    prove_xor_impossibility,
)
from src.neural_fundamentals.perceptron import Perceptron
from src.neural_fundamentals.sigmoid_neuron import SigmoidNeuron


# ---------------------------------------------------------------------------
# McCulloch-Pitts Tests
# ---------------------------------------------------------------------------


def test_mcculloch_pitts_and_gate():
    """Verify McCulloch-Pitts neuron correctly models AND logic."""
    gate = build_and_gate(n_inputs=2)
    assert gate.predict([0, 0]) == 0
    assert gate.predict([0, 1]) == 0
    assert gate.predict([1, 0]) == 0
    assert gate.predict([1, 1]) == 1


def test_mcculloch_pitts_or_gate():
    """Verify McCulloch-Pitts neuron correctly models OR logic."""
    gate = build_or_gate(n_inputs=2)
    assert gate.predict([0, 0]) == 0
    assert gate.predict([0, 1]) == 1
    assert gate.predict([1, 0]) == 1
    assert gate.predict([1, 1]) == 1


def test_mcculloch_pitts_not_gate():
    """Verify McCulloch-Pitts neuron correctly models NOT logic via inhibitory input."""
    gate = build_not_gate()
    assert gate.predict([0]) == 1
    assert gate.predict([1]) == 0


def test_mcculloch_pitts_nand_gate():
    """Verify McCulloch-Pitts neuron correctly models NAND logic."""
    gate = build_nand_gate(n_inputs=2)
    assert gate.predict([0, 0]) == 1
    assert gate.predict([0, 1]) == 1
    assert gate.predict([1, 0]) == 1
    assert gate.predict([1, 1]) == 0


def test_mcculloch_pitts_nor_gate():
    """Verify McCulloch-Pitts neuron correctly models NOR logic."""
    gate = build_nor_gate(n_inputs=2)
    assert gate.predict([0, 0]) == 1
    assert gate.predict([0, 1]) == 0
    assert gate.predict([1, 0]) == 0
    assert gate.predict([1, 1]) == 0


def test_mcculloch_pitts_xor_impossibility():
    """Verify mathematical proof that a single M-P neuron cannot solve XOR."""
    proof = prove_xor_impossibility()
    assert proof["is_solvable_single_neuron"] is False
    assert "contradiction" in proof["mathematical_contradiction"].lower()


# ---------------------------------------------------------------------------
# Sigmoid Neuron Tests
# ---------------------------------------------------------------------------


def test_sigmoid_neuron_activation_and_derivative():
    """Verify Sigmoid neuron forward output range (0, 1) and exact analytical derivative."""
    neuron = SigmoidNeuron(n_inputs=2, seed=42)
    X = np.array([[0.0, 0.0], [1.0, 1.0], [-2.0, 3.0]])

    probs = neuron.forward(X)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Analytical derivative check: d sigma / dz = sigma * (1 - sigma)
    z = np.array([-2.0, 0.0, 1.5, 3.0])
    s = sigmoid(z)
    expected_ds = s * (1.0 - s)
    actual_ds = sigmoid_derivative(z)
    np.testing.assert_allclose(actual_ds, expected_ds, atol=1e-12)


def test_sigmoid_neuron_training():
    """Verify Sigmoid neuron trains via gradient descent and loss decreases."""
    neuron = SigmoidNeuron(n_inputs=2, seed=42)
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 1.0])  # OR gate

    history = neuron.fit(X, y, epochs=200, lr=1.0)
    assert history["loss"][-1] < history["loss"][0]
    assert history["accuracy"][-1] == 1.0


# ---------------------------------------------------------------------------
# Perceptron Tests
# ---------------------------------------------------------------------------


def test_perceptron_linearly_separable():
    """Verify Perceptron learning algorithm converges in finite steps on AND gate."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    p = Perceptron(n_inputs=2, learning_rate=0.1, max_epochs=50)
    p.fit(X, y)

    assert p.converged is True
    assert p.convergence_epoch is not None
    assert p.convergence_epoch < 50
    preds = p.predict(X)
    np.testing.assert_array_equal(preds, y)


def test_perceptron_xor_failure():
    """Verify Perceptron fails to converge on XOR due to linear non-separability."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    p = Perceptron(n_inputs=2, learning_rate=0.1, max_epochs=40)
    p.fit(X, y)

    assert p.converged is False
    assert p.convergence_epoch is None


# ---------------------------------------------------------------------------
# Activation Functions Tests
# ---------------------------------------------------------------------------


def test_activation_derivatives_vs_finite_difference():
    """Verify analytical derivatives match central finite differences for smooth activations."""
    z = np.linspace(-3.0, 3.0, 25)
    eps = 1e-6

    # Sigmoid
    num_ds = (sigmoid(z + eps) - sigmoid(z - eps)) / (2.0 * eps)
    ana_ds = sigmoid_derivative(z)
    np.testing.assert_allclose(ana_ds, num_ds, atol=1e-5)

    # Tanh
    num_dt = (tanh(z + eps) - tanh(z - eps)) / (2.0 * eps)
    ana_dt = tanh_derivative(z)
    np.testing.assert_allclose(ana_dt, num_dt, atol=1e-5)

    # LeakyReLU (away from 0)
    z_nonzero = np.array([-2.5, -1.0, -0.1, 0.1, 1.0, 2.5])
    num_dl = (leaky_relu(z_nonzero + eps) - leaky_relu(z_nonzero - eps)) / (2.0 * eps)
    ana_dl = leaky_relu_derivative(z_nonzero)
    np.testing.assert_allclose(ana_dl, num_dl, atol=1e-5)

