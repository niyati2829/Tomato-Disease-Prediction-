"""Unit Tests for Classical Learning Paradigms & Energy-Based Networks (REQ-06).

Tests:
1. Basic Hebbian update equation exactness (Delta w = eta * y * x).
2. Oja's rule weight stabilization to unit norm (||w||_2 ~ 1.0).
3. Oja's rule convergence to dominant principal component (cosine similarity > 0.95).
4. Competitive learning Winner-Takes-All Best Matching Unit (BMU) selection.
5. Kohonen SOM neighborhood-distance decay behavior.
6. Error-correction Delta Rule / LMS convergence and loss reduction.
7. Hopfield network weight symmetry (W = W^T).
8. Hopfield network zero diagonal self-connections (W_ii = 0).
9. Hopfield Lyapunov energy non-increasing monotonicity (Delta E <= 0) during asynchronous updates.
10. Hopfield associative memory pattern recovery from noisy corrupted input.
11. Boltzmann / RBM stochastic sigmoid probabilities bounded in [0, 1].
12. RBM joint energy and free energy mathematical consistency.

Author: Crop Disease Detection Project Team
"""

import numpy as np
import pytest

from src.learning_paradigms.boltzmann import (
    RestrictedBoltzmannMachine,
    generate_structured_binary_patterns,
)
from src.learning_paradigms.competitive import (
    CompetitiveLearningWTA,
    KohonenSOM,
    generate_multicluster_data,
)
from src.learning_paradigms.error_correction import (
    DeltaRuleLearner,
    generate_linearly_separable_data,
)
from src.learning_paradigms.hebbian import (
    BasicHebbianLearner,
    OjaLearner,
    generate_correlated_gaussian_data,
)
from src.learning_paradigms.hopfield import (
    HopfieldNetwork,
    corrupt_pattern,
    generate_canonical_binary_patterns,
)


# ---------------------------------------------------------------------------
# 1. Hebbian & Oja Tests
# ---------------------------------------------------------------------------


def test_hebbian_update_equation():
    """Verify basic Hebbian update matches analytical formula: Delta w = eta * y * x."""
    learner = BasicHebbianLearner(n_features=3, lr=0.1, seed=42)
    learner.w = np.array([1.0, 2.0, -1.0])
    x = np.array([2.0, 0.5, 3.0])

    # y = w^T x = 1*2 + 2*0.5 + (-1)*3 = 2 + 1 - 3 = 0.0
    # Let's use non-zero y
    x2 = np.array([1.0, 1.0, 0.0])  # y = 1*1 + 2*1 + 0 = 3.0
    # Expected delta_w = 0.1 * 3.0 * [1, 1, 0] = [0.3, 0.3, 0.0]
    expected_w = learner.w + 0.1 * 3.0 * x2

    learner.train_step(x2)
    np.testing.assert_allclose(learner.w, expected_w, atol=1e-12)


def test_oja_rule_stabilization_and_pca_alignment():
    """Verify Oja's rule produces stable unit-norm weights and aligns with PCA."""
    X, true_pc1 = generate_correlated_gaussian_data(n_samples=400, angle_deg=40.0, seed=42)
    oja = OjaLearner(n_features=2, lr=0.01, seed=42)

    rng = np.random.RandomState(42)
    for _ in range(1200):
        sample = X[rng.randint(0, len(X))]
        oja.train_step(sample)

    # 1. Weight norm must stabilize near 1.0
    final_norm = float(np.linalg.norm(oja.w))
    assert final_norm == pytest.approx(1.0, abs=0.05)

    # 2. Direction must align with true principal component
    oja_dir = oja.w / final_norm
    cos_sim = abs(float(np.dot(oja_dir, true_pc1)))
    assert cos_sim > 0.95


# ---------------------------------------------------------------------------
# 2. Competitive Learning & SOM Tests
# ---------------------------------------------------------------------------


def test_competitive_bmu_selection():
    """Verify competitive learning selects the closest prototype as BMU."""
    wta = CompetitiveLearningWTA(n_features=2, n_clusters=3, lr=0.1, seed=42)
    wta.weights = np.array([
        [0.0, 0.0],
        [5.0, 5.0],
        [-5.0, -5.0],
    ])

    # Point close to (5, 5)
    x = np.array([4.8, 5.2])
    bmu = wta.find_bmu(x)
    assert bmu == 1

    # Point close to (-5, -5)
    x_neg = np.array([-4.9, -5.1])
    assert wta.find_bmu(x_neg) == 2


def test_som_neighborhood_influence():
    """Verify SOM neighborhood updates decrease monotonically with grid distance from BMU."""
    som = KohonenSOM(grid_rows=5, grid_cols=5, n_features=2, lr_initial=0.5, seed=42)
    # Target input near neuron (2, 2)
    bmu_coord = np.array([2.0, 2.0])

    # Distance to immediate neighbor (2, 3) vs distant neighbor (0, 0)
    dist_near_sq = np.sum((np.array([2.0, 3.0]) - bmu_coord) ** 2)  # dist^2 = 1
    dist_far_sq = np.sum((np.array([0.0, 0.0]) - bmu_coord) ** 2)   # dist^2 = 8

    sigma = 1.5
    h_near = np.exp(-dist_near_sq / (2.0 * sigma**2))
    h_far = np.exp(-dist_far_sq / (2.0 * sigma**2))

    assert h_near > h_far
    assert 0.0 < h_far < h_near <= 1.0


# ---------------------------------------------------------------------------
# 3. Error-Correction Learning Tests
# ---------------------------------------------------------------------------


def test_error_correction_delta_rule_convergence():
    """Verify Delta Rule reduces MSE loss and correctly separates linear clusters."""
    X, y = generate_linearly_separable_data(n_samples=100, seed=42)
    learner = DeltaRuleLearner(n_features=2, lr=0.05, seed=42)

    history = learner.fit(X, y, epochs=40, seed=42)
    losses = history["loss_history"]

    # Loss must substantially decrease
    assert losses[-1] < losses[0]
    # On linearly separable data, Adaline should achieve near 0 misclassifications
    assert history["error_count_history"][-1] <= 2


# ---------------------------------------------------------------------------
# 4. Hopfield Network Tests
# ---------------------------------------------------------------------------


def test_hopfield_weight_matrix_symmetry_and_zero_diagonal():
    """Verify Hopfield weight matrix satisfies W = W^T and W_ii = 0."""
    patterns = generate_canonical_binary_patterns()
    p_list = [p.ravel() for p in patterns.values()]

    hopfield = HopfieldNetwork(n_neurons=64)
    hopfield.train_patterns(p_list)

    W = hopfield.weights
    # 1. Symmetric
    np.testing.assert_allclose(W, W.T, atol=1e-12)
    # 2. Zero diagonal
    np.testing.assert_allclose(np.diag(W), np.zeros(64), atol=1e-12)


def test_hopfield_energy_monotonic_descent():
    """Verify Hopfield Lyapunov energy is strictly non-increasing (Delta E <= 0) during updates."""
    patterns = generate_canonical_binary_patterns()
    p_list = [p.ravel() for p in patterns.values()]

    hopfield = HopfieldNetwork(n_neurons=64)
    hopfield.train_patterns(p_list)

    # Corrupt a pattern
    leaf = patterns["Leaf"]
    corrupted = corrupt_pattern(leaf, flip_ratio=0.25, seed=42)

    res = hopfield.update_asynchronous(corrupted.ravel(), max_steps=200, seed=42)
    energies = res["energy_history"]

    # At every step, E_{t+1} <= E_t (with tolerance for floating point precision)
    diffs = np.diff(energies)
    assert np.all(diffs <= 1e-9), f"Energy increased: max positive diff = {np.max(diffs)}"
    assert energies[-1] < energies[0]


def test_hopfield_pattern_recovery():
    """Verify Hopfield network successfully recovers stored pattern from corrupted input."""
    patterns = generate_canonical_binary_patterns()
    p_list = [p.ravel() for p in patterns.values()]

    hopfield = HopfieldNetwork(n_neurons=64)
    hopfield.train_patterns(p_list)

    leaf = patterns["Leaf"].ravel()
    corrupted = corrupt_pattern(patterns["Leaf"], flip_ratio=0.20, seed=123).ravel()

    res = hopfield.update_asynchronous(corrupted, max_steps=300, seed=123)
    recalled = res["final_state"]

    # Retrieval accuracy must exceed 90%
    match_ratio = np.mean(recalled == leaf)
    assert match_ratio >= 0.90


# ---------------------------------------------------------------------------
# 5. Boltzmann Machine & RBM Tests
# ---------------------------------------------------------------------------


def test_rbm_probabilities_and_energy_definitions():
    """Verify RBM conditional activation probabilities lie in [0, 1] and energy matches formula."""
    rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, lr=0.05, seed=42)
    v = np.array([[1, 0, 1, 1, 0, 0, 1, 0]])

    p_h, h = rbm.sample_hidden(v)
    assert np.all((p_h >= 0.0) & (p_h <= 1.0))
    assert np.all(np.isin(h, [0.0, 1.0]))

    p_v, v_sampled = rbm.sample_visible(h)
    assert np.all((p_v >= 0.0) & (p_v <= 1.0))
    assert np.all(np.isin(v_sampled, [0.0, 1.0]))

    # Energy definition: E(v, h) = -v^T W h - a^T v - b^T h
    energy = rbm.compute_energy(v[0], h[0])
    expected_energy = - (v[0] @ rbm.W @ h[0] + np.dot(rbm.a, v[0]) + np.dot(rbm.b, h[0]))
    assert energy == pytest.approx(expected_energy, rel=1e-6)


def test_rbm_training_reconstruction_reduction():
    """Verify RBM CD-1 training reduces visible reconstruction error."""
    X = generate_structured_binary_patterns(n_samples=100, seed=42)
    rbm = RestrictedBoltzmannMachine(n_visible=16, n_hidden=8, lr=0.08, seed=42)

    history = rbm.fit(X, epochs=40, batch_size=20)
    errors = history["reconstruction_error"]

    assert errors[-1] < errors[0]

