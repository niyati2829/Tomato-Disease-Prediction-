"""Unit and property tests for calibration metrics and temperature scaling."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.calibration.calibration_metrics import (
    compute_brier_score,
    compute_ece,
    compute_mce,
    compute_nll,
    compute_reliability_bins,
)
from src.calibration.temperature_scaling import ModelWithTemperature, TemperatureScaler


def test_perfect_calibration_ece():
    """A perfectly calibrated predictor should have zero ECE."""
    # 100 samples with 1.0 confidence on true label
    probs = np.zeros((100, 2))
    probs[:, 0] = 1.0
    labels = np.zeros(100, dtype=int)

    ece = compute_ece(probs, labels, n_bins=10)
    assert pytest.approx(ece, abs=1e-5) == 0.0


def test_overconfident_ece_and_mce():
    """An overconfident wrong predictor should produce high ECE and MCE."""
    # Predict class 0 with 0.99 confidence, but true label is 1
    probs = np.zeros((50, 2))
    probs[:, 0] = 0.99
    probs[:, 1] = 0.01
    labels = np.ones(50, dtype=int)

    ece = compute_ece(probs, labels, n_bins=10)
    mce = compute_mce(probs, labels, n_bins=10)

    assert ece > 0.95
    assert mce > 0.95


def test_brier_score_properties():
    """Perfect forecasts achieve Brier score 0; worst forecasts achieve 2."""
    probs_perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert pytest.approx(compute_brier_score(probs_perfect, labels), abs=1e-6) == 0.0

    probs_inverted = np.array([[0.0, 1.0], [1.0, 0.0]])
    # (1-0)^2 + (0-1)^2 = 2.0 per sample
    assert pytest.approx(compute_brier_score(probs_inverted, labels), abs=1e-6) == 2.0


def test_nll_properties():
    """NLL equals cross-entropy."""
    probs = np.array([[0.8, 0.2], [0.1, 0.9]])
    labels = np.array([0, 1])
    expected_nll = -0.5 * (np.log(0.8) + np.log(0.9))
    assert pytest.approx(compute_nll(probs, labels), abs=1e-5) == expected_nll


def test_reliability_bins_structure():
    """Verify bin statistics outputs and shapes."""
    probs = np.random.dirichlet(np.ones(5), size=200)
    labels = np.random.randint(0, 5, size=200)
    stats = compute_reliability_bins(probs, labels, n_bins=10)

    assert len(stats["bin_edges"]) == 11
    assert len(stats["bin_centers"]) == 10
    assert len(stats["bin_accuracies"]) == 10
    assert len(stats["bin_confidences"]) == 10
    assert len(stats["bin_counts"]) == 10
    assert np.sum(stats["bin_counts"]) == 200


def test_temperature_scaler_optimization():
    """Temperature scaling should optimize T > 1 on overconfident uncalibrated logits."""
    torch.manual_seed(42)
    # Generate overconfident logits: large magnitude
    logits = torch.randn(300, 4) * 5.0
    labels = torch.randint(0, 4, (300,))

    scaler = TemperatureScaler(initial_temp=1.0)
    opt_t = scaler.fit(logits, labels)

    assert opt_t > 0.0
    cal_probs = scaler.transform(logits)
    assert cal_probs.shape == (300, 4)
    assert np.allclose(cal_probs.sum(axis=1), 1.0, atol=1e-5)


def test_model_with_temperature_module():
    """ModelWithTemperature wrapper should scale logits by T."""
    base_model = nn.Linear(10, 3)
    wrapper = ModelWithTemperature(base_model, initial_temperature=2.0)

    dummy_x = torch.randn(4, 10)
    raw_logits = base_model(dummy_x)
    scaled_logits = wrapper(dummy_x)

    assert torch.allclose(scaled_logits, raw_logits / 2.0, atol=1e-5)
