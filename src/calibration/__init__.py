"""Model calibration package for crop disease classification.

Provides temperature scaling, reliability diagrams, and calibration metrics
(ECE, MCE, Brier score, Negative Log-Likelihood).
"""

from .calibration_metrics import (
    compute_calibration_metrics,
    compute_ece,
    compute_mce,
    compute_brier_score,
    compute_nll,
    compute_reliability_bins,
)
from .temperature_scaling import ModelWithTemperature, TemperatureScaler

__all__ = [
    "compute_calibration_metrics",
    "compute_ece",
    "compute_mce",
    "compute_brier_score",
    "compute_nll",
    "compute_reliability_bins",
    "ModelWithTemperature",
    "TemperatureScaler",
]
