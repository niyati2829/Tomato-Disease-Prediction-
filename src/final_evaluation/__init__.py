"""Comprehensive final error analysis and evaluation suite."""

from .error_analysis import (
    analyze_classification_errors,
    plot_detailed_confusion_matrix,
    plot_error_confidence_distribution,
    plot_top_confusions,
)

__all__ = [
    "analyze_classification_errors",
    "plot_detailed_confusion_matrix",
    "plot_error_confidence_distribution",
    "plot_top_confusions",
]
