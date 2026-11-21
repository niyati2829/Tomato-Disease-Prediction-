"""Regularization, Weight Initialization & Normalization Package (REQ-05).

Implements from scratch:
- Weight initialization strategies (Zero, Random, Xavier/Glorot, He/Kaiming)
- Layer-by-layer activation and gradient variance tracking
- Batch Normalization vs Layer Normalization comparison
- Regularizers: L1, L2 (Weight Decay), and Inverted Dropout
- Controlled ablation studies

Author: Crop Disease Detection Project Team
"""

from src.regularization.initialization import (
    compare_initialization_schemes,
    he_initialization,
    plot_initialization_diagnostics,
    random_initialization,
    simulate_deep_network_flow,
    xavier_initialization,
    zero_initialization,
)
from src.regularization.normalization import (
    BatchNorm1dNumPy,
    LayerNorm1dNumPy,
    compare_batch_and_layer_norm,
    plot_normalization_comparison,
)
from src.regularization.regularizers import (
    DropoutLayer,
    L1Regularizer,
    L2Regularizer,
    compare_regularization_schemes,
)

__all__ = [
    "zero_initialization",
    "random_initialization",
    "xavier_initialization",
    "he_initialization",
    "simulate_deep_network_flow",
    "compare_initialization_schemes",
    "plot_initialization_diagnostics",
    "BatchNorm1dNumPy",
    "LayerNorm1dNumPy",
    "compare_batch_and_layer_norm",
    "plot_normalization_comparison",
    "L1Regularizer",
    "L2Regularizer",
    "DropoutLayer",
    "compare_regularization_schemes",
]

