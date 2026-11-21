"""Optimization Dynamics & Algorithms Package (REQ-04).

Implements from scratch:
- Batch GD, Stochastic GD (SGD), and Mini-batch GD
- Advanced optimizers: Momentum, Nesterov Accelerated Gradient (NAG), Adam
- Learning rate stability and divergence dynamics
- 2D and 3D loss landscape trajectory visualizers

Author: Crop Disease Detection Project Team
"""

from src.optimization.gradient_descent import (
    BatchGradientDescent,
    MiniBatchGradientDescent,
    StochasticGradientDescent,
    compare_gd_variants,
)
from src.optimization.optimizers import (
    AdamOptimizer,
    GradientDescentOptimizer,
    MomentumOptimizer,
    NesterovOptimizer,
    compare_advanced_optimizers,
)

__all__ = [
    "BatchGradientDescent",
    "StochasticGradientDescent",
    "MiniBatchGradientDescent",
    "compare_gd_variants",
    "GradientDescentOptimizer",
    "MomentumOptimizer",
    "NesterovOptimizer",
    "AdamOptimizer",
    "compare_advanced_optimizers",
]

