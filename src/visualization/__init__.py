"""Convolution & Feature Map Visualizations Package (REQ-08).

Provides:
- 2D discrete convolution from scratch in NumPy with stride & padding support.
- Standard classical spatial filter kernels (Sobel X/Y, Gaussian, Laplacian, Sharpen).
- Deep feature map extraction and hierarchical visualization from ResNet18.
"""

from src.visualization.convolution_visualizer import (
    convolve2d_numpy,
    get_classical_kernels,
    visualize_classical_filters,
)
from src.visualization.feature_maps import ResNetFeatureExtractor, plot_feature_map_hierarchy

__all__ = [
    "convolve2d_numpy",
    "get_classical_kernels",
    "visualize_classical_filters",
    "ResNetFeatureExtractor",
    "plot_feature_map_hierarchy",
]

