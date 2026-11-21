"""Model Interpretability Package (REQ-08).

Provides:
- GradCAM: Gradient-weighted Class Activation Mapping for CNNs (ResNet18).
- Visualizer: Heatmap generation, colormap application, and overlay.
"""

from src.interpretability.grad_cam import GradCAM
from src.interpretability.visualization import overlay_heatmap, plot_gradcam_explanation

__all__ = ["GradCAM", "overlay_heatmap", "plot_gradcam_explanation"]

