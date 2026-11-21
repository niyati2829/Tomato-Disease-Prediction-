"""ResNet18 Hierarchical Feature Map Extraction & Visualization (REQ-08).

Extracts and visualizes intermediate activation maps from multiple stages
of the production ResNet18 hierarchy:
1. `conv1`: Initial 7x7 spatial filter bank (64 channels, 112x112) - edge and color contrast.
2. `layer1`: Low-level residual blocks (64 channels, 56x56) - localized texture and leaf venation.
3. `layer2`: Mid-level residual blocks (128 channels, 28x28) - leaf margins and spot contours.
4. `layer3`: High-level residual blocks (256 channels, 14x14) - complex lesion clusters.
5. `layer4`: Deep semantic blocks (512 channels, 7x7) - global pathological disease signatures.

Author: Crop Disease Detection Project Team
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class ResNetFeatureExtractor:
    """Non-destructive forward-hook extractor for intermediate ResNet18 activations."""

    TARGET_LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4"]

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on key hierarchical stages."""
        for name, module in self.model.named_children():
            if name in self.TARGET_LAYERS:
                def get_hook(layer_name: str):
                    def hook(m: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
                        self.feature_maps[layer_name] = out.detach().cpu()
                    return hook
                self._hooks.append(module.register_forward_hook(get_hook(name)))

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self) -> "ResNetFeatureExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()

    def extract(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform forward pass and return dictionary of feature tensors."""
        self.feature_maps.clear()
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        return self.feature_maps


def plot_feature_map_hierarchy(
    rgb_image: np.ndarray,
    feature_maps: Dict[str, torch.Tensor],
    n_channels_per_layer: int = 6,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot multi-stage feature map evolution across ResNet18 depth.

    Args:
        rgb_image: uint8 RGB original input image of shape (H, W, 3).
        feature_maps: Dictionary of extracted feature tensors per stage.
        n_channels_per_layer: Number of sample channel slices to display per layer.
        save_path: Optional path to save figure.

    Returns:
        matplotlib Figure object.
    """
    layers_ordered = ["conv1", "layer1", "layer2", "layer3", "layer4"]
    n_layers = len([l for l in layers_ordered if l in feature_maps])

    fig, axes = plt.subplots(n_layers, n_channels_per_layer + 1, figsize=(18, 2.8 * n_layers), dpi=300)

    layer_descriptions = {
        "conv1": "conv1 (7x7, 64ch, 112x112)\nEdges & Color Contrast",
        "layer1": "layer1 (64ch, 56x56)\nLeaf Veins & Textures",
        "layer2": "layer2 (128ch, 28x28)\nMarginal Contours & Spots",
        "layer3": "layer3 (256ch, 14x14)\nLesion Clusters & Patches",
        "layer4": "layer4 (512ch, 7x7)\nDeep Semantic Pathology",
    }

    for row_idx, layer_name in enumerate(layers_ordered):
        if layer_name not in feature_maps:
            continue

        feat = feature_maps[layer_name][0]  # Shape (C, H, W)
        num_c, h, w = feat.shape

        # Column 0: Original input thumbnail with layer description
        axes[row_idx, 0].imshow(rgb_image)
        axes[row_idx, 0].set_title(layer_descriptions.get(layer_name, layer_name), fontsize=9.5, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Select evenly spaced channels
        channel_indices = np.linspace(0, num_c - 1, n_channels_per_layer, dtype=int)

        for col_idx, c_idx in enumerate(channel_indices, start=1):
            channel_slice = feat[c_idx].numpy()

            # Normalization for display
            vmin, vmax = np.percentile(channel_slice, [2, 98])
            if vmax <= vmin:
                vmax = vmin + 1e-5

            im = axes[row_idx, col_idx].imshow(channel_slice, cmap="magma", vmin=vmin, vmax=vmax)
            axes[row_idx, col_idx].set_title(f"Ch #{c_idx} ({h}x{w})", fontsize=8.5)
            axes[row_idx, col_idx].axis("off")

    fig.suptitle(
        "Hierarchical Deep Feature Representations in ResNet18 Across Network Depth",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")

    return fig

