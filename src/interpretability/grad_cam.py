"""Grad-CAM: Gradient-Weighted Class Activation Mapping (REQ-08).

Mathematical Formulation:
1. Feature map extraction:
       A = f(x; theta)   where A in R^{B x C x H x W}
2. Output score for class c:
       y^c = Model(x)_c
3. Channel-wise importance weights via global average pooling of gradients:
       alpha_k^c = (1 / Z) * sum_{i=1}^H sum_{j=1}^W (partial y^c / partial A_{ij}^k)
4. Class-discriminative activation map:
       L_{Grad-CAM}^c = ReLU(sum_{k=1}^C alpha_k^c * A^k)
5. Heatmap normalization:
       Heatmap = (L - min(L)) / (max(L) - min(L) + 1e-8)

Author: Crop Disease Detection Project Team
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Computes Grad-CAM heatmaps for arbitrary PyTorch convolutional models."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Initialize Grad-CAM with target model and convolutional layer.

        Args:
            model: PyTorch classification model (e.g. TomatoResNet18).
            target_layer: Target convolutional layer (e.g. model.layer4[-1]).
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> None:
            # grad_output[0] contains gradients w.r.t the layer's output feature map
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Remove registered hooks to prevent memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Generate Grad-CAM heatmap for an input tensor.

        Args:
            input_tensor: 4D tensor (1, C, H, W).
            target_class: Optional ground truth or specific class index.
                          If None, uses the model's highest scoring predicted class.

        Returns:
            Tuple of:
                - 2D normalized heatmap (H, W) in [0, 1] as float32 numpy array.
                - Predicted class index.
                - Predicted class confidence / probability.
        """
        if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
            raise ValueError(f"GradCAM expects input tensor of shape (1, C, H, W), got {input_tensor.shape}")

        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        pred_class = int(torch.argmax(logits, dim=1).item())
        confidence = float(probabilities[0, pred_class].item())

        class_idx = target_class if target_class is not None else pred_class

        # Target score for backpropagation
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations or gradients not captured. Ensure target layer was invoked in forward pass.")

        # Gradients: (1, C, H_feat, W_feat)
        # Activations: (1, C, H_feat, W_feat)
        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients over spatial dimensions (H_feat, W_feat)
        # alpha_k: (1, C, 1, 1)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps: sum_k alpha_k * A^k
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)

        # Apply ReLU: only positive contributions to the class of interest are retained
        cam = F.relu(cam)

        # Upsample to match original input image resolution (H, W)
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Convert to numpy and normalize to [0, 1]
        cam_np = cam.squeeze().cpu().numpy()
        cam_min = float(np.min(cam_np))
        cam_max = float(np.max(cam_np))
        if cam_max - cam_min > 1e-8:
            heatmap = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam_np)

        return heatmap.astype(np.float32), pred_class, confidence

