"""Unit Tests for Grad-CAM Model Interpretability (REQ-08).

Verifies:
1. Hook registration and cleanup handles.
2. Gradient capture during backpropagation.
3. Output heatmap dimensions match input spatial resolution (224, 224).
4. Heatmap values are finite and strictly bounded in [0.0, 1.0].
5. Production model parameters remain unchanged (zero weight alteration).

Author: Crop Disease Detection Project Team
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.interpretability.grad_cam import GradCAM
from src.models.transfer_learning import TomatoResNet18


@pytest.fixture
def dummy_resnet():
    """Create a lightweight ResNet18 model fixture."""
    model = TomatoResNet18(num_classes=10)
    model.eval()
    return model


def test_gradcam_hook_registration_and_cleanup(dummy_resnet):
    """Verify that hooks register and can be cleanly removed without leakage."""
    target_layer = dummy_resnet.layer4[-1].conv2
    grad_cam = GradCAM(dummy_resnet, target_layer)

    assert len(grad_cam._hooks) == 2, "Expected 2 hooks (1 forward, 1 backward)"

    grad_cam.remove_hooks()
    assert len(grad_cam._hooks) == 0, "Hooks handle list should be empty after removal"


def test_gradcam_heatmap_dimensions_and_bounds(dummy_resnet):
    """Verify generated heatmap has shape (H, W) and normalized values in [0.0, 1.0]."""
    target_layer = dummy_resnet.layer4[-1].conv2
    input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)

    with GradCAM(dummy_resnet, target_layer) as grad_cam:
        heatmap, pred_class, conf = grad_cam.generate_heatmap(input_tensor)

        assert heatmap.shape == (224, 224), f"Expected (224, 224), got {heatmap.shape}"
        assert isinstance(heatmap, np.ndarray)
        assert np.all(np.isfinite(heatmap)), "Heatmap must contain only finite numbers"
        assert float(np.min(heatmap)) >= 0.0, f"Min heatmap value < 0: {np.min(heatmap)}"
        assert float(np.max(heatmap)) <= 1.0, f"Max heatmap value > 1: {np.max(heatmap)}"
        assert 0 <= pred_class < 10
        assert 0.0 <= conf <= 1.0


def test_gradcam_does_not_modify_model_weights(dummy_resnet):
    """Strictly verify that computing Grad-CAM does not mutate any model parameter."""
    initial_weights = copy.deepcopy(dummy_resnet.state_dict())
    target_layer = dummy_resnet.layer4[-1].conv2
    input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)

    with GradCAM(dummy_resnet, target_layer) as grad_cam:
        _ = grad_cam.generate_heatmap(input_tensor, target_class=3)

    final_weights = dummy_resnet.state_dict()
    for name in initial_weights:
        torch.testing.assert_close(
            initial_weights[name],
            final_weights[name],
            msg=f"Model weight {name} was unexpectedly modified by Grad-CAM!",
        )

