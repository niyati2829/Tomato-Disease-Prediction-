"""Unit Tests for 2D Spatial Convolution & Deep Feature Extraction (REQ-08).

Verifies:
1. Output dimensions across stride and padding configurations.
2. Exact numerical parity between pure NumPy convolve2d and PyTorch F.conv2d.
3. ResNetFeatureExtractor captures intermediate activations across all 5 stages.

Author: Crop Disease Detection Project Team
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.models.transfer_learning import TomatoResNet18
from src.visualization.convolution_visualizer import convolve2d_numpy, get_classical_kernels
from src.visualization.feature_maps import ResNetFeatureExtractor


@pytest.mark.parametrize(
    "H, W, K, stride, padding, expected_shape",
    [
        (8, 8, 3, 1, 0, (6, 6)),
        (8, 8, 3, 1, 1, (8, 8)),
        (10, 10, 3, 2, 1, (5, 5)),
        (16, 16, 5, 2, 0, (6, 6)),
    ],
)
def test_convolve2d_output_dimensions(H, W, K, stride, padding, expected_shape):
    """Verify spatial output geometry formula floor((H + 2p - K)/s) + 1."""
    img = np.random.randn(H, W)
    kernel = np.random.randn(K, K)
    out = convolve2d_numpy(img, kernel, stride=stride, padding=padding)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"


def test_convolve2d_numerical_parity_with_pytorch():
    """Verify pure NumPy 2D convolution matches PyTorch F.conv2d within 1e-10."""
    np.random.seed(42)
    img_np = np.random.randn(12, 12)
    kernel_np = get_classical_kernels()["Sobel X (Vertical Veins)"]

    for stride, padding in [(1, 0), (1, 1), (2, 1)]:
        out_np = convolve2d_numpy(img_np, kernel_np, stride=stride, padding=padding)

        # PyTorch reference: shape (B=1, C=1, H, W)
        img_pt = torch.tensor(img_np, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        kernel_pt = torch.tensor(kernel_np, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        out_pt = F.conv2d(img_pt, kernel_pt, stride=stride, padding=padding).squeeze().numpy()

        np.testing.assert_allclose(out_np, out_pt, atol=1e-10)


def test_resnet_feature_map_extractor_hierarchy():
    """Verify forward hooks extract feature maps from all 5 ResNet18 stages."""
    model = TomatoResNet18(num_classes=10)
    model.eval()
    x = torch.randn(1, 3, 224, 224)

    with ResNetFeatureExtractor(model) as extractor:
        feats = extractor.extract(x)

    expected_shapes = {
        "conv1": (1, 64, 112, 112),
        "layer1": (1, 64, 56, 56),
        "layer2": (1, 128, 28, 28),
        "layer3": (1, 256, 14, 14),
        "layer4": (1, 512, 7, 7),
    }

    for layer_name, exp_shape in expected_shapes.items():
        assert layer_name in feats, f"Missing feature map for {layer_name}"
        assert feats[layer_name].shape == exp_shape, f"Shape mismatch for {layer_name}"

    assert len(extractor._hooks) == 0, "Hooks must be cleaned up on context exit"

