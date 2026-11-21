"""Unit tests for corruption transforms and FGSM adversarial perturbations."""

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from src.robustness.corruptions import (
    CORRUPTION_REGISTRY,
    apply_corruption,
    apply_gaussian_noise,
    apply_gaussian_blur,
    apply_motion_blur,
    apply_brightness_increase,
    apply_brightness_decrease,
    apply_contrast_change,
    apply_jpeg_compression,
    apply_resolution_reduction,
)
from src.robustness.adversarial_tests import fgsm_attack, unnormalize_imagenet, normalize_imagenet


@pytest.fixture
def dummy_leaf():
    """Create a synthetic RGB test image."""
    arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_corruption_registry_coverage():
    """Verify registry contains all 8 required corruption modes."""
    required = {
        "gaussian_noise",
        "gaussian_blur",
        "motion_blur",
        "brightness_up",
        "brightness_down",
        "contrast",
        "jpeg_compression",
        "resolution_reduction",
    }
    assert required.issubset(set(CORRUPTION_REGISTRY.keys()))


def test_corruptions_return_pil_with_valid_size(dummy_leaf):
    """Every corruption must preserve (224, 224, 3) dimensions and valid RGB range."""
    for name, func in CORRUPTION_REGISTRY.items():
        corrupted = func(dummy_leaf, severity=2)
        assert isinstance(corrupted, Image.Image)
        assert corrupted.size == (224, 224)
        assert corrupted.mode == "RGB"


def test_severity_monotonicity(dummy_leaf):
    """Severity 3 should induce larger pixel deviation from clean than Severity 1."""
    clean_arr = np.array(dummy_leaf, dtype=np.float32)

    # Test Gaussian noise severity scaling
    noisy1 = np.array(apply_gaussian_noise(dummy_leaf, severity=1), dtype=np.float32)
    noisy3 = np.array(apply_gaussian_noise(dummy_leaf, severity=3), dtype=np.float32)

    diff1 = np.mean(np.abs(clean_arr - noisy1))
    diff3 = np.mean(np.abs(clean_arr - noisy3))
    assert diff3 > diff1

    # Test Gaussian blur severity scaling
    blur1 = np.array(apply_gaussian_blur(dummy_leaf, severity=1), dtype=np.float32)
    blur3 = np.array(apply_gaussian_blur(dummy_leaf, severity=3), dtype=np.float32)
    assert np.mean(np.abs(clean_arr - blur3)) > np.mean(np.abs(clean_arr - blur1))


def test_invalid_corruption_raises_error(dummy_leaf):
    """Querying an unknown corruption must raise KeyError."""
    with pytest.raises(KeyError):
        apply_corruption(dummy_leaf, "non_existent_mode", severity=1)


def test_fgsm_attack_perturbation_bound():
    """FGSM adversarial perturbation must respect the specified epsilon bound."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 4),
    )
    model.eval()

    # Create dummy batch of normalized images
    raw_images = torch.rand(2, 3, 32, 32)
    norm_images = normalize_imagenet(raw_images)
    labels = torch.tensor([0, 2])

    eps = 0.05
    adv_norm = fgsm_attack(model, norm_images, labels, epsilon=eps)

    # Unnormalize both to inspect pixel space delta
    raw_orig = unnormalize_imagenet(norm_images)
    raw_adv = unnormalize_imagenet(adv_norm)

    delta = torch.abs(raw_adv - raw_orig)
    assert torch.all(delta <= eps + 1e-4)
    assert torch.all(raw_adv >= 0.0) and torch.all(raw_adv <= 1.0)
