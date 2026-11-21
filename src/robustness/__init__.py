"""Model robustness evaluation package.

Provides systematic synthetic corruptions (noise, blur, illumination, compression, resolution)
and FGSM adversarial perturbations to test model out-of-distribution stability.
"""

from .corruptions import (
    apply_gaussian_noise,
    apply_gaussian_blur,
    apply_motion_blur,
    apply_brightness_change,
    apply_contrast_change,
    apply_jpeg_compression,
    apply_resolution_reduction,
    apply_corruption,
    CORRUPTION_REGISTRY,
)
from .adversarial_tests import fgsm_attack, evaluate_fgsm_robustness

__all__ = [
    "apply_gaussian_noise",
    "apply_gaussian_blur",
    "apply_motion_blur",
    "apply_brightness_change",
    "apply_contrast_change",
    "apply_jpeg_compression",
    "apply_resolution_reduction",
    "apply_corruption",
    "CORRUPTION_REGISTRY",
    "fgsm_attack",
    "evaluate_fgsm_robustness",
]
