"""Systematic image corruptions for computer vision robustness benchmarking.

Implements 8 realistic environmental and sensor degradation modes across 3 severity levels:
1. Gaussian Noise (sensor thermal noise, low-light sensor grain)
2. Gaussian Blur (camera defocus, shallow depth of field)
3. Motion Blur (wind movement in field, shaking handheld mobile camera)
4. Brightness Increase (direct harsh sunlight, overexposure)
5. Brightness Decrease (heavy cloud cover, deep canopy shadow)
6. Contrast Reduction (morning mist, agricultural fog, haze)
7. JPEG Compression (aggressive mobile compression, low-bandwidth edge transfer)
8. Resolution Reduction (distance from leaf, low-end mobile camera sensors)
"""

import io
from typing import Callable, Dict, Tuple, Union
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import torch


def _to_pil(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
    """Ensure image is a PIL RGB Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        return Image.fromarray(image).convert("RGB")
    elif isinstance(image, torch.Tensor):
        # Shape (C, H, W) or (H, W, C)
        arr = image.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in [1, 3]:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        return Image.fromarray(arr).convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


# ---------------------------------------------------------------------------
# 1. Gaussian Noise
# ---------------------------------------------------------------------------
def apply_gaussian_noise(image: Union[Image.Image, np.ndarray, torch.Tensor], severity: int = 1) -> Image.Image:
    """Add zero-mean Gaussian noise."""
    img_pil = _to_pil(image)
    arr = np.array(img_pil, dtype=np.float32) / 255.0
    sigma_map = {1: 0.05, 2: 0.12, 3: 0.22}
    sigma = sigma_map.get(severity, 0.08)

    noise = np.random.normal(0.0, sigma, arr.shape)
    noisy_arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy_arr * 255.0).astype(np.uint8))


# ---------------------------------------------------------------------------
# 2. Gaussian Blur
# ---------------------------------------------------------------------------
def apply_gaussian_blur(image: Union[Image.Image, np.ndarray, torch.Tensor], severity: int = 1) -> Image.Image:
    """Apply Gaussian blur simulating optical defocus."""
    img_pil = _to_pil(image)
    radius_map = {1: 1.2, 2: 2.5, 3: 4.5}
    radius = radius_map.get(severity, 2.0)
    return img_pil.filter(ImageFilter.GaussianBlur(radius=radius))


# ---------------------------------------------------------------------------
# 3. Motion Blur
# ---------------------------------------------------------------------------
def apply_motion_blur(image: Union[Image.Image, np.ndarray, torch.Tensor], severity: int = 1) -> Image.Image:
    """Simulate motion blur caused by handheld phone jitter or wind."""
    img_pil = _to_pil(image)
    arr = np.array(img_pil)
    kernel_size_map = {1: 5, 2: 11, 3: 19}
    k = kernel_size_map.get(severity, 9)

    kernel = np.zeros((k, k), dtype=np.float32)
    # Diagonal motion blur
    np.fill_diagonal(kernel, 1.0 / k)
    blurred = cv2.filter2D(arr, -1, kernel)
    return Image.fromarray(blurred)


# ---------------------------------------------------------------------------
# 4. Brightness Increase
# ---------------------------------------------------------------------------
def apply_brightness_increase(image: Union[Image.Image, np.ndarray, torch.Tensor], severity: int = 1) -> Image.Image:
    """Simulate overexposure or harsh direct field sunlight."""
    img_pil = _to_pil(image)
    factor_map = {1: 1.25, 2: 1.55, 3: 1.90}
    factor = factor_map.get(severity, 1.4)
    enhancer = ImageEnhance.Brightness(img_pil)
    return enhancer.enhance(factor)


# ---------------------------------------------------------------------------
# 5. Brightness Decrease (Dimming)
# ---------------------------------------------------------------------------
def apply_brightness_decrease(image: Union[Image.Image, np.ndarray, torch.Tensor], severity: int = 1) -> Image.Image:
    """Simulate underexposure, heavy overcast, or deep leaf canopy shadows."""
    img_pil = _to_pil(image)
    factor_map = {1: 0.75, 2: 0.45, 3: 0.25}
    factor = factor_map.get(severity, 0.5)
    enhancer = ImageEnhance.Brightness(img_pil)
    return enhancer.enhance(factor)


# ---------------------------------------------------------------------------
# 6. Contrast Reduction
# ---------------------------------------------------------------------------
def apply_contrast_change(image: Union[Image.Image, np.ndarray, torch.Tensor], severity: int = 1) -> Image.Image:
    """Simulate morning mist, greenhouse humidity haze, or low-contrast lighting."""
    img_pil = _to_pil(image)
    factor_map = {1: 0.65, 2: 0.35, 3: 0.15}
    factor = factor_map.get(severity, 0.4)
    enhancer = ImageEnhance.Contrast(img_pil)
    return enhancer.enhance(factor)


# ---------------------------------------------------------------------------
# 7. JPEG Compression Artifacts
# ---------------------------------------------------------------------------
def apply_jpeg_compression(image: Union[Image.Image, np.ndarray, torch.Tensor], severity: int = 1) -> Image.Image:
    """Simulate edge network bandwidth loss and lossy DCT block artifacts."""
    img_pil = _to_pil(image)
    quality_map = {1: 45, 2: 20, 3: 8}
    quality = quality_map.get(severity, 25)

    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy().convert("RGB")


# ---------------------------------------------------------------------------
# 8. Resolution Reduction (Downsampling & Upsampling)
# ---------------------------------------------------------------------------
def apply_resolution_reduction(image: Union[Image.Image, np.ndarray, torch.Tensor], severity: int = 1) -> Image.Image:
    """Simulate low-resolution mobile sensors and long-distance capture."""
    img_pil = _to_pil(image)
    orig_w, orig_h = img_pil.size
    scale_map = {1: 0.50, 2: 0.25, 3: 0.125}
    scale = scale_map.get(severity, 0.25)

    low_w = max(int(orig_w * scale), 8)
    low_h = max(int(orig_h * scale), 8)

    downscaled = img_pil.resize((low_w, low_h), resample=Image.Resampling.BILINEAR)
    upscaled = downscaled.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)
    return upscaled


# Alias for unified interface
apply_brightness_change = apply_brightness_increase

CORRUPTION_REGISTRY: Dict[str, Callable[[Union[Image.Image, np.ndarray, torch.Tensor], int], Image.Image]] = {
    "gaussian_noise": apply_gaussian_noise,
    "gaussian_blur": apply_gaussian_blur,
    "motion_blur": apply_motion_blur,
    "brightness_up": apply_brightness_increase,
    "brightness_down": apply_brightness_decrease,
    "contrast": apply_contrast_change,
    "jpeg_compression": apply_jpeg_compression,
    "resolution_reduction": apply_resolution_reduction,
}


def apply_corruption(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    corruption_name: str,
    severity: int = 1,
) -> Image.Image:
    """Apply named corruption from registry at specified severity (1, 2, or 3)."""
    if corruption_name not in CORRUPTION_REGISTRY:
        raise KeyError(f"Unknown corruption '{corruption_name}'. Available: {list(CORRUPTION_REGISTRY.keys())}")
    return CORRUPTION_REGISTRY[corruption_name](image, severity=severity)
