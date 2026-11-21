"""Device Management & Hardware Acceleration Utility.

Detects and selects optimal compute device:
1. Apple Silicon GPU acceleration via Metal Performance Shaders (`mps`) on macOS M-series.
2. NVIDIA CUDA acceleration (`cuda`) if running in GPU environment.
3. Standard CPU fallback (`cpu`).

Author: Crop Disease Detection Project Team
"""

import platform
from typing import Dict

import torch


def get_device(preferred: str = "mps") -> torch.device:
    """Select the optimal compute device.

    Args:
        preferred: Preferred device string ('mps', 'cuda', or 'cpu'). Defaults to 'mps'.

    Returns:
        torch.device instance.
    """
    if preferred == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                return torch.device("mps")
    elif preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")

    return torch.device("cpu")


def get_device_info(preferred: str = "mps") -> Dict[str, any]:
    """Retrieve diagnostic information regarding hardware accelerators.

    Returns:
        Dictionary of device specifications.
    """
    device = get_device(preferred=preferred)
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()

    info = {
        "selected_device": str(device),
        "device_type": device.type,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "mps_available": mps_available,
        "cuda_available": cuda_available,
    }

    if device.type == "mps":
        info["device_name"] = "Apple Silicon GPU (Metal Performance Shaders)"
    elif device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(0)
    else:
        info["device_name"] = "Host CPU"

    return info
