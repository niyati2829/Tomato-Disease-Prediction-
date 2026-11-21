"""Reproducibility & Deterministic Execution Utility.

Fixes random seeds across Python, NumPy, and PyTorch (CPU and MPS/CUDA)
to ensure experiments are reproducible.

Author: Crop Disease Detection Project Team
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across all libraries to guarantee reproducible experiments.

    Args:
        seed: Integer seed value (default 42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Note: On Apple Silicon MPS, torch.manual_seed seeds both CPU and MPS generators.
