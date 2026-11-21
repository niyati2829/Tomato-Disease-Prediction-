"""Model parameter and memory footprint profiler."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


def profile_model_memory(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, any]:
    """Profile parameters, disk footprint, and activation memory of a model.

    Args:
        model: PyTorch model.
        input_shape: (B, C, H, W) of input.
        checkpoint_path: Optional path to saved checkpoint file on disk.
        device: Device to profile on.

    Returns:
        Dict of memory and parameter metrics.
    """
    model.eval()
    model.to(device)

    # 1. Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Parameter memory in MB (FP32 = 4 bytes)
    param_size_mb = (total_params * 4) / (1024 * 1024)

    # 2. Disk checkpoint size
    disk_size_mb = 0.0
    if checkpoint_path is not None:
        cp_path = Path(checkpoint_path)
        if cp_path.exists():
            disk_size_mb = cp_path.stat().st_size / (1024 * 1024)

    # 3. Forward activation memory estimation
    # Track tensor hook activations
    activation_bytes = [0]

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            activation_bytes[0] += output.numel() * output.element_size()
        elif isinstance(output, (tuple, list)):
            for o in output:
                if isinstance(o, torch.Tensor):
                    activation_bytes[0] += o.numel() * o.element_size()

    hooks = []
    for layer in model.modules():
        if layer != model:
            hooks.append(layer.register_forward_hook(hook_fn))

    dummy_input = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        _ = model(dummy_input)

    for h in hooks:
        h.remove()

    estimated_activations_mb = activation_bytes[0] / (1024 * 1024)
    total_estimated_ram_mb = param_size_mb + estimated_activations_mb

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "param_size_fp32_mb": param_size_mb,
        "checkpoint_disk_size_mb": disk_size_mb,
        "estimated_activation_mb": estimated_activations_mb,
        "estimated_total_forward_ram_mb": total_estimated_ram_mb,
        "batch_size": input_shape[0],
    }
