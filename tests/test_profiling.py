"""Unit tests for inference latency, throughput, and memory profiling."""

import pytest
import torch
import torch.nn as nn

from src.profiling.latency_profiler import profile_latency, synchronize_device
from src.profiling.memory_profiler import profile_model_memory


class TinyToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 5)

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).flatten(1))


def test_synchronize_device_cpu():
    """Device synchronization should execute cleanly on CPU."""
    device = torch.device("cpu")
    synchronize_device(device)  # Should not raise exception


def test_profile_latency_structure():
    """Verify latency profiler outputs all expected latency and throughput metrics."""
    model = TinyToyModel()
    metrics = profile_latency(
        model=model,
        input_shape=(2, 3, 32, 32),
        device=torch.device("cpu"),
        warmup_runs=2,
        num_runs=5,
    )

    required_keys = {
        "batch_size",
        "mean_batch_latency_ms",
        "median_batch_latency_ms",
        "p95_batch_latency_ms",
        "mean_per_sample_ms",
        "throughput_fps",
    }
    assert required_keys.issubset(set(metrics.keys()))
    assert metrics["batch_size"] == 2
    assert metrics["mean_batch_latency_ms"] > 0.0
    assert metrics["throughput_fps"] > 0.0
    assert metrics["p95_batch_latency_ms"] >= metrics["median_batch_latency_ms"]


def test_profile_model_memory():
    """Verify parameter count and memory estimation calculation."""
    model = TinyToyModel()
    mem_stats = profile_model_memory(
        model=model,
        input_shape=(1, 3, 32, 32),
        device=torch.device("cpu"),
    )

    assert mem_stats["total_parameters"] > 0
    assert mem_stats["trainable_parameters"] == mem_stats["total_parameters"]
    assert mem_stats["param_size_fp32_mb"] > 0.0
    assert mem_stats["estimated_activation_mb"] > 0.0
    assert mem_stats["estimated_total_forward_ram_mb"] >= mem_stats["param_size_fp32_mb"]
