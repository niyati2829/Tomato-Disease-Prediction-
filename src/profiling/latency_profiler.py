"""High-precision latency and throughput profiler for PyTorch models."""

import time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn


def synchronize_device(device: torch.device):
    """Ensure asynchronous device queues have finished executing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass


def profile_latency(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: torch.device = torch.device("cpu"),
    warmup_runs: int = 10,
    num_runs: int = 50,
) -> Dict[str, float]:
    """Measure inference latency and throughput for a model with specified input shape.

    Args:
        model: PyTorch model in eval mode.
        input_shape: Tuple of (batch_size, channels, height, width).
        device: Execution device.
        warmup_runs: Number of unmeasured initial runs.
        num_runs: Number of timed repetitions.

    Returns:
        Dict containing mean, median (P50), P95, std, and throughput metrics.
    """
    model.eval()
    model.to(device)

    batch_size = input_shape[0]
    dummy_input = torch.randn(*input_shape, device=device)

    # 1. Warm-up
    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            synchronize_device(device)

    # 2. Benchmark runs
    batch_latencies_ms = []
    with torch.inference_mode():
        for _ in range(num_runs):
            synchronize_device(device)
            t0 = time.perf_counter()
            _ = model(dummy_input)
            synchronize_device(device)
            t1 = time.perf_counter()
            batch_latencies_ms.append((t1 - t0) * 1000.0)

    latencies = np.array(batch_latencies_ms)
    mean_batch_ms = float(np.mean(latencies))
    median_batch_ms = float(np.median(latencies))
    p95_batch_ms = float(np.percentile(latencies, 95))
    std_batch_ms = float(np.std(latencies))

    mean_per_sample_ms = mean_batch_ms / batch_size
    median_per_sample_ms = median_batch_ms / batch_size
    p95_per_sample_ms = p95_batch_ms / batch_size

    throughput_fps = (batch_size / (mean_batch_ms / 1000.0)) if mean_batch_ms > 0 else 0.0

    return {
        "batch_size": batch_size,
        "mean_batch_latency_ms": mean_batch_ms,
        "median_batch_latency_ms": median_batch_ms,
        "p95_batch_latency_ms": p95_batch_ms,
        "std_batch_latency_ms": std_batch_ms,
        "mean_per_sample_ms": mean_per_sample_ms,
        "median_per_sample_ms": median_per_sample_ms,
        "p95_per_sample_ms": p95_per_sample_ms,
        "throughput_fps": throughput_fps,
    }


def benchmark_latencies(
    models: Dict[str, nn.Module],
    batch_sizes: List[int] = [1, 4, 8, 16],
    device: torch.device = torch.device("cpu"),
    warmup_runs: int = 10,
    num_runs: int = 40,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Benchmark multiple models across multiple batch sizes."""
    results = {}
    for model_name, model in models.items():
        results[model_name] = {}
        for b in batch_sizes:
            metrics = profile_latency(
                model=model,
                input_shape=(b, 3, 224, 224),
                device=device,
                warmup_runs=warmup_runs,
                num_runs=num_runs,
            )
            results[model_name][b] = metrics
    return results
