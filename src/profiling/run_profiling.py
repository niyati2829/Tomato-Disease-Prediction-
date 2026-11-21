"""Run latency and memory profiling across models and batch sizes."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.models.cnn import BaselineCNN
from src.models.transfer_learning import TomatoResNet18
from src.utils.device import get_device
from src.profiling.latency_profiler import benchmark_latencies
from src.profiling.memory_profiler import profile_model_memory


def run_profiling_suite(
    output_dir: Path = PROJECT_ROOT / "artifacts/figures/profiling",
    latency_csv: Path = PROJECT_ROOT / "reports/final_latency_benchmark.csv",
    memory_csv: Path = PROJECT_ROOT / "reports/final_memory_profile.csv",
    batch_sizes: list = [1, 4, 8, 16],
):
    print("=" * 70)
    print("STARTING LATENCY & MEMORY PROFILING SUITE (Phase F — REQ-10)")
    print("=" * 70)

    device = get_device(preferred="mps")
    print(f"Profiling compute device: {device}")

    # 1. Instantiate models
    print("\nInstantiating models...")
    baseline_cnn = BaselineCNN(num_classes=10)
    baseline_cp = PROJECT_ROOT / "artifacts/checkpoints/best_baseline_cnn.pt"
    if baseline_cp.exists():
        cp = torch.load(baseline_cp, map_location=device)
        baseline_cnn.load_state_dict(cp.get("model_state_dict", cp))

    resnet_unweighted = TomatoResNet18(num_classes=10, weights=None)
    unweighted_cp = PROJECT_ROOT / "artifacts/checkpoints/best_resnet18.pt"
    if unweighted_cp.exists():
        cp = torch.load(unweighted_cp, map_location=device)
        resnet_unweighted.load_state_dict(cp.get("model_state_dict", cp))

    resnet_prod = TomatoResNet18(num_classes=10, weights=None)
    prod_cp = PROJECT_ROOT / "artifacts/checkpoints/best_resnet18_weighted.pt"
    if prod_cp.exists():
        cp = torch.load(prod_cp, map_location=device)
        resnet_prod.load_state_dict(cp.get("model_state_dict", cp))

    models_dict = {
        "Baseline CNN": baseline_cnn,
        "ResNet18 (Unweighted)": resnet_unweighted,
        "ResNet18 (Production Weighted)": resnet_prod,
    }

    # 2. Latency Benchmarking
    print("\nExecuting latency & throughput profiling across batch sizes...")
    latency_results = benchmark_latencies(
        models=models_dict,
        batch_sizes=batch_sizes,
        device=device,
        warmup_runs=10,
        num_runs=40,
    )

    latency_rows = []
    for model_name, b_dict in latency_results.items():
        for b, metrics in b_dict.items():
            latency_rows.append({
                "Model": model_name,
                "Batch_Size": b,
                "Mean_Batch_Latency_ms": metrics["mean_batch_latency_ms"],
                "Median_Batch_Latency_ms": metrics["median_batch_latency_ms"],
                "P95_Batch_Latency_ms": metrics["p95_batch_latency_ms"],
                "Mean_Per_Sample_ms": metrics["mean_per_sample_ms"],
                "Median_Per_Sample_ms": metrics["median_per_sample_ms"],
                "P95_Per_Sample_ms": metrics["p95_per_sample_ms"],
                "Throughput_FPS": metrics["throughput_fps"],
            })

    df_latency = pd.DataFrame(latency_rows)
    latency_csv.parent.mkdir(parents=True, exist_ok=True)
    df_latency.to_csv(latency_csv, index=False)
    print(f"Saved latency benchmark to: {latency_csv}")

    # 3. Memory & Parameter Profiling
    print("\nProfiling memory footprints and parameter counts...")
    checkpoints = {
        "Baseline CNN": baseline_cp,
        "ResNet18 (Unweighted)": unweighted_cp,
        "ResNet18 (Production Weighted)": prod_cp,
    }

    memory_rows = []
    for model_name, model in models_dict.items():
        mem_stats = profile_model_memory(
            model=model,
            input_shape=(1, 3, 224, 224),
            checkpoint_path=checkpoints[model_name],
            device=device,
        )
        memory_rows.append({
            "Model": model_name,
            "Total_Parameters": mem_stats["total_parameters"],
            "Trainable_Parameters": mem_stats["trainable_parameters"],
            "Parameter_Size_FP32_MB": mem_stats["param_size_fp32_mb"],
            "Checkpoint_Disk_Size_MB": mem_stats["checkpoint_disk_size_mb"],
            "Activation_Memory_B1_MB": mem_stats["estimated_activation_mb"],
            "Total_Forward_RAM_B1_MB": mem_stats["estimated_total_forward_ram_mb"],
        })

    df_memory = pd.DataFrame(memory_rows)
    memory_csv.parent.mkdir(parents=True, exist_ok=True)
    df_memory.to_csv(memory_csv, index=False)
    print(f"Saved memory profile to: {memory_csv}")

    # 4. Visualizations
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Per-Sample Latency vs Batch Size
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"Baseline CNN": "#3498db", "ResNet18 (Unweighted)": "#95a5a6", "ResNet18 (Production Weighted)": "#2ecc71"}
    markers = {"Baseline CNN": "o", "ResNet18 (Unweighted)": "s", "ResNet18 (Production Weighted)": "^"}

    for model_name in models_dict.keys():
        sub = df_latency[df_latency["Model"] == model_name]
        ax.plot(
            sub["Batch_Size"],
            sub["Mean_Per_Sample_ms"],
            marker=markers[model_name],
            color=colors[model_name],
            linewidth=2.2,
            markersize=7,
            label=model_name,
        )

    ax.set_xticks(batch_sizes)
    ax.set_xlabel("Inference Batch Size", fontsize=11, weight="bold")
    ax.set_ylabel("Latency per Sample (ms)", fontsize=11, weight="bold")
    ax.set_title("Inference Latency Amortization vs Batch Size", fontsize=13, weight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(framealpha=0.9)
    lat_fig_path = output_dir / "latency_vs_batch_size.png"
    plt.tight_layout()
    plt.savefig(lat_fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved latency curve to: {lat_fig_path}")

    # Plot 2: Throughput (FPS) Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(batch_sizes))
    width = 0.25

    for idx, model_name in enumerate(models_dict.keys()):
        sub = df_latency[df_latency["Model"] == model_name]
        ax.bar(
            x + (idx - 1) * width,
            sub["Throughput_FPS"],
            width=width,
            color=colors[model_name],
            label=model_name,
            edgecolor="black",
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Batch {b}" for b in batch_sizes], fontsize=11)
    ax.set_xlabel("Batch Size", fontsize=11, weight="bold")
    ax.set_ylabel("Throughput (Frames Per Second)", fontsize=11, weight="bold")
    ax.set_title("Inference Throughput Across Batch Sizes", fontsize=13, weight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(framealpha=0.9)
    tp_fig_path = output_dir / "throughput_comparison.png"
    plt.tight_layout()
    plt.savefig(tp_fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved throughput comparison to: {tp_fig_path}")

    # Plot 3: Memory & Parameter Footprint
    fig, ax1 = plt.subplots(figsize=(8, 5))
    model_labels = list(models_dict.keys())
    params_m = df_memory["Total_Parameters"] / 1e6
    disk_mb = df_memory["Checkpoint_Disk_Size_MB"]

    x = np.arange(len(model_labels))
    ax1.bar(x - 0.18, params_m, width=0.35, color="#34495e", label="Parameters (Millions)", edgecolor="black")
    ax1.set_ylabel("Parameters (Millions)", fontsize=11, weight="bold", color="#34495e")

    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, disk_mb, width=0.35, color="#e67e22", label="Checkpoint Disk Size (MB)", edgecolor="black")
    ax2.set_ylabel("Disk Footprint (MB)", fontsize=11, weight="bold", color="#e67e22")

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=10, weight="bold")
    ax1.set_title("Model Complexity & Storage Footprint", fontsize=13, weight="bold", pad=12)
    mem_fig_path = output_dir / "memory_footprint_comparison.png"
    plt.tight_layout()
    plt.savefig(mem_fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved memory footprint comparison to: {mem_fig_path}")

    print("\nProfiling suite completed successfully!")
    return df_latency, df_memory


if __name__ == "__main__":
    run_profiling_suite()
