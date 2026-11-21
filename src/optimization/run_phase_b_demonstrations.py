"""Phase B Demonstration Driver & Figure Generation Pipeline.

Executes all Phase B foundational experiments:
1. Gradient descent variants (Batch GD, SGD, Mini-Batch GD)
2. Learning rate stability & divergence experiments
3. 2D and 3D loss landscape visualizations with optimizer trajectories
4. Weight initialization schemes (Zero, Random, Xavier, He) and layer-by-layer variance flow
5. BatchNorm vs LayerNorm distributions and batch-size sensitivity
6. Regularization benchmarks: L1, L2, and Inverted Dropout
7. Controlled ablation studies across regularizers, augmentations, and initializations

Exports all publication figures to:
- artifacts/figures/optimization/
- artifacts/figures/regularization/

Author: Crop Disease Detection Project Team
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.gradient_descent import (
    compare_gd_variants,
    generate_synthetic_regression_data,
)
from src.optimization.learning_rate_experiments import (
    plot_learning_rate_comparison,
    run_learning_rate_experiments,
)
from src.optimization.loss_landscape import (
    compute_optimizer_trajectories,
    plot_2d_contour_trajectories,
    plot_3d_surface_trajectories,
    plot_optimizer_convergence_comparison,
)
from src.regularization.ablation_studies import run_ablation_benchmark
from src.regularization.initialization import (
    compare_initialization_schemes,
    plot_initialization_diagnostics,
)
from src.regularization.normalization import (
    compare_batch_and_layer_norm,
    plot_normalization_comparison,
)
from src.regularization.regularizers import (
    compare_regularization_schemes,
    plot_regularization_figures,
)


def run_all_phase_b_demonstrations() -> None:
    print("=" * 80)
    print("PHASE B: OPTIMIZATION DYNAMICS, WEIGHT INITIALIZATION & REGULARIZATION")
    print("=" * 80)

    # 1. Gradient Descent Variants
    print("\n[1/7] Running Gradient Descent Variants (Batch GD, SGD, Mini-Batch GD)...")
    X, y, _ = generate_synthetic_regression_data(n_samples=250, n_features=5, seed=42)
    gd_res = compare_gd_variants(X, y, lr=0.05, epochs=50, batch_size=16, seed=42)
    print(f"  - Batch GD final loss:     {gd_res['batch_gd']['final_loss']:.5f} (Updates: {gd_res['batch_gd']['total_updates']})")
    print(f"  - Mini-Batch GD final loss: {gd_res['mini_batch_gd']['final_loss']:.5f} (Updates: {gd_res['mini_batch_gd']['total_updates']})")
    print(f"  - SGD final loss:           {gd_res['sgd']['final_loss']:.5f} (Updates: {gd_res['sgd']['total_updates']})")

    # 2. Learning Rate Dynamics
    print("\n[2/7] Running Learning Rate Stability & Divergence Experiments...")
    lr_data = run_learning_rate_experiments(learning_rates=[0.001, 0.05, 0.25, 0.42], n_steps=35, seed=42)
    lr_fig = plot_learning_rate_comparison(lr_data, "artifacts/figures/optimization/learning_rate_comparison.png")
    print(f"  - Figure exported: {lr_fig.name}")

    # 3. Loss Landscape Visualizations
    print("\n[3/7] Generating 2D & 3D Loss Landscape Trajectories (Anisotropic Valley)...")
    trajectories = compute_optimizer_trajectories(w_start=(-4.0, 2.5), n_steps=45, lr=0.15, condition_number=10.0)
    fig_2d = plot_2d_contour_trajectories(trajectories, "artifacts/figures/optimization/loss_landscape_2d_contours.png")
    fig_3d = plot_3d_surface_trajectories(trajectories, "artifacts/figures/optimization/loss_landscape_3d_surface.png")
    fig_conv = plot_optimizer_convergence_comparison(trajectories, "artifacts/figures/optimization/optimizer_convergence_comparison.png")
    print(f"  - Figures exported: {fig_2d.name}, {fig_3d.name}, {fig_conv.name}")

    # 4. Weight Initialization Analysis
    print("\n[4/7] Simulating Deep Network Signal Propagation across Initializations...")
    init_sim = compare_initialization_schemes(seed=42)
    fig_w_hist, fig_act_flow = plot_initialization_diagnostics(init_sim, "artifacts/figures/regularization/")
    print(f"  - Figures exported: {fig_w_hist.name}, {fig_act_flow.name}")

    # 5. Normalization Techniques (BatchNorm vs LayerNorm)
    print("\n[5/7] Benchmarking BatchNorm vs LayerNorm Distributions across Batch Sizes...")
    norm_res = compare_batch_and_layer_norm(batch_sizes=[32, 4, 2], num_features=50, seed=42)
    norm_fig = plot_normalization_comparison(norm_res, "artifacts/figures/regularization/batchnorm_vs_layernorm_distributions.png")
    print(f"  - Figure exported: {norm_fig.name}")

    # 6. Regularization Comparisons (L1, L2, Dropout)
    print("\n[6/7] Training Models with L1, L2, and Dropout on Over-parameterized Data...")
    reg_res = compare_regularization_schemes(seed=42)
    fig_reg_curves, fig_reg_weights, fig_dropout = plot_regularization_figures(reg_res, "artifacts/figures/regularization/")
    print(f"  - Figures exported: {fig_reg_curves.name}, {fig_reg_weights.name}, {fig_dropout.name}")

    # 7. Ablation Benchmark
    print("\n[7/7] Executing Controlled Single-Factor Ablation Studies...")
    ablation_res = run_ablation_benchmark(seed=42)
    print(f"  - Ablation results saved to: {ablation_res['json_path']}")
    for r in ablation_res["records"]:
        print(f"    * {r['experiment']}: Train Acc={r['train_accuracy_pct']}%, Val Acc={r['val_accuracy_pct']}%, Gen Gap={r['generalization_gap_pct']}%")

    print("\n" + "=" * 80)
    print("ALL PHASE B DEMONSTRATIONS SUCCESSFULLY COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_all_phase_b_demonstrations()

