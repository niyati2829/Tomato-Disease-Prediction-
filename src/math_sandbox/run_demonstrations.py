"""Phase A Mathematical & Neural Fundamentals Demonstration Pipeline.

Executes all Phase A foundational demonstrations:
1. Tensor operations & broadcasting rules.
2. SVD image compression across ranks k=[5, 20, 50, 100].
3. Finite difference numerical gradient checking vs analytical derivatives.
4. McCulloch-Pitts logic gates (AND, OR, NOT, NAND, NOR).
5. Perceptron training on linearly separable gates and failure on XOR.
6. Activation functions & derivatives landscape.
7. Multilayer Perceptron from scratch:
   - Synthetic classification dataset (make_moons / make_blobs)
   - Vectorized forward pass
   - Manual backpropagation
   - Gradient checking verification on all weight and bias tensors
   - Mini-batch training loss/accuracy curve export

Exports all figures directly to artifacts/figures/foundations/.

Author: Crop Disease Detection Project Team
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.math_sandbox.gradient_checker import (
    check_gradient_scalar,
    plot_gradient_check_verification,
)
from src.math_sandbox.svd_compression import plot_svd_compression_comparison
from src.math_sandbox.tensor_ops import (
    demonstrate_broadcasting_rules,
    demonstrate_tensor_operations,
    plot_broadcasting_examples,
)
from src.neural_fundamentals.activations import (
    get_activation_specifications,
    plot_activation_comparison,
)
from src.neural_fundamentals.mcculloch_pitts import demonstrate_mp_gates
from src.neural_fundamentals.mlp_from_scratch import (
    MultilayerPerceptron,
    plot_mlp_training_results,
)
from src.neural_fundamentals.perceptron import plot_perceptron_decision_boundaries

FIGURES_FOUNDATIONS = PROJECT_ROOT / "artifacts" / "figures" / "foundations"
FIGURES_FOUNDATIONS.mkdir(parents=True, exist_ok=True)


def run_all_phase_a_demonstrations():
    print("=" * 80)
    print("PHASE A: MATHEMATICAL FOUNDATIONS & NEURAL NETWORK FUNDAMENTALS")
    print("=" * 80)

    results_summary = {}

    # -----------------------------------------------------------------------
    # 1. Tensor Operations & Broadcasting (REQ-01)
    # -----------------------------------------------------------------------
    print("\n[1/6] Running Tensor Operations & Broadcasting Demonstrations...")
    tensor_ops_res = demonstrate_tensor_operations()
    bcast_res = demonstrate_broadcasting_rules()
    bcast_fig = plot_broadcasting_examples(FIGURES_FOUNDATIONS / "broadcasting_visualization.png")
    print(f"  - Dot Product ([1,2,3] . [4,5,6]): {tensor_ops_res['dot_product']}")
    print(f"  - Frobenius Norm of 2x2 matrix: {tensor_ops_res['frobenius_norm']:.4f}")
    print(f"  - Broadcasting Rules Tested: {len(bcast_res)} cases")
    print(f"  - Figure saved: {bcast_fig.name}")
    results_summary["tensor_operations"] = tensor_ops_res
    results_summary["broadcasting"] = bcast_res

    # -----------------------------------------------------------------------
    # 2. SVD Image Compression (REQ-01)
    # -----------------------------------------------------------------------
    print("\n[2/6] Running Singular Value Decomposition (SVD) Image Compression...")
    sample_leaf = PROJECT_ROOT / "data/raw/PlantVillage-Dataset/raw/color/Tomato___Bacterial_spot/0045ba29-ed1b-43b4-afde-719cc7adefdb___GCREC_Bact.Sp 6254.JPG"
    svd_fig = plot_svd_compression_comparison(
        image_path=sample_leaf,
        output_path=FIGURES_FOUNDATIONS / "svd_compression_comparison.png",
        k_ranks=[5, 20, 50, 100],
    )
    print(f"  - SVD compression evaluated across ranks k = [5, 20, 50, 100]")
    print(f"  - Figure saved: {svd_fig.name}")

    # -----------------------------------------------------------------------
    # 3. Finite Difference Numerical Gradient Checking (REQ-01)
    # -----------------------------------------------------------------------
    print("\n[3/6] Running Numerical Gradient Checking (Forward vs Central Difference)...")
    f_test = lambda x: np.sin(x) * np.exp(x / 2.0)
    fp_test = lambda x: np.cos(x) * np.exp(x / 2.0) + 0.5 * np.sin(x) * np.exp(x / 2.0)
    scalar_check = check_gradient_scalar(f_test, fp_test, x=1.2, eps_list=[1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
    grad_fig = plot_gradient_check_verification(FIGURES_FOUNDATIONS / "gradient_check_verification.png")
    best_central_err = min(s["central_abs_error"] for s in scalar_check)
    print(f"  - Analytical vs Central Difference best error: {best_central_err:.2e}")
    print(f"  - Figure saved: {grad_fig.name}")
    results_summary["numerical_gradient_checking"] = scalar_check

    # -----------------------------------------------------------------------
    # 4. McCulloch-Pitts & Perceptron Fundamentals (REQ-02)
    # -----------------------------------------------------------------------
    print("\n[4/6] Running McCulloch-Pitts & Perceptron Demonstrations...")
    mp_res = demonstrate_mp_gates()
    print("  - McCulloch-Pitts Logic Gates:")
    for gate, ginfo in mp_res.items():
        print(f"    * Gate {gate:<4}: Passed={ginfo['passed']}, Threshold={ginfo['threshold']}")

    percep_fig = plot_perceptron_decision_boundaries(FIGURES_FOUNDATIONS / "perceptron_decision_boundaries.png")
    print(f"  - Perceptron decision boundary figure saved: {percep_fig.name}")
    results_summary["mcculloch_pitts"] = mp_res

    # -----------------------------------------------------------------------
    # 5. Activation Functions Landscape (REQ-02)
    # -----------------------------------------------------------------------
    print("\n[5/6] Generating Activation Functions & Derivatives Landscape...")
    act_specs = get_activation_specifications()
    act_fig = plot_activation_comparison(FIGURES_FOUNDATIONS / "activation_functions_comparison.png")
    print(f"  - Evaluated: Step, Sigmoid, Tanh, ReLU, LeakyReLU")
    print(f"  - Figure saved: {act_fig.name}")
    results_summary["activation_specs"] = act_specs

    # -----------------------------------------------------------------------
    # 6. Multilayer Perceptron from Scratch with Manual Backprop (REQ-03)
    # -----------------------------------------------------------------------
    print("\n[6/6] Training Multilayer Perceptron from Scratch & Verifying Backprop...")
    np.random.seed(42)

    # Generate synthetic 3-class 2D spiral/moon-like classification dataset
    N_per_class = 150
    D = 2
    K = 3
    X = np.zeros((N_per_class * K, D))
    y = np.zeros(N_per_class * K, dtype=int)

    for j in range(K):
        ix = range(N_per_class * j, N_per_class * (j + 1))
        # Start radius slightly away from origin to avoid ReLU non-differentiability kink at exactly zero
        r = np.linspace(0.05, 1.0, N_per_class)
        t = np.linspace(j * 4, (j + 1) * 4, N_per_class) + np.random.randn(N_per_class) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    # One-hot encode targets
    Y_onehot = np.zeros((N_per_class * K, K))
    Y_onehot[np.arange(N_per_class * K), y] = 1.0

    # Stratified train/val split (80/20)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_val = X[indices[:split]], X[indices[split:]]
    y_train, y_val = Y_onehot[indices[:split]], Y_onehot[indices[split:]]

    # Instantiate MLP: 2 inputs -> 16 hidden (ReLU) -> 12 hidden (ReLU) -> 3 outputs (Softmax)
    mlp = MultilayerPerceptron(layer_sizes=[2, 16, 12, 3], hidden_activation="relu", seed=42)

    # Comprehensive numerical gradient checking on mini-batch BEFORE training
    print("  - Running mathematical gradient check on all parameter tensors...")
    grad_check_results = mlp.verify_gradients_numerically(X_train[:16], y_train[:16], eps=1e-7)

    print(f"    * Overall Gradient Check Passed: {grad_check_results['all_passed']}")
    print(f"    * Max Relative Gradient Error: {grad_check_results['max_relative_error']:.2e} (tolerance < 1e-5)")
    for p_name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        info = grad_check_results[p_name]
        print(f"      - {p_name}: RelError = {info['relative_error']:.2e} [Passed: {info['passed']}]")

    assert grad_check_results["all_passed"], f"FATAL: Gradient check failed! Max error: {grad_check_results['max_relative_error']}"

    # Train MLP
    print("  - Training MLP from scratch for 120 epochs using mini-batch gradient descent...")
    history = mlp.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=120,
        batch_size=32,
        lr=0.1,
    )

    mlp_fig = plot_mlp_training_results(history, FIGURES_FOUNDATIONS / "mlp_training_loss_curves.png")
    final_train_acc = history["train_acc"][-1]
    final_val_acc = history["val_acc"][-1]
    final_train_loss = history["train_loss"][-1]
    final_val_loss = history["val_loss"][-1]
    print(f"    * Final Train Loss: {final_train_loss:.4f} | Train Acc: {final_train_acc:.2f}%")
    print(f"    * Final Val Loss:   {final_val_loss:.4f} | Val Acc:   {final_val_acc:.2f}%")
    print(f"  - Figure saved: {mlp_fig.name}")

    results_summary["mlp_from_scratch"] = {
        "architecture": mlp.layer_sizes,
        "hidden_activation": mlp.hidden_activation,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "gradient_check": grad_check_results,
    }

    print("\n" + "=" * 80)
    print("ALL PHASE A DEMONSTRATIONS SUCCESSFULLY COMPLETED")
    print("=" * 80)
    return results_summary


if __name__ == "__main__":
    run_all_phase_a_demonstrations()
