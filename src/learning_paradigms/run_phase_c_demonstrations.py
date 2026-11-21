"""Phase C Demonstration Driver & Figure Generation Pipeline (REQ-06).

Executes all Phase C foundational experiments:
1. Hebbian Learning & Oja's Rule (Weight explosion vs unit-norm stabilization & PCA extraction)
2. Competitive Learning (WTA) & Kohonen Self-Organizing Map (SOM)
3. Error-Correction Learning (Delta Rule / Widrow-Hoff LMS)
4. Discrete Hopfield Network (Associative Memory & Lyapunov Energy Minimization)
5. Restricted Boltzmann Machine (RBM Energy-Based Probabilistic Learning)

Exports all publication figures to:
    artifacts/figures/learning_paradigms/

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

from src.learning_paradigms.boltzmann import (
    demonstrate_boltzmann_sampling,
    plot_boltzmann_figures,
)
from src.learning_paradigms.competitive import (
    demonstrate_competitive_and_som,
    plot_competitive_and_som_figures,
)
from src.learning_paradigms.error_correction import (
    demonstrate_error_correction,
    plot_error_correction_figures,
)
from src.learning_paradigms.hebbian import (
    compare_hebbian_and_oja,
    plot_hebbian_and_oja_figures,
)
from src.learning_paradigms.hopfield import (
    demonstrate_hopfield_associative_memory,
    plot_hopfield_figures,
)


def run_all_phase_c_demonstrations() -> None:
    print("=" * 80)
    print("PHASE C: CLASSICAL LEARNING PARADIGMS & ENERGY-BASED NETWORKS (REQ-06)")
    print("=" * 80)

    out_dir = Path("artifacts/figures/learning_paradigms/")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Hebbian vs Oja's Rule
    print("\n[1/5] Running Hebbian Learning vs Oja's Rule Demonstrations...")
    hebb_res = compare_hebbian_and_oja(n_iterations=1500, seed=42)
    fig_hebb_1, fig_hebb_2 = plot_hebbian_and_oja_figures(hebb_res, out_dir=out_dir)
    print(f"  - Basic Hebbian final ||w||: {hebb_res['final_hebb_norm']:.4e} (Unbounded Growth)")
    print(f"  - Oja's Rule final ||w||:    {hebb_res['final_oja_norm']:.4f} (Stabilized to Unit Norm)")
    print(f"  - Oja PCA Cosine Similarity: {hebb_res['final_cosine_similarity']:.4f}")
    print(f"  - Figures exported: {fig_hebb_1.name}, {fig_hebb_2.name}")

    # 2. Competitive Learning & SOM
    print("\n[2/5] Running Competitive Learning (WTA) & Kohonen SOM Demonstrations...")
    comp_res = demonstrate_competitive_and_som(seed=42)
    fig_comp_1, fig_comp_2 = plot_competitive_and_som_figures(comp_res, out_dir=out_dir)
    print(f"  - WTA Initial MSE: {comp_res['wta']['history_error'][0]:.4f} -> Final MSE: {comp_res['wta']['history_error'][-1]:.4f}")
    print(f"  - SOM 5x5 Grid organized across 4 data clusters")
    print(f"  - Figures exported: {fig_comp_1.name}, {fig_comp_2.name}")

    # 3. Error-Correction Learning (Delta Rule / LMS)
    print("\n[3/5] Running Error-Correction Learning (Delta Rule / Adaline)...")
    err_res = demonstrate_error_correction(seed=42)
    fig_err = plot_error_correction_figures(err_res, out_dir=out_dir)
    print(f"  - Delta Rule Initial MSE: {err_res['history']['loss_history'][0]:.4f} -> Final MSE: {err_res['history']['loss_history'][-1]:.4f}")
    print(f"  - Final Misclassified:     {err_res['history']['error_count_history'][-1]} / 200 samples")
    print(f"  - Figure exported: {fig_err.name}")

    # 4. Hopfield Network
    print("\n[4/5] Running Hopfield Associative Memory & Energy Minimization...")
    hop_res = demonstrate_hopfield_associative_memory(seed=42)
    fig_hop_1, fig_hop_2 = plot_hopfield_figures(hop_res, out_dir=out_dir)
    print(f"  - Stored 3 canonical 8x8 binary patterns (64 neurons)")
    print(f"  - Pattern Retrieval Accuracy: {hop_res['retrieval_accuracy_pct']:.1f}% from 28% corrupted input")
    print(f"  - Energy Trajectory: Initial={hop_res['energy_history'][0]:.2f} -> Attractor={hop_res['energy_history'][-1]:.2f}")
    print(f"  - Delta E <= 0 Monotonicity: {bool(np.all(np.diff(hop_res['energy_history']) <= 1e-9))}")
    print(f"  - Figures exported: {fig_hop_1.name}, {fig_hop_2.name}")

    # 5. Restricted Boltzmann Machine
    print("\n[5/5] Running Restricted Boltzmann Machine (RBM) Probabilistic Learning...")
    boltz_res = demonstrate_boltzmann_sampling(seed=42)
    fig_boltz = plot_boltzmann_figures(boltz_res, out_dir=out_dir)
    print(f"  - Initial Reconstruction MSE: {boltz_res['training_history']['reconstruction_error'][0]:.4f}")
    print(f"  - Final Reconstruction MSE:   {boltz_res['training_history']['reconstruction_error'][-1]:.4f}")
    print(f"  - Final Average Free Energy:  {boltz_res['training_history']['free_energy'][-1]:.4f}")
    print(f"  - Figure exported: {fig_boltz.name}")

    print("\n" + "=" * 80)
    print("ALL PHASE C DEMONSTRATIONS SUCCESSFULLY COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_all_phase_c_demonstrations()

