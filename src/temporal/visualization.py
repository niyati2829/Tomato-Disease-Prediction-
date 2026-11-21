"""Visualization Utilities for Temporal Progression & Internal State Dynamics (REQ-09).

Generates 3 high-resolution figures in artifacts/figures/temporal/:
1. synthetic_disease_progression_patterns.png
2. rnn_vs_lstm_loss_and_predictions.png
3. rnn_vs_lstm_state_dynamics.png

Every figure explicitly documents that the dataset is a synthetic simulation.

Author: Crop Disease Detection Project Team
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.temporal.sequence_generator import PATTERN_NAMES


def plot_synthetic_progression_patterns(
    splits: Dict[str, Any],
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot sample synthetic trajectories across all 4 mathematical disease progression models."""
    X_train = splits["train"]["X"]
    y_train = splits["train"]["y"]
    patterns = splits["train"]["patterns"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    colors = {0: "#2ca02c", 1: "#1f77b4", 2: "#d62728", 3: "#9467bd"}

    t_inputs = np.arange(X_train.shape[1])
    t_full = np.arange(X_train.shape[1] + 1)

    # Panel 0: Individual example trajectories per pattern
    for p_id, p_name in PATTERN_NAMES.items():
        idxs = np.where(patterns == p_id)[0][:3]
        for k, idx in enumerate(idxs):
            full_s = np.append(X_train[idx, :, 0], y_train[idx])
            label = p_name if k == 0 else None
            axes[0].plot(t_full, full_s, marker="o", markersize=3.5, linewidth=1.8, color=colors[p_id], alpha=0.85, label=label)

    axes[0].axvline(X_train.shape[1] - 0.5, color="black", linestyle="--", alpha=0.6, label="Forecast Horizon (t=10)")
    axes[0].set_title("Simulated Plant Trajectories (Synthetic Benchmark)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Discrete Observation Step (t)", fontsize=10)
    axes[0].set_ylabel("Simulated Disease Severity Index s_t", fontsize=10)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=8.5, loc="upper left")

    # Panel 1: Mean +/- Std across entire training distribution
    for p_id, p_name in PATTERN_NAMES.items():
        idxs = np.where(patterns == p_id)[0]
        full_mat = np.column_stack([X_train[idxs, :, 0], y_train[idxs]])
        mean_curve = np.mean(full_mat, axis=0)
        std_curve = np.std(full_mat, axis=0)

        axes[1].plot(t_full, mean_curve, linewidth=2.4, color=colors[p_id], label=p_name)
        axes[1].fill_between(t_full, np.clip(mean_curve - std_curve, 0, 1), np.clip(mean_curve + std_curve, 0, 1), color=colors[p_id], alpha=0.18)

    axes[1].set_title("Cohort Mean & Spread Across Synthetic Splits (N=420 Train Plants)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Discrete Observation Step (t)", fontsize=10)
    axes[1].set_ylabel("Mean Severity ± 1 SD", fontsize=10)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=8.5, loc="upper left")

    fig.suptitle(
        "Synthetic Longitudinal Disease Progression Benchmark (Educational Mathematical Simulation)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")

    return fig


def plot_rnn_vs_lstm_results(
    history: Dict[str, Any],
    eval_results: Dict[str, Any],
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot validation MSE curves and test set actual vs predicted severity scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    rnn_hist = history["VanillaRNN"]
    lstm_hist = history["LSTM"]

    epochs = np.arange(1, len(rnn_hist["val_loss"]) + 1)

    ax1.plot(epochs, rnn_hist["train_loss"], linestyle="--", color="#ff7f0e", alpha=0.7, label="Vanilla RNN Train MSE")
    ax1.plot(epochs, rnn_hist["val_loss"], linewidth=2.2, color="#ff7f0e", label="Vanilla RNN Val MSE")
    ax1.plot(epochs, lstm_hist["train_loss"], linestyle="--", color="#1f77b4", alpha=0.7, label="LSTM Train MSE")
    ax1.plot(epochs, lstm_hist["val_loss"], linewidth=2.2, color="#1f77b4", label="LSTM Val MSE")

    ax1.set_title("Training & Validation MSE Convergence (RNN vs LSTM)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=10)
    ax1.set_yscale("log")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(fontsize=9)

    # Panel 2: Untouched Synthetic Test Set Predictions vs Actual
    rnn_preds = np.array(eval_results["rnn_full"]["predictions"])
    lstm_preds = np.array(eval_results["lstm_full"]["predictions"])
    targets = np.array(eval_results["rnn_full"]["targets"])

    rnn_rmse = eval_results["report"]["VanillaRNN"]["overall_metrics"]["RMSE"]
    lstm_rmse = eval_results["report"]["LSTM"]["overall_metrics"]["RMSE"]

    ax2.scatter(targets, rnn_preds, alpha=0.65, s=28, color="#ff7f0e", marker="o", label=f"Vanilla RNN (RMSE={rnn_rmse:.4f})")
    ax2.scatter(targets, lstm_preds, alpha=0.65, s=28, color="#1f77b4", marker="^", label=f"LSTM (RMSE={lstm_rmse:.4f})")
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Prediction (y = x)")

    ax2.set_title("Next-Step Severity Prediction (Untouched Synthetic Test Set, N=90)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Ground-Truth Simulated Severity s_{10}", fontsize=10)
    ax2.set_ylabel("Predicted Severity ŝ_{10}", fontsize=10)
    ax2.set_xlim(-0.02, 1.05)
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Empirical Comparison: Vanilla RNN vs LSTM on Synthetic Longitudinal Progression",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")

    return fig


def plot_state_and_gate_dynamics(
    rnn_model: torch.nn.Module,
    lstm_model: torch.nn.Module,
    X_sample: torch.Tensor,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Visualize internal hidden-state, cell-state, and LSTM gate magnitudes over time."""
    rnn_model.eval()
    lstm_model.eval()

    with torch.no_grad():
        _, rnn_states = rnn_model(X_sample, return_states=True)
        _, lstm_states = lstm_model(X_sample, return_states=True)

    # Norms averaged over batch -> shape (Seq_Len,)
    rnn_h_norm = torch.norm(rnn_states["hidden_states"], dim=-1).mean(dim=0).cpu().numpy()
    lstm_h_norm = torch.norm(lstm_states["hidden_states"], dim=-1).mean(dim=0).cpu().numpy()
    lstm_c_norm = torch.norm(lstm_states["cell_states"], dim=-1).mean(dim=0).cpu().numpy()

    f_mean = lstm_states["forget_gates"].mean(dim=(0, 2)).cpu().numpy()
    i_mean = lstm_states["input_gates"].mean(dim=(0, 2)).cpu().numpy()
    o_mean = lstm_states["output_gates"].mean(dim=(0, 2)).cpu().numpy()

    t_steps = np.arange(X_sample.shape[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: State Magnitudes
    ax1.plot(t_steps, rnn_h_norm, marker="o", linewidth=2.2, color="#ff7f0e", label="Vanilla RNN Hidden Norm ||h_t||_2")
    ax1.plot(t_steps, lstm_h_norm, marker="s", linewidth=2.2, color="#1f77b4", label="LSTM Hidden Norm ||h_t||_2")
    ax1.plot(t_steps, lstm_c_norm, marker="^", linewidth=2.2, linestyle="--", color="#2ca02c", label="LSTM Cell State Norm ||C_t||_2")

    ax1.set_title("Recurrent State Magnitude Evolution Over Time", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time Step (t)", fontsize=10)
    ax1.set_ylabel("Mean L2 Norm Across Cohort", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(fontsize=9)

    # Panel 2: LSTM Gate Dynamics
    ax2.plot(t_steps, f_mean, marker="o", linewidth=2.2, color="#d62728", label="Forget Gate f_t (Retention)")
    ax2.plot(t_steps, i_mean, marker="s", linewidth=2.2, color="#2ca02c", label="Input Gate i_t (Update)")
    ax2.plot(t_steps, o_mean, marker="^", linewidth=2.2, color="#9467bd", label="Output Gate o_t (Exposure)")

    ax2.set_title("Mean LSTM Gate Activation Trajectories", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time Step (t)", fontsize=10)
    ax2.set_ylabel("Mean Sigmoid Gate Activation [0, 1]", fontsize=10)
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Internal Activation Analysis: Hidden States, Cell Memory & Gate Dynamics (Synthetic Data)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")

    return fig

