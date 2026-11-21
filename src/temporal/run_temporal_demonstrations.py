"""Automated Execution Driver for Temporal Progression Demonstrations (REQ-09).

Executes:
1. Synthetic progression benchmark generation (600 simulated plants, split by plant_id).
2. Training of Vanilla RNN and LSTM regressors.
3. Evaluation on untouched synthetic test set.
4. Export of 3 visualizations in artifacts/figures/temporal/.

Author: Crop Disease Detection Project Team
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.temporal.evaluate import run_temporal_evaluation
from src.temporal.sequence_generator import generate_synthetic_progression_sequences
from src.temporal.temporal_model import LSTMRegressor, VanillaRNNRegressor
from src.temporal.train import run_temporal_training
from src.temporal.visualization import (
    plot_rnn_vs_lstm_results,
    plot_state_and_gate_dynamics,
    plot_synthetic_progression_patterns,
)
from src.utils.device import get_device


def run_all_temporal_demonstrations(
    out_dir: Path | str = "artifacts/figures/temporal",
    seed: int = 42,
) -> None:
    """Run all Phase E temporal training, evaluation, and plotting steps."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic data & visualize patterns
    splits = generate_synthetic_progression_sequences(
        n_plants=600, seq_len=10, seed=seed, save_dir="data/processed/synthetic_temporal"
    )
    fig1 = plot_synthetic_progression_patterns(
        splits, save_path=out_p / "synthetic_disease_progression_patterns.png"
    )
    plt.close(fig1)

    # 2. Train Vanilla RNN & LSTM
    history = run_temporal_training(epochs=40, batch_size=32, hidden_size=32, lr=2e-3, seed=seed)

    # 3. Evaluate on untouched synthetic test split
    eval_res = run_temporal_evaluation(seed=seed)

    # 4. Plot training curves & test predictions
    fig2 = plot_rnn_vs_lstm_results(
        history, eval_res, save_path=out_p / "rnn_vs_lstm_loss_and_predictions.png"
    )
    plt.close(fig2)

    # 5. Extract & plot internal state dynamics
    device = get_device(preferred="mps")
    rnn_ckpt = torch.load("artifacts/checkpoints/temporal/best_rnn.pt", map_location=device)
    lstm_ckpt = torch.load("artifacts/checkpoints/temporal/best_lstm.pt", map_location=device)

    rnn = VanillaRNNRegressor(input_size=3, hidden_size=32).to(device)
    rnn.load_state_dict(rnn_ckpt["model_state_dict"])

    lstm = LSTMRegressor(input_size=3, hidden_size=32).to(device)
    lstm.load_state_dict(lstm_ckpt["model_state_dict"])

    X_test_tensor = torch.tensor(splits["test"]["X"], dtype=torch.float32, device=device)
    fig3 = plot_state_and_gate_dynamics(
        rnn, lstm, X_test_tensor, save_path=out_p / "rnn_vs_lstm_state_dynamics.png"
    )
    plt.close(fig3)

    print(f"\n[Phase E] All Temporal Demonstrations complete! Figures saved in: {out_p}")


if __name__ == "__main__":
    run_all_temporal_demonstrations()

