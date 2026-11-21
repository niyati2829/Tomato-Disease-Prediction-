"""Training Engine Comparing Vanilla RNN vs LSTM on Synthetic Sequences (REQ-09).

Trains both models on identical sequence splits, saving best checkpoints by
validation MSE to artifacts/checkpoints/temporal/.

Author: Crop Disease Detection Project Team
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.temporal.sequence_dataset import DiseaseSequenceDataset
from src.temporal.sequence_generator import generate_synthetic_progression_sequences
from src.temporal.temporal_model import LSTMRegressor, VanillaRNNRegressor
from src.utils.device import get_device


def train_single_temporal_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 40,
    lr: float = 2e-3,
    ckpt_path: Path = Path("artifacts/checkpoints/temporal/best_model.pt"),
) -> Dict[str, Any]:
    """Train a sequence regressor and record history & best validation checkpoint."""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        batch_t_losses = []
        for X_b, y_b, _, _ in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            batch_t_losses.append(loss.item())

        avg_train_loss = float(np.mean(batch_t_losses))

        # Validation
        model.eval()
        batch_v_losses = []
        with torch.no_grad():
            for X_b, y_b, _, _ in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b)
                loss = criterion(preds, y_b)
                batch_v_losses.append(loss.item())

        avg_val_loss = float(np.mean(batch_v_losses))
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_name": model_name,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mse": best_val_loss,
                "hidden_size": model.hidden_size,
                "input_size": model.input_size,
            }, ckpt_path)

    total_time = time.perf_counter() - start_time
    param_count = sum(p.numel() for p in model.parameters())

    print(
        f"[{model_name}] Training Complete in {total_time:.2f}s | Params: {param_count:,} | "
        f"Best Val MSE: {best_val_loss:.6f}"
    )

    return {
        "model_name": model_name,
        "parameter_count": param_count,
        "training_time_sec": total_time,
        "best_val_mse": best_val_loss,
        "train_loss": train_losses,
        "val_loss": val_losses,
    }


def run_temporal_training(
    epochs: int = 40,
    batch_size: int = 32,
    hidden_size: int = 32,
    lr: float = 2e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train both Vanilla RNN and LSTM on identical synthetic disease progression splits."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device(preferred="mps")
    print(f"[Phase E] Training Temporal Models on device: {device}")

    splits = generate_synthetic_progression_sequences(n_plants=600, seq_len=10, seed=seed)
    train_ds = DiseaseSequenceDataset(splits["train"])
    val_ds = DiseaseSequenceDataset(splits["val"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    rnn = VanillaRNNRegressor(input_size=3, hidden_size=hidden_size)
    lstm = LSTMRegressor(input_size=3, hidden_size=hidden_size)

    rnn_hist = train_single_temporal_model(
        rnn, "VanillaRNN", train_loader, val_loader, device,
        epochs=epochs, lr=lr, ckpt_path=Path("artifacts/checkpoints/temporal/best_rnn.pt"),
    )

    lstm_hist = train_single_temporal_model(
        lstm, "LSTM", train_loader, val_loader, device,
        epochs=epochs, lr=lr, ckpt_path=Path("artifacts/checkpoints/temporal/best_lstm.pt"),
    )

    summary = {
        "VanillaRNN": rnn_hist,
        "LSTM": lstm_hist,
        "config": {"epochs": epochs, "batch_size": batch_size, "hidden_size": hidden_size, "lr": lr, "seed": seed},
    }

    hist_p = Path("reports/temporal_training_history.json")
    hist_p.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_temporal_training()

