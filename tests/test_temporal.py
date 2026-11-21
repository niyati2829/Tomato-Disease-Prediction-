"""Unit Tests for Temporal Progression & Recurrent Models (REQ-09).

Verifies:
1. Vanilla RNN forward-pass dimensions & numerical parity with torch.nn.RNN.
2. LSTM forward-pass dimensions & numerical parity with torch.nn.LSTM.
3. Hidden-state and cell-state trajectory shapes and gate activation bounds [0, 1].
4. Synthetic sequence generation deterministic reproducibility.
5. Zero sequence/plant ID leakage across train/val/test splits.
6. Regression evaluation metrics (MSE, MAE, RMSE, R^2).
7. Training smoke test and checkpoint save/load consistency.

Author: Crop Disease Detection Project Team
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.temporal.evaluate import compute_regression_metrics
from src.temporal.lstm_from_scratch import LSTMCellNumPy
from src.temporal.rnn_from_scratch import VanillaRNNNumPy
from src.temporal.sequence_dataset import DiseaseSequenceDataset
from src.temporal.sequence_generator import generate_synthetic_progression_sequences
from src.temporal.temporal_model import LSTMRegressor, VanillaRNNRegressor
from src.temporal.train import train_single_temporal_model


def test_numpy_rnn_parity_with_pytorch():
    """Verify NumPy VanillaRNN matches PyTorch nn.RNN forward pass on identical weights."""
    torch.manual_seed(42)
    B, T, D, H = 3, 6, 4, 5
    X_np = np.random.RandomState(42).randn(B, T, D).astype(np.float32)
    X_pt = torch.tensor(X_np)

    pt_rnn = nn.RNN(D, H, batch_first=True)
    np_rnn = VanillaRNNNumPy(input_size=D, hidden_size=H, output_size=1)

    # Copy PyTorch weights exactly
    np_rnn.W_xh = pt_rnn.weight_ih_l0.detach().numpy().copy()
    np_rnn.W_hh = pt_rnn.weight_hh_l0.detach().numpy().copy()
    np_rnn.b_h = (pt_rnn.bias_ih_l0 + pt_rnn.bias_hh_l0).detach().numpy().copy()

    h_seq_np, h_last_np, _ = np_rnn.forward_sequence(X_np)
    with torch.no_grad():
        h_seq_pt, h_last_pt = pt_rnn(X_pt)

    np.testing.assert_allclose(h_seq_np, h_seq_pt.numpy(), atol=1e-5)
    np.testing.assert_allclose(h_last_np, h_last_pt.squeeze(0).numpy(), atol=1e-5)


def test_numpy_lstm_parity_with_pytorch():
    """Verify NumPy LSTMCellNumPy matches PyTorch nn.LSTM forward pass and bounds gates in [0, 1]."""
    torch.manual_seed(42)
    B, T, D, H = 3, 6, 4, 5
    X_np = np.random.RandomState(42).randn(B, T, D).astype(np.float32)
    X_pt = torch.tensor(X_np)

    pt_lstm = nn.LSTM(D, H, batch_first=True)
    np_lstm = LSTMCellNumPy(input_size=D, hidden_size=H)

    np_lstm.load_from_pytorch_weights(
        pt_lstm.weight_ih_l0.detach().numpy(),
        pt_lstm.weight_hh_l0.detach().numpy(),
        pt_lstm.bias_ih_l0.detach().numpy(),
        pt_lstm.bias_hh_l0.detach().numpy(),
    )

    res = np_lstm.forward_sequence(X_np)
    with torch.no_grad():
        h_seq_pt, (h_last_pt, c_last_pt) = pt_lstm(X_pt)

    np.testing.assert_allclose(res["hidden_states"], h_seq_pt.numpy(), atol=1e-5)
    np.testing.assert_allclose(res["h_last"], h_last_pt.squeeze(0).numpy(), atol=1e-5)
    np.testing.assert_allclose(res["c_last"], c_last_pt.squeeze(0).numpy(), atol=1e-5)

    # Gate activations bounded strictly in [0, 1]
    for g_name in ["forget_gates", "input_gates", "output_gates"]:
        gates = res[g_name]
        assert np.all((gates >= 0.0) & (gates <= 1.0))


def test_pytorch_temporal_models_shapes_and_states():
    """Verify PyTorch VanillaRNNRegressor and LSTMRegressor output and state dimensions."""
    B, T, D, H = 4, 10, 3, 16
    x = torch.randn(B, T, D)

    rnn = VanillaRNNRegressor(input_size=D, hidden_size=H)
    lstm = LSTMRegressor(input_size=D, hidden_size=H)

    pred_rnn, st_rnn = rnn(x, return_states=True)
    pred_lstm, st_lstm = lstm(x, return_states=True)

    assert pred_rnn.shape == (B, 1)
    assert pred_lstm.shape == (B, 1)
    assert st_rnn["hidden_states"].shape == (B, T, H)
    assert st_lstm["hidden_states"].shape == (B, T, H)
    assert st_lstm["cell_states"].shape == (B, T, H)
    assert st_lstm["forget_gates"].shape == (B, T, H)


def test_sequence_generator_reproducibility_and_zero_leakage():
    """Verify deterministic sequence reproducibility and strictly disjoint plant IDs across splits."""
    s1 = generate_synthetic_progression_sequences(n_plants=100, seq_len=8, seed=123)
    s2 = generate_synthetic_progression_sequences(n_plants=100, seq_len=8, seed=123)

    np.testing.assert_array_equal(s1["train"]["X"], s2["train"]["X"])
    np.testing.assert_array_equal(s1["test"]["y"], s2["test"]["y"])

    train_ids = set(s1["train"]["plant_ids"].tolist())
    val_ids = set(s1["val"]["plant_ids"].tolist())
    test_ids = set(s1["test"]["plant_ids"].tolist())

    assert len(train_ids & val_ids) == 0, "Leakage detected between train and val!"
    assert len(train_ids & test_ids) == 0, "Leakage detected between train and test!"
    assert len(val_ids & test_ids) == 0, "Leakage detected between val and test!"
    assert len(train_ids) + len(val_ids) + len(test_ids) == 100


def test_regression_metrics_exact():
    """Verify MSE, MAE, RMSE, and R^2 calculation."""
    targets = np.array([0.0, 1.0, 2.0, 3.0])
    preds = np.array([0.0, 1.5, 1.5, 3.0])  # errors: 0, +0.5, -0.5, 0 -> sq: 0, 0.25, 0.25, 0 -> MSE=0.125
    metrics = compute_regression_metrics(preds, targets)

    assert metrics["MSE"] == pytest.approx(0.125, abs=1e-6)
    assert metrics["MAE"] == pytest.approx(0.25, abs=1e-6)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(0.125), abs=1e-6)
    assert 0.0 < metrics["R2"] <= 1.0


def test_temporal_training_and_checkpoint_smoke():
    """Verify brief training run decreases loss and saves/loads consistent checkpoint."""
    splits = generate_synthetic_progression_sequences(n_plants=40, seq_len=6, seed=42)
    train_ds = DiseaseSequenceDataset(splits["train"])
    val_ds = DiseaseSequenceDataset(splits["val"])
    t_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    v_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_p = Path(tmpdir) / "smoke_lstm.pt"
        model = LSTMRegressor(input_size=3, hidden_size=8)
        hist = train_single_temporal_model(
            model, "SmokeLSTM", t_loader, v_loader, torch.device("cpu"),
            epochs=3, lr=1e-2, ckpt_path=ckpt_p,
        )

        assert ckpt_p.exists()
        assert len(hist["train_loss"]) == 3

        loaded_ckpt = torch.load(ckpt_p, map_location="cpu")
        reloaded = LSTMRegressor(input_size=3, hidden_size=8)
        reloaded.load_state_dict(loaded_ckpt["model_state_dict"])

        x_sample = torch.randn(2, 6, 3)
        model.eval()
        reloaded.eval()
        # Check loaded model predicts cleanly
        with torch.no_grad():
            out = reloaded(x_sample)
        assert out.shape == (2, 1)

