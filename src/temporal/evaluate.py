"""Evaluation Engine Comparing Vanilla RNN vs LSTM on Untouched Synthetic Test Split (REQ-09).

Reports MSE, MAE, RMSE, R^2, per-pattern breakdown, and inference latency.
Saves results to reports/temporal_evaluation_results.json.

Author: Crop Disease Detection Project Team
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.temporal.sequence_dataset import DiseaseSequenceDataset
from src.temporal.sequence_generator import PATTERN_NAMES, generate_synthetic_progression_sequences
from src.temporal.temporal_model import LSTMRegressor, VanillaRNNRegressor
from src.utils.device import get_device


def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute MSE, MAE, RMSE, and R^2."""
    diff = preds - targets
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}


def evaluate_temporal_checkpoint(
    model_cls,
    ckpt_path: Path,
    test_ds: DiseaseSequenceDataset,
    device: torch.device,
) -> Dict[str, Any]:
    """Load checkpoint and evaluate on test set."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = model_cls(
        input_size=ckpt.get("input_size", 3),
        hidden_size=ckpt.get("hidden_size", 32),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    X_test, y_test, patterns, _ = next(iter(loader))
    X_test = X_test.to(device)

    # Measure single-sample inference latency
    latencies = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            x_single = X_test[i : i + 1]
            t0 = time.perf_counter()
            _ = model(x_single)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        preds = model(X_test).cpu().numpy().ravel()

    targets = y_test.numpy().ravel()
    patterns_np = patterns.numpy().ravel()

    overall = compute_regression_metrics(preds, targets)
    per_pattern = {}
    for p_id, p_name in PATTERN_NAMES.items():
        mask = (patterns_np == p_id)
        if np.any(mask):
            per_pattern[p_name] = compute_regression_metrics(preds[mask], targets[mask])

    return {
        "model_name": ckpt.get("model_name", model_cls.__name__),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "test_samples": len(test_ds),
        "inference_latency_ms": float(np.mean(latencies)),
        "overall_metrics": overall,
        "per_pattern_metrics": per_pattern,
        "predictions": preds.tolist(),
        "targets": targets.tolist(),
    }


def run_temporal_evaluation(seed: int = 42) -> Dict[str, Any]:
    """Evaluate both Vanilla RNN and LSTM checkpoints on synthetic test data."""
    device = get_device(preferred="mps")
    splits = generate_synthetic_progression_sequences(n_plants=600, seq_len=10, seed=seed)
    test_ds = DiseaseSequenceDataset(splits["test"])

    rnn_res = evaluate_temporal_checkpoint(
        VanillaRNNRegressor, Path("artifacts/checkpoints/temporal/best_rnn.pt"), test_ds, device
    )
    lstm_res = evaluate_temporal_checkpoint(
        LSTMRegressor, Path("artifacts/checkpoints/temporal/best_lstm.pt"), test_ds, device
    )

    report = {
        "note": "Evaluated strictly on synthetic disease progression test split (90 unseen simulated plants).",
        "VanillaRNN": {k: v for k, v in rnn_res.items() if k not in ["predictions", "targets"]},
        "LSTM": {k: v for k, v in lstm_res.items() if k not in ["predictions", "targets"]},
    }

    report_p = Path("reports/temporal_evaluation_results.json")
    report_p.parent.mkdir(parents=True, exist_ok=True)
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n================ Synthetic Temporal Test Evaluation ================")
    for name, res in [("VanillaRNN", report["VanillaRNN"]), ("LSTM", report["LSTM"])]:
        m = res["overall_metrics"]
        print(
            f"{name:12s} | Params: {res['parameter_count']:,} | MAE: {m['MAE']:.5f} | "
            f"RMSE: {m['RMSE']:.5f} | R2: {m['R2']:.4f} | Latency: {res['inference_latency_ms']:.3f} ms"
        )
    print(f"Results saved to: {report_p}")

    return {"report": report, "rnn_full": rnn_res, "lstm_full": lstm_res}


if __name__ == "__main__":
    run_temporal_evaluation()

