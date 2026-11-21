"""Run calibration pipeline: extract logits, fit temperature on val.csv, evaluate on test.csv."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.models.transfer_learning import TomatoResNet18
from src.data.dataset import create_dataloaders
from src.utils.device import get_device
from src.calibration.calibration_metrics import compute_calibration_metrics
from src.calibration.temperature_scaling import TemperatureScaler
from src.calibration.reliability_diagram import plot_reliability_diagram, plot_confidence_histograms


def extract_logits_and_labels(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model inference to collect all raw logits and ground truth labels."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            all_logits.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_labels, axis=0)
    return logits, targets


def run_model_calibration(
    checkpoint_path: Path = PROJECT_ROOT / "artifacts/checkpoints/best_resnet18_weighted.pt",
    splits_dir: Path = PROJECT_ROOT / "data/splits",
    output_dir: Path = PROJECT_ROOT / "artifacts/figures/calibration",
    report_csv: Path = PROJECT_ROOT / "reports/final_calibration_results.csv",
) -> Dict[str, any]:
    """Execute end-to-end model calibration on validation and holdout test sets."""
    print("=" * 70)
    print("STARTING MODEL CALIBRATION PIPELINE (Phase F — REQ-10)")
    print("=" * 70)

    device = get_device(preferred="mps")
    print(f"Device: {device}")

    # 1. Load production model
    print(f"Loading checkpoint: {checkpoint_path}")
    model = TomatoResNet18(num_classes=10, weights=None)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # 2. Data loaders
    print("Creating validation and test dataloaders...")
    loaders = create_dataloaders(splits_dir=splits_dir, batch_size=64, num_workers=0)
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    print(f"Extracting validation logits (N = {len(val_loader.dataset)})...")
    val_logits, val_labels = extract_logits_and_labels(model, val_loader, device)

    print(f"Extracting test logits (N = {len(test_loader.dataset)})...")
    test_logits, test_labels = extract_logits_and_labels(model, test_loader, device)

    # 3. Fit Temperature Scaler strictly on validation data
    print("\nFitting Temperature Scaler on validation set...")
    scaler = TemperatureScaler(initial_temp=1.5, max_iter=100, lr=0.05)
    optimal_t = scaler.fit(val_logits, val_labels, verbose=True)

    # 4. Compute probabilities
    # Uncalibrated (Softmax with T=1.0)
    val_probs_uncal = torch.softmax(torch.from_numpy(val_logits), dim=-1).numpy()
    test_probs_uncal = torch.softmax(torch.from_numpy(test_logits), dim=-1).numpy()

    # Calibrated (Softmax with T=T*)
    val_probs_cal = scaler.transform(val_logits)
    test_probs_cal = scaler.transform(test_logits)

    # 5. Compute metrics
    metrics_val_uncal = compute_calibration_metrics(val_probs_uncal, val_labels, n_bins=15)
    metrics_val_cal = compute_calibration_metrics(val_probs_cal, val_labels, n_bins=15)

    metrics_test_uncal = compute_calibration_metrics(test_probs_uncal, test_labels, n_bins=15)
    metrics_test_cal = compute_calibration_metrics(test_probs_cal, test_labels, n_bins=15)

    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Optimal Temperature T*: {optimal_t:.4f}")
    print(f"{'Split / Metric':<25} | {'Uncalibrated':<15} | {'Calibrated':<15} | {'Improvement':<15}")
    print("-" * 75)

    eval_table = [
        ("Val ECE (15 bins)", f"{metrics_val_uncal['ece']*100:.3f}%", f"{metrics_val_cal['ece']*100:.3f}%", f"{(metrics_val_uncal['ece'] - metrics_val_cal['ece'])*100:+.3f}%"),
        ("Val MCE (15 bins)", f"{metrics_val_uncal['mce']*100:.3f}%", f"{metrics_val_cal['mce']*100:.3f}%", f"{(metrics_val_uncal['mce'] - metrics_val_cal['mce'])*100:+.3f}%"),
        ("Val NLL", f"{metrics_val_uncal['nll']:.5f}", f"{metrics_val_cal['nll']:.5f}", f"{metrics_val_uncal['nll'] - metrics_val_cal['nll']:+.5f}"),
        ("Val Brier Score", f"{metrics_val_uncal['brier_score']:.5f}", f"{metrics_val_cal['brier_score']:.5f}", f"{metrics_val_uncal['brier_score'] - metrics_val_cal['brier_score']:+.5f}"),
        ("Val Accuracy", f"{metrics_val_uncal['accuracy']*100:.2f}%", f"{metrics_val_cal['accuracy']*100:.2f}%", "0.00% (Preserved)"),
        ("-" * 25, "-" * 15, "-" * 15, "-" * 15),
        ("Test ECE (15 bins)", f"{metrics_test_uncal['ece']*100:.3f}%", f"{metrics_test_cal['ece']*100:.3f}%", f"{(metrics_test_uncal['ece'] - metrics_test_cal['ece'])*100:+.3f}%"),
        ("Test MCE (15 bins)", f"{metrics_test_uncal['mce']*100:.3f}%", f"{metrics_test_cal['mce']*100:.3f}%", f"{(metrics_test_uncal['mce'] - metrics_test_cal['mce'])*100:+.3f}%"),
        ("Test NLL", f"{metrics_test_uncal['nll']:.5f}", f"{metrics_test_cal['nll']:.5f}", f"{metrics_test_uncal['nll'] - metrics_test_cal['nll']:+.5f}"),
        ("Test Brier Score", f"{metrics_test_uncal['brier_score']:.5f}", f"{metrics_test_cal['brier_score']:.5f}", f"{metrics_test_uncal['brier_score'] - metrics_test_cal['brier_score']:+.5f}"),
        ("Test Accuracy", f"{metrics_test_uncal['accuracy']*100:.2f}%", f"{metrics_test_cal['accuracy']*100:.2f}%", "0.00% (Preserved)"),
        ("Test Mean Confidence", f"{metrics_test_uncal['mean_confidence']*100:.2f}%", f"{metrics_test_cal['mean_confidence']*100:.2f}%", f"{(metrics_test_cal['mean_confidence'] - metrics_test_uncal['mean_confidence'])*100:+.2f}%"),
    ]

    for row in eval_table:
        print(f"{row[0]:<25} | {row[1]:<15} | {row[2]:<15} | {row[3]:<15}")
    print("=" * 75)

    # 6. Save visualizations
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_path = output_dir / "reliability_diagram_test.png"
    hist_path = output_dir / "confidence_histogram_test.png"

    print(f"\nSaving reliability diagram to: {diag_path}")
    plot_reliability_diagram(test_probs_uncal, test_probs_cal, test_labels, save_path=str(diag_path))

    print(f"Saving confidence histogram to: {hist_path}")
    plot_confidence_histograms(test_probs_uncal, test_probs_cal, save_path=str(hist_path))

    # 7. Export results to CSV
    df_results = pd.DataFrame([
        {
            "Split": "validation",
            "Optimal_T": optimal_t,
            "Accuracy_Uncal": metrics_val_uncal["accuracy"],
            "Accuracy_Cal": metrics_val_cal["accuracy"],
            "ECE_Uncal": metrics_val_uncal["ece"],
            "ECE_Cal": metrics_val_cal["ece"],
            "MCE_Uncal": metrics_val_uncal["mce"],
            "MCE_Cal": metrics_val_cal["mce"],
            "Brier_Uncal": metrics_val_uncal["brier_score"],
            "Brier_Cal": metrics_val_cal["brier_score"],
            "NLL_Uncal": metrics_val_uncal["nll"],
            "NLL_Cal": metrics_val_cal["nll"],
            "MeanConf_Uncal": metrics_val_uncal["mean_confidence"],
            "MeanConf_Cal": metrics_val_cal["mean_confidence"],
        },
        {
            "Split": "test",
            "Optimal_T": optimal_t,
            "Accuracy_Uncal": metrics_test_uncal["accuracy"],
            "Accuracy_Cal": metrics_test_cal["accuracy"],
            "ECE_Uncal": metrics_test_uncal["ece"],
            "ECE_Cal": metrics_test_cal["ece"],
            "MCE_Uncal": metrics_test_uncal["mce"],
            "MCE_Cal": metrics_test_cal["mce"],
            "Brier_Uncal": metrics_test_uncal["brier_score"],
            "Brier_Cal": metrics_test_cal["brier_score"],
            "NLL_Uncal": metrics_test_uncal["nll"],
            "NLL_Cal": metrics_test_cal["nll"],
            "MeanConf_Uncal": metrics_test_uncal["mean_confidence"],
            "MeanConf_Cal": metrics_test_cal["mean_confidence"],
        },
    ])
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(report_csv, index=False)
    print(f"Exported calibration benchmark to: {report_csv}")

    # Also save calibrated temperature artifact for production use
    temp_meta_path = PROJECT_ROOT / "artifacts/checkpoints/temperature_scale.json"
    temp_metadata = {
        "optimal_temperature": optimal_t,
        "calibration_dataset": "data/splits/val.csv (N=1816)",
        "evaluation_dataset": "data/splits/test.csv (N=2724)",
        "method": "Temperature Scaling (L-BFGS optimization on NLL with strong Wolfe line search)",
        "test_ece_uncalibrated": metrics_test_uncal["ece"],
        "test_ece_calibrated": metrics_test_cal["ece"],
        "test_mce_uncalibrated": metrics_test_uncal["mce"],
        "test_mce_calibrated": metrics_test_cal["mce"],
        "test_brier_uncalibrated": metrics_test_uncal["brier_score"],
        "test_brier_calibrated": metrics_test_cal["brier_score"],
        "test_accuracy": metrics_test_uncal["accuracy"],
    }
    with open(temp_meta_path, "w") as f:
        json.dump(temp_metadata, f, indent=2)
    print(f"Saved temperature calibration metadata to: {temp_meta_path}")

    return {
        "optimal_temperature": optimal_t,
        "val_metrics_uncal": metrics_val_uncal,
        "val_metrics_cal": metrics_val_cal,
        "test_metrics_uncal": metrics_test_uncal,
        "test_metrics_cal": metrics_test_cal,
    }


if __name__ == "__main__":
    run_model_calibration()
