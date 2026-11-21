"""Run comprehensive robustness and adversarial evaluation for tomato leaf classification."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image
import torch

from src.models.transfer_learning import TomatoResNet18
from src.data.dataset import get_datasets, create_dataloaders
from src.utils.device import get_device
from src.robustness.robustness_evaluation import (
    evaluate_corruption_robustness,
    plot_robustness_curves,
    plot_corruption_panel,
    plot_fgsm_curve,
)
from src.robustness.adversarial_tests import evaluate_fgsm_robustness


def run_robustness_pipeline(
    checkpoint_path: Path = PROJECT_ROOT / "artifacts/checkpoints/best_resnet18_weighted.pt",
    splits_dir: Path = PROJECT_ROOT / "data/splits",
    output_dir: Path = PROJECT_ROOT / "artifacts/figures/robustness",
    report_csv: Path = PROJECT_ROOT / "reports/final_robustness_results.csv",
    max_samples: int = 500,
):
    print("=" * 70)
    print("STARTING ROBUSTNESS & ADVERSARIAL EVALUATION (Phase F — REQ-10)")
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

    # 2. Load dataset
    print(f"Loading test dataset from {splits_dir}...")
    datasets = get_datasets(splits_dir=splits_dir)
    test_dataset = datasets["test"]

    # 3. Visual corruption panel
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_row = test_dataset.df.iloc[0]
    sample_img_path = test_dataset.root_dir / sample_row["image_path"]
    with Image.open(sample_img_path) as img:
        sample_img = img.convert("RGB")

    panel_path = output_dir / "corruption_visual_panel.png"
    print(f"Generating visual corruption panel -> {panel_path}")
    plot_corruption_panel(sample_img, save_path=str(panel_path))

    # 4. Evaluate Corruptions across Severities
    print(f"\nBenchmarking 8 Environmental & Sensor Corruptions (N = {max_samples})...")
    results = evaluate_corruption_robustness(
        model=model,
        dataset=test_dataset,
        device=device,
        severities=[1, 2, 3],
        max_samples=max_samples,
    )

    curves_path = output_dir / "robustness_curves.png"
    print(f"\nSaving robustness curves -> {curves_path}")
    plot_robustness_curves(results, save_path=str(curves_path))

    # 5. Format and Export Corruption Results Table
    clean_acc = results["clean"][0]["accuracy"]
    clean_conf = results["clean"][0]["mean_confidence"]

    rows = []
    for corr, sev_dict in results.items():
        if corr == "clean":
            continue
        for sev, stats in sev_dict.items():
            acc = stats["accuracy"]
            conf = stats["mean_confidence"]
            rows.append({
                "Corruption": corr,
                "Severity": sev,
                "Accuracy": acc,
                "Accuracy_Drop": clean_acc - acc,
                "Relative_Retention": acc / clean_acc if clean_acc > 0 else 0.0,
                "Mean_Confidence": conf,
                "Confidence_Drop": clean_conf - conf,
            })

    df_corruptions = pd.DataFrame(rows)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    df_corruptions.to_csv(report_csv, index=False)
    print(f"Saved corruption results table to {report_csv}")

    # 6. Adversarial Robustness Testing (FGSM)
    print("\nBenchmarking FGSM Adversarial Perturbations...")
    loaders = create_dataloaders(splits_dir=splits_dir, batch_size=32, num_workers=0)
    test_loader = loaders["test"]
    epsilons = [0.0, 0.005, 0.01, 0.02, 0.05]
    fgsm_results = evaluate_fgsm_robustness(
        model=model,
        dataloader=test_loader,
        epsilons=epsilons,
        device=device,
        max_samples=300,
    )

    fgsm_curve_path = output_dir / "fgsm_robustness_curve.png"
    print(f"Saving FGSM degradation curve -> {fgsm_curve_path}")
    plot_fgsm_curve(fgsm_results, save_path=str(fgsm_curve_path))

    fgsm_csv = PROJECT_ROOT / "reports/final_fgsm_results.csv"
    fgsm_rows = [
        {"Epsilon": eps, "Accuracy": res["accuracy"], "Mean_Confidence": res["mean_confidence"], "Samples": res["samples"]}
        for eps, res in fgsm_results.items()
    ]
    pd.DataFrame(fgsm_rows).to_csv(fgsm_csv, index=False)
    print(f"Saved FGSM results table to {fgsm_csv}")

    print("\nRobustness evaluation pipeline completed successfully!")
    return results, fgsm_results


if __name__ == "__main__":
    run_robustness_pipeline()
