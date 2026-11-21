"""Stage 6: Final Holdout Test Evaluation, Error Analysis & Benchmarking.

Evaluates the frozen selected model and all baseline candidates on the
holdout test set (data/splits/test.csv — 2,724 images).

Strict Scientific Safeguards:
1. Model selection was frozen prior to accessing the test set based on
   validation Macro F1:
   -> Frozen Selected Model: artifacts/checkpoints/best_resnet18_weighted.pt (Exp A)
2. Evaluation is strictly non-destructive (torch.no_grad()).
3. No training, fine-tuning, or parameter changes are performed.
4. Test set is evaluated exactly once for final benchmark reporting.

Author: Crop Disease Detection Project Team
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import PlantVillageTomatoDataset, get_transforms
from src.evaluation.error_analysis import (
    analyze_confusion_pairs,
    find_misclassifications,
    plot_misclassification_gallery,
)
from src.evaluation.metrics import (
    compute_classification_metrics,
    format_classification_report,
)
from src.evaluation.plots import (
    plot_model_benchmarking_bars,
    plot_confusion_matrix_heatmap,
    plot_per_class_metric_comparison,
)
from src.models.cnn import BaselineCNN, count_parameters
from src.models.transfer_learning import TomatoResNet18, count_trainable_parameters
from src.utils.device import get_device, get_device_info
from src.utils.reproducibility import set_seed


# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------

SEED = 42
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
NUM_WORKERS = 0

TEST_CSV = PROJECT_ROOT / "data/splits/test.csv"
CLASS_MAPPING_JSON = PROJECT_ROOT / "data/splits/class_mapping.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "artifacts/figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "artifacts/checkpoints"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    {
        "id": "baseline_cnn",
        "display_name": "BaselineCNN",
        "checkpoint": CHECKPOINTS_DIR / "best_baseline_cnn.pt",
        "arch": "cnn",
        "val_acc": 96.18,
        "val_macro_f1": 0.9511,
    },
    {
        "id": "resnet18_orig",
        "display_name": "ResNet18 (Stage 4)",
        "checkpoint": CHECKPOINTS_DIR / "best_resnet18.pt",
        "arch": "resnet18",
        "val_acc": 98.24,
        "val_macro_f1": 0.9771,
    },
    {
        "id": "exp_a_weighted",
        "display_name": "Exp A (Weighted Loss)",
        "checkpoint": CHECKPOINTS_DIR / "best_resnet18_weighted.pt",
        "arch": "resnet18",
        "val_acc": 99.12,
        "val_macro_f1": 0.9897,
        "selected_frozen_model": True,
    },
    {
        "id": "exp_b_aug_weighted",
        "display_name": "Exp B (Weighted + Aug)",
        "checkpoint": CHECKPOINTS_DIR / "best_resnet18_aug_weighted.pt",
        "arch": "resnet18",
        "val_acc": 98.53,
        "val_macro_f1": 0.9861,
    },
]


# ---------------------------------------------------------------------------
# Model Loader Helper
# ---------------------------------------------------------------------------

def load_candidate_model(cand_info: Dict[str, Any], device: torch.device) -> Tuple[nn.Module, int, float]:
    """Load PyTorch model from checkpoint and compute parameter count and file size."""
    ckpt_path = cand_info["checkpoint"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    file_size_mb = ckpt_path.stat().st_size / (1024 * 1024)

    if cand_info["arch"] == "cnn":
        model = BaselineCNN(num_classes=10)
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        num_params, _ = count_parameters(model)
    elif cand_info["arch"] == "resnet18":
        model = TomatoResNet18(num_classes=10, weights=None)
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        num_params, _ = count_trainable_parameters(model)
        # ResNet18 total parameters
        num_params = sum(p.numel() for p in model.parameters())
    else:
        raise ValueError(f"Unknown architecture: {cand_info['arch']}")

    model.to(device)
    model.eval()
    return model, num_params, file_size_mb


# ---------------------------------------------------------------------------
# Test Inference Helper
# ---------------------------------------------------------------------------

def run_test_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Run deterministic test-set inference under torch.no_grad().

    Returns:
        Tuple of (y_true, y_pred, y_probs, total_eval_time_sec, avg_latency_ms).
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    # Warmup pass (2 batches) to eliminate device initialization artifact
    with torch.no_grad():
        for i, (warm_img, _) in enumerate(dataloader):
            if i >= 2:
                break
            _ = model(warm_img.to(device))

    if device.type == "mps":
        torch.mps.synchronize()

    start_time = time.time()
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.numpy())
            all_probs.append(probs)
            total_samples += images.size(0)

    if device.type == "mps":
        torch.mps.synchronize()

    total_eval_time = time.time() - start_time
    avg_latency_ms = (total_eval_time / total_samples) * 1000.0

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs, axis=0)

    return y_true, y_pred, y_probs, total_eval_time, avg_latency_ms


# ---------------------------------------------------------------------------
# Main Stage 6 Pipeline
# ---------------------------------------------------------------------------

def run_stage6_evaluation():
    print("=" * 80)
    print("STAGE 6: FINAL HOLDOUT TEST EVALUATION & PROJECT BENCHMARKING")
    print("=" * 80)

    set_seed(SEED)
    device = get_device(preferred="mps")
    device_info = get_device_info(preferred="mps")
    print(f"Device: {device} ({device_info.get('device_name', 'Host')})")
    print(f"Test Manifest: {TEST_CSV}")

    # Load class mapping
    with open(CLASS_MAPPING_JSON, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    print(f"Classes ({len(class_names)}): {class_names}\n")

    # Load test dataset with deterministic evaluation transforms
    test_transform = get_transforms(image_size=IMAGE_SIZE, is_training=False)
    test_dataset = PlantVillageTomatoDataset(
        split_csv=TEST_CSV,
        root_dir=PROJECT_ROOT,
        transform=test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_file_paths = list(test_dataset.df["image_path"])
    print(f"Loaded Holdout Test Set: {len(test_dataset)} images in {len(test_loader)} batches.\n")

    # -----------------------------------------------------------------------
    # Step 1: Pre-Evaluation Audit & Frozen Model Selection Confirmation
    # -----------------------------------------------------------------------
    print("-" * 80)
    print("STEP 1: PRE-EVALUATION AUDIT & FINAL MODEL FREEZING")
    print("-" * 80)
    print("Candidate Review Based Solely on Validation Metrics:")
    for c in CANDIDATES:
        status = "★ SELECTED FINAL MODEL" if c.get("selected_frozen_model") else "Candidate"
        print(f"  - {c['display_name']:<28} Val Acc: {c['val_acc']:.2f}% | Val Macro F1: {c['val_macro_f1']:.4f} [{status}]")

    frozen_model_info = [c for c in CANDIDATES if c.get("selected_frozen_model")][0]
    print(f"\nFROZEN MODEL: {frozen_model_info['display_name']} ({frozen_model_info['checkpoint'].name})")
    print("Rationale: Highest Validation Macro F1 (0.9897) and lowest Validation Loss (0.0323).")
    print("Model selection is permanently frozen before test-set evaluation proceeds.")
    print("-" * 80 + "\n")

    # -----------------------------------------------------------------------
    # Step 2: Evaluate All Candidates for Benchmarking
    # -----------------------------------------------------------------------
    benchmark_results = []
    candidates_metrics_dict = {}
    selected_model_artifacts = {}

    for cand in CANDIDATES:
        print(f"Evaluating: {cand['display_name']}...")
        model, num_params, file_size_mb = load_candidate_model(cand, device)
        y_true, y_pred, y_probs, eval_time, latency_ms = run_test_inference(model, test_loader, device)
        del model  # Free GPU memory

        metrics = compute_classification_metrics(y_true, y_pred, class_names)
        candidates_metrics_dict[cand["display_name"]] = metrics

        bench_row = {
            "model_id": cand["id"],
            "model_display_name": cand["display_name"],
            "checkpoint_name": cand["checkpoint"].name,
            "val_accuracy": cand["val_acc"],
            "val_macro_f1": cand["val_macro_f1"],
            "test_accuracy": metrics["overall_accuracy"],
            "test_macro_precision": metrics["macro_precision"],
            "test_macro_recall": metrics["macro_recall"],
            "test_macro_f1": metrics["macro_f1"],
            "test_weighted_f1": metrics["weighted_f1"],
            "early_blight_recall": metrics["per_class"]["Tomato___Early_blight"]["recall"],
            "early_blight_f1": metrics["per_class"]["Tomato___Early_blight"]["f1_score"],
            "num_parameters": num_params,
            "checkpoint_size_mb": round(file_size_mb, 2),
            "eval_time_sec": round(eval_time, 2),
            "latency_ms_per_image": round(latency_ms, 2),
            "device": str(device),
        }
        benchmark_results.append(bench_row)

        print(f"  Test Acc: {metrics['overall_accuracy']:.2f}% | Test Macro F1: {metrics['macro_f1']:.4f} | "
              f"Early Blight Recall: {bench_row['early_blight_recall']*100:.2f}% | Latency: {latency_ms:.2f}ms/img")

        # Save artifacts specifically for the frozen selected model
        if cand.get("selected_frozen_model"):
            selected_model_artifacts = {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_probs": y_probs,
                "metrics": metrics,
                "eval_time": eval_time,
                "latency_ms": latency_ms,
            }

    print("\n" + "=" * 80)
    print("STEP 2 COMPLETED: ALL CANDIDATES EVALUATED ON HOLDOUT TEST SET")
    print("=" * 80 + "\n")

    # -----------------------------------------------------------------------
    # Step 3: Deep Error Analysis for the Frozen Selected Model
    # -----------------------------------------------------------------------
    print("-" * 80)
    print("STEP 3: COMPREHENSIVE ERROR ANALYSIS FOR SELECTED FINAL MODEL")
    print("-" * 80)

    sel_y_true = selected_model_artifacts["y_true"]
    sel_y_pred = selected_model_artifacts["y_pred"]
    sel_y_probs = selected_model_artifacts["y_probs"]
    sel_metrics = selected_model_artifacts["metrics"]

    errors = find_misclassifications(
        y_true=sel_y_true,
        y_pred=sel_y_pred,
        y_probs=sel_y_probs,
        file_paths=test_file_paths,
        class_names=class_names,
    )
    confused_pairs, per_class_fp_fn = analyze_confusion_pairs(
        y_true=sel_y_true,
        y_pred=sel_y_pred,
        class_names=class_names,
    )

    total_test = len(sel_y_true)
    total_errors = len(errors)
    test_accuracy = sel_metrics["overall_accuracy"]

    print(f"Total Test Samples: {total_test}")
    print(f"Correct Predictions: {total_test - total_errors} (Accuracy: {test_accuracy:.2f}%)")
    print(f"Total Misclassifications: {total_errors} (Error Rate: {100.0 - test_accuracy:.2f}%)\n")

    print("Top Confused Disease Pairs (Ground Truth -> Predicted):")
    for cp in confused_pairs[:6]:
        true_s = cp["true_class"].replace("Tomato___", "").replace("_", " ")
        pred_s = cp["pred_class"].replace("Tomato___", "").replace("_", " ")
        print(f"  - True: '{true_s}' -> Predicted: '{pred_s}': {cp['count']} errors")

    print("\nClass-wise False Positives and False Negatives:")
    for c_name in class_names:
        short_c = c_name.replace("Tomato___", "").replace("_", " ")
        fp = per_class_fp_fn[c_name]["false_positives"]
        fn = per_class_fp_fn[c_name]["false_negatives"]
        supp = sel_metrics["per_class"][c_name]["support"]
        rec = sel_metrics["per_class"][c_name]["recall"] * 100.0
        prec = sel_metrics["per_class"][c_name]["precision"] * 100.0
        f1 = sel_metrics["per_class"][c_name]["f1_score"] * 100.0
        print(f"  {short_c:<40} Supp: {supp:>4} | FN: {fn:>2} | FP: {fp:>2} | Rec: {rec:>5.1f}% | Prec: {prec:>5.1f}% | F1: {f1:>5.1f}%")

    # -----------------------------------------------------------------------
    # Step 4: Generate All Required Visualizations
    # -----------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 4: GENERATING PUBLICATION-QUALITY FIGURES")
    print("-" * 80)

    # 1. Confusion Matrix (Raw Counts)
    cm_raw_path = FIGURES_DIR / "final_test_confusion_matrix_raw.png"
    plot_confusion_matrix_heatmap(
        cm=sel_metrics["confusion_matrix"],
        class_names=class_names,
        output_path=cm_raw_path,
        title="Final Model (Exp A) — Holdout Test Set Confusion Matrix (Raw Counts)",
        normalize=False,
        cmap="Blues",
    )
    print(f"  [1/6] Raw Confusion Matrix: {cm_raw_path.name}")

    # 2. Confusion Matrix (Normalized Recall)
    cm_norm_path = FIGURES_DIR / "final_test_confusion_matrix_normalized.png"
    plot_confusion_matrix_heatmap(
        cm=sel_metrics["confusion_matrix"],
        class_names=class_names,
        output_path=cm_norm_path,
        title="Final Model (Exp A) — Holdout Test Set Confusion Matrix (Row-Normalized Recall)",
        normalize=True,
        cmap="Blues",
    )
    print(f"  [2/6] Normalized Confusion Matrix: {cm_norm_path.name}")

    # 3. Per-Class F1 Comparison Chart across Candidates
    f1_comp_path = FIGURES_DIR / "per_class_f1_comparison.png"
    plot_per_class_metric_comparison(
        candidates_dict=candidates_metrics_dict,
        class_names=class_names,
        output_path=f1_comp_path,
        metric_key="f1_score",
        title="Per-Class F1-Score Comparison Across All Models (Holdout Test Set)",
    )
    print(f"  [3/6] Per-Class F1 Comparison: {f1_comp_path.name}")

    # 4. Per-Class Recall Comparison Chart across Candidates
    recall_comp_path = FIGURES_DIR / "per_class_recall_comparison.png"
    plot_per_class_metric_comparison(
        candidates_dict=candidates_metrics_dict,
        class_names=class_names,
        output_path=recall_comp_path,
        metric_key="recall",
        title="Per-Class Recall Comparison Across All Models (Holdout Test Set)",
    )
    print(f"  [4/6] Per-Class Recall Comparison: {recall_comp_path.name}")

    # 5. Model Benchmarking Multi-Bar Chart
    bench_plot_path = FIGURES_DIR / "model_benchmarking_comparison.png"
    plot_model_benchmarking_bars(
        benchmark_data=benchmark_results,
        output_path=bench_plot_path,
    )
    print(f"  [5/6] Model Benchmarking Multi-Bar: {bench_plot_path.name}")

    # 6. Misclassification Gallery
    gallery_path = FIGURES_DIR / "misclassification_gallery.png"
    plot_misclassification_gallery(
        errors=errors,
        output_path=gallery_path,
        root_dir=PROJECT_ROOT,
        max_images=12,
    )
    print(f"  [6/6] Misclassification Gallery: {gallery_path.name}")

    # -----------------------------------------------------------------------
    # Step 5: Export Evaluation Reports and Benchmark CSV
    # -----------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 5: EXPORTING REPORTS AND BENCHMARK DATA")
    print("-" * 80)

    # 1. Benchmark CSV
    bench_df = pd.DataFrame(benchmark_results)
    bench_csv_path = REPORTS_DIR / "final_model_benchmark.csv"
    bench_df.to_csv(bench_csv_path, index=False)
    print(f"  Benchmark CSV saved: {bench_csv_path.name}")

    # 2. Detailed Selected Model Test Metrics JSON
    test_metrics_json_path = REPORTS_DIR / "final_test_evaluation_exp_a.json"
    detailed_export = {
        "frozen_selected_model": frozen_model_info["display_name"],
        "checkpoint": str(frozen_model_info["checkpoint"].relative_to(PROJECT_ROOT)),
        "selection_rationale": "Highest validation Macro F1 (0.9897) and lowest validation loss (0.0323) in Stage 5.",
        "test_metrics": sel_metrics,
        "evaluation_time_sec": selected_model_artifacts["eval_time"],
        "latency_ms_per_image": selected_model_artifacts["latency_ms"],
        "total_test_samples": total_test,
        "total_errors": total_errors,
        "top_confused_pairs": confused_pairs[:10],
        "per_class_fp_fn": per_class_fp_fn,
        "error_examples": errors[:15],
    }
    with open(test_metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_export, f, indent=2)
    print(f"  Test Evaluation JSON saved: {test_metrics_json_path.name}")

    print("\n" + "=" * 80)
    print("STAGE 6 TEST EVALUATION COMPLETE")
    print("=" * 80)

    return {
        "frozen_model": frozen_model_info["display_name"],
        "test_accuracy": test_accuracy,
        "test_macro_f1": sel_metrics["macro_f1"],
        "test_weighted_f1": sel_metrics["weighted_f1"],
        "total_errors": total_errors,
        "benchmark_results": benchmark_results,
        "figures": [
            str(cm_raw_path),
            str(cm_norm_path),
            str(f1_comp_path),
            str(recall_comp_path),
            str(bench_plot_path),
            str(gallery_path),
        ],
    }


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    run_stage6_evaluation()
