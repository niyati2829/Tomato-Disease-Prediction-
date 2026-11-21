"""Phase M2: U-Net Holdout Test Split Comparative Evaluation Engine.

Compares the improved real U-Net checkpoint against the original Phase K baseline
on the untouched 99-image PlantSeg test split under strictly identical conditions.

Disciplines:
- Both models evaluated strictly on data/processed/tomato_lesions_plantseg/splits/test.csv (99 images).
- Fixed threshold of 0.50 for both models (no test-set threshold tuning).
- Identical preprocessing: 256x256, aspect ratio padding, ImageNet normalization.
- Calculates: Macro Dice, Macro IoU, Macro Precision, Macro Recall, Pixel Accuracy, Micro Dice & IoU.
- Disease-level and per-image comparative analysis.
- Generates 6-panel prediction and error overlays (TP=Green, FP=Red, FN=Blue).
- Reports CPU inference latency benchmarks under identical conditions.
- Preserves all existing checkpoints, datasets, splits, and production modules.

Author: Crop Disease Detection Project Team
Phase: Phase M2
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.segmentation.plantseg_dataset import PlantSegTomatoLesionDataset
from src.segmentation.unet import UNet
from src.utils.device import get_device


def load_unet_checkpoint(checkpoint_path: Path | str, device: torch.device) -> Tuple[UNet, Dict[str, Any]]:
    """Load U-Net model from checkpoint with metadata verification."""
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_p}")

    checkpoint = torch.load(ckpt_p, map_location=device)
    init_features = checkpoint.get("init_features", 16)
    in_channels = checkpoint.get("in_channels", 3)
    num_classes = checkpoint.get("num_classes", 1)

    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        init_features=init_features,
        bilinear=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    meta = {
        "checkpoint_path": str(ckpt_p),
        "epoch": checkpoint.get("epoch", "N/A"),
        "val_dice": checkpoint.get("val_dice", "N/A"),
        "val_loss": checkpoint.get("val_loss", "N/A"),
        "init_features": init_features,
        "loss_type": checkpoint.get("loss_type", "bce_dice"),
        "augmentation": checkpoint.get("augmentation", False),
    }
    return model, meta


def benchmark_cpu_latency(
    model: UNet,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    n_warmup: int = 15,
    n_runs: int = 99,
) -> Dict[str, float]:
    """Benchmark inference latency strictly on CPU under deterministic conditions."""
    cpu_device = torch.device("cpu")
    model_cpu = model.to(cpu_device)
    model_cpu.eval()

    dummy_input = torch.randn(*input_shape, device=cpu_device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model_cpu(dummy_input)

    # Timed runs
    latencies_ms = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model_cpu(dummy_input)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies_ms))
    p50_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    std_lat = float(np.std(latencies_ms))
    throughput = float(1000.0 / max(mean_lat, 1e-4))

    return {
        "device": "cpu",
        "num_runs": n_runs,
        "warmup_runs": n_warmup,
        "mean_latency_ms": round(mean_lat, 2),
        "median_p50_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "throughput_fps": round(throughput, 1),
    }


def evaluate_model_on_dataset(
    model: UNet,
    dataset: PlantSegTomatoLesionDataset,
    device: torch.device,
    threshold: float = 0.50,
    model_tag: str = "model",
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Run full evaluation on dataset, computing per-image and aggregate metrics."""
    smooth = 1e-5
    records = []
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    per_image_preds = {}

    with torch.no_grad():
        for i in range(len(dataset)):
            img_tensor, mask_tensor = dataset[i]
            meta = dataset.get_metadata(i)

            img_input = img_tensor.unsqueeze(0).to(device)
            target = mask_tensor.unsqueeze(0).to(device)

            logits = model(img_input)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            tp = float((preds * target).sum().item())
            fp = float((preds * (1.0 - target)).sum().item())
            fn = float(((1.0 - preds) * target).sum().item())
            tn = float(((1.0 - preds) * (1.0 - target)).sum().item())

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
            iou = (tp + smooth) / (tp + fp + fn + smooth)
            precision = (tp + smooth) / (tp + fp + smooth)
            recall = (tp + smooth) / (tp + fn + smooth)
            accuracy = (tp + tn) / target.numel()

            img_name = meta["image_name"]
            records.append({
                "image_name": img_name,
                "disease": meta["disease"],
                f"{model_tag}_dice": round(dice, 4),
                f"{model_tag}_iou": round(iou, 4),
                f"{model_tag}_precision": round(precision, 4),
                f"{model_tag}_recall": round(recall, 4),
                f"{model_tag}_pixel_accuracy": round(accuracy, 4),
                f"{model_tag}_tp_pixels": int(tp),
                f"{model_tag}_fp_pixels": int(fp),
                f"{model_tag}_fn_pixels": int(fn),
                f"{model_tag}_tn_pixels": int(tn),
                "gt_lesion_pixels": int(tp + fn),
                f"{model_tag}_pred_lesion_pixels": int(tp + fp),
                "gt_lesion_ratio": round(float((tp + fn) / target.numel()), 4),
                f"{model_tag}_pred_lesion_ratio": round(float((tp + fp) / target.numel()), 4),
            })

            per_image_preds[img_name] = {
                "probs": probs[0, 0].cpu(),
                "preds": preds[0, 0].cpu(),
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

    df = pd.DataFrame(records)

    # Aggregates
    macro_dice = float(df[f"{model_tag}_dice"].mean())
    macro_iou = float(df[f"{model_tag}_iou"].mean())
    macro_precision = float(df[f"{model_tag}_precision"].mean())
    macro_recall = float(df[f"{model_tag}_recall"].mean())
    macro_pixel_accuracy = float(df[f"{model_tag}_pixel_accuracy"].mean())

    micro_dice = float((2.0 * total_tp + smooth) / (2.0 * total_tp + total_fp + total_fn + smooth))
    micro_iou = float((total_tp + smooth) / (total_tp + total_fp + total_fn + smooth))
    micro_precision = float((total_tp + smooth) / (total_tp + total_fp + smooth))
    micro_recall = float((total_tp + smooth) / (total_tp + total_fn + smooth))
    micro_pixel_accuracy = float((total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn))

    summary = {
        "macro_metrics": {
            "macro_dice": round(macro_dice, 4),
            "macro_iou": round(macro_iou, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_pixel_accuracy": round(macro_pixel_accuracy, 4),
        },
        "micro_metrics": {
            "micro_dice": round(micro_dice, 4),
            "micro_iou": round(micro_iou, 4),
            "micro_precision": round(micro_precision, 4),
            "micro_recall": round(micro_recall, 4),
            "micro_pixel_accuracy": round(micro_pixel_accuracy, 4),
        },
        "confusion_pixels": {
            "total_tp": int(total_tp),
            "total_fp": int(total_fp),
            "total_fn": int(total_fn),
            "total_tn": int(total_tn),
        },
    }

    return df, summary, per_image_preds


def render_comparison_overlays(
    dataset: PlantSegTomatoLesionDataset,
    baseline_preds: Dict[str, Dict[str, Any]],
    improved_preds: Dict[str, Dict[str, Any]],
    output_dir: Path,
    samples_per_disease: int = 1,
) -> List[str]:
    """Generate 6-panel visual comparison figures across representative test images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    disease_counts = {}

    for i in range(len(dataset)):
        meta = dataset.get_metadata(i)
        disease = meta["disease"]
        img_name = meta["image_name"]

        if disease_counts.get(disease, 0) >= samples_per_disease:
            continue

        disease_counts[disease] = disease_counts.get(disease, 0) + 1

        img_tensor, mask_tensor = dataset[i]
        img_t = img_tensor.permute(1, 2, 0).numpy()
        img_rgb = np.clip((img_t * std + mean) * 255.0, 0, 255).astype(np.uint8)
        gt_mask = mask_tensor[0].numpy() > 0.5

        b_data = baseline_preds[img_name]
        i_data = improved_preds[img_name]

        # Grayscale background for error maps
        gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        # Baseline error map
        b_pred = b_data["preds"].numpy() > 0.5
        b_diff = np.stack([gray * 0.35] * 3, axis=-1).astype(np.uint8)
        b_diff[b_pred & gt_mask] = [40, 220, 60]    # TP: Green
        b_diff[b_pred & (~gt_mask)] = [230, 40, 40] # FP: Red
        b_diff[(~b_pred) & gt_mask] = [40, 100, 240] # FN: Blue

        # Improved error map
        i_pred = i_data["preds"].numpy() > 0.5
        i_diff = np.stack([gray * 0.35] * 3, axis=-1).astype(np.uint8)
        i_diff[i_pred & gt_mask] = [40, 220, 60]    # TP: Green
        i_diff[i_pred & (~gt_mask)] = [230, 40, 40] # FP: Red
        i_diff[(~i_pred) & gt_mask] = [40, 100, 240] # FN: Blue

        # 6-panel comparison figure
        fig, axes = plt.subplots(1, 6, figsize=(27, 4.5))

        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Test RGB Image (256x256)\n{img_name}\nDisease: {disease}", fontsize=8)
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Ground-Truth Mask\nLesion Ratio: {gt_mask.mean():.3f}", fontsize=8)
        axes[1].axis("off")

        im_b_prob = axes[2].imshow(b_data["probs"].numpy(), cmap="inferno", vmin=0.0, vmax=1.0)
        axes[2].set_title("Baseline Prob Map (Sigmoid)\nPhase K Model", fontsize=8)
        axes[2].axis("off")

        axes[3].imshow(b_diff)
        axes[3].set_title(
            f"Baseline Error Map\nDice: {b_data['dice']:.3f} | Prec: {b_data['precision']:.3f}\nFP: {int(b_data['fp'])} px | FN: {int(b_data['fn'])} px",
            fontsize=8,
        )
        axes[3].axis("off")

        im_i_prob = axes[4].imshow(i_data["probs"].numpy(), cmap="inferno", vmin=0.0, vmax=1.0)
        axes[4].set_title("Improved Prob Map (Sigmoid)\nPhase M1 Model (Tversky)", fontsize=8)
        axes[4].axis("off")

        axes[5].imshow(i_diff)
        fp_change = int(i_data["fp"]) - int(b_data["fp"])
        dice_change = (i_data["dice"] - b_data["dice"]) * 100.0
        axes[5].set_title(
            f"Improved Error Map\nDice: {i_data['dice']:.3f} ({'+' if dice_change>=0 else ''}{dice_change:.1f}%) | Prec: {i_data['precision']:.3f}\nFP: {int(i_data['fp'])} px ({'+' if fp_change>=0 else ''}{fp_change} px)",
            fontsize=8,
            color="darkgreen" if dice_change >= 0 else "darkred",
        )
        axes[5].axis("off")

        plt.tight_layout()
        safe_stem = Path(img_name).stem
        out_file = output_dir / f"comparison_{safe_stem}.png"
        plt.savefig(out_file, dpi=130)
        plt.close(fig)
        generated_files.append(str(out_file))

    return generated_files


def run_phase_m2_evaluation(
    baseline_checkpoint: Path | str = "artifacts/checkpoints/segmentation_real/best_unet_real.pt",
    improved_checkpoint: Path | str = "artifacts/checkpoints/segmentation_real/best_unet_improved.pt",
    output_comparison_json: Path | str = "reports/unet_test_comparison_phase_m2.json",
    output_comparison_csv: Path | str = "reports/unet_test_comparison_per_image.csv",
    output_improved_json: Path | str = "reports/unet_improved_test_evaluation.json",
    output_improved_csv: Path | str = "reports/unet_improved_test_per_image.csv",
    output_overlay_dir: Path | str = "reports/plantseg_audit/test_prediction_overlays_comparison",
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Execute complete Phase M2 comparative test evaluation protocol."""
    base_ckpt_p = Path(baseline_checkpoint)
    impr_ckpt_p = Path(improved_checkpoint)

    out_comp_json_p = Path(output_comparison_json)
    out_comp_csv_p = Path(output_comparison_csv)
    out_impr_json_p = Path(output_improved_json)
    out_impr_csv_p = Path(output_improved_csv)
    out_ovl_dir_p = Path(output_overlay_dir)

    out_comp_json_p.parent.mkdir(parents=True, exist_ok=True)
    out_ovl_dir_p.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE M2: IMPROVED U-NET HOLDOUT TEST EVALUATION & COMPARISON")
    print("=" * 80)
    print(f"Baseline Checkpoint: {base_ckpt_p}")
    print(f"Improved Checkpoint: {impr_ckpt_p}")
    print(f"Fixed Classification Threshold: {threshold} (strictly no test tuning)")
    print("=" * 80)

    # 1. Load Test Dataset
    test_ds = PlantSegTomatoLesionDataset(
        split="test",
        image_size=(256, 256),
        as_float_mask=True,
        preserve_aspect_ratio=True,
        augment=False,
    )
    if len(test_ds) != 99:
        raise ValueError(f"Expected exactly 99 test images, found {len(test_ds)}")
    print(f"Loaded untouched holdout test dataset: {len(test_ds)} images.\n")

    # 2. Select Device
    eval_device = get_device(preferred="mps")
    print(f"Evaluation inference device: {eval_device}")

    # 3. Load Checkpoints
    base_model, base_meta = load_unet_checkpoint(base_ckpt_p, eval_device)
    impr_model, impr_meta = load_unet_checkpoint(impr_ckpt_p, eval_device)
    print(f"Baseline Model loaded from Epoch {base_meta['epoch']} (Val Dice: {base_meta['val_dice']:.4f})")
    print(f"Improved Model loaded from Epoch {impr_meta['epoch']} (Val Dice: {impr_meta['val_dice']:.4f})\n")

    # 4. Evaluate Baseline Model
    print("Evaluating Baseline U-Net on 99 test images...")
    df_base, sum_base, preds_base = evaluate_model_on_dataset(
        base_model, test_ds, eval_device, threshold=threshold, model_tag="baseline"
    )

    # 5. Evaluate Improved Model
    print("Evaluating Improved U-Net on 99 test images...")
    df_impr, sum_impr, preds_impr = evaluate_model_on_dataset(
        impr_model, test_ds, eval_device, threshold=threshold, model_tag="improved"
    )

    # 6. Build Consolidated Per-Image Comparison DataFrame
    df_merged = pd.DataFrame()
    df_merged["image_name"] = df_base["image_name"]
    df_merged["disease"] = df_base["disease"]
    df_merged["gt_lesion_pixels"] = df_base["gt_lesion_pixels"]
    df_merged["gt_lesion_ratio"] = df_base["gt_lesion_ratio"]

    # Baseline metrics
    df_merged["baseline_dice"] = df_base["baseline_dice"]
    df_merged["baseline_iou"] = df_base["baseline_iou"]
    df_merged["baseline_precision"] = df_base["baseline_precision"]
    df_merged["baseline_recall"] = df_base["baseline_recall"]
    df_merged["baseline_pixel_acc"] = df_base["baseline_pixel_accuracy"]
    df_merged["baseline_tp"] = df_base["baseline_tp_pixels"]
    df_merged["baseline_fp"] = df_base["baseline_fp_pixels"]
    df_merged["baseline_fn"] = df_base["baseline_fn_pixels"]

    # Improved metrics
    df_merged["improved_dice"] = df_impr["improved_dice"]
    df_merged["improved_iou"] = df_impr["improved_iou"]
    df_merged["improved_precision"] = df_impr["improved_precision"]
    df_merged["improved_recall"] = df_impr["improved_recall"]
    df_merged["improved_pixel_acc"] = df_impr["improved_pixel_accuracy"]
    df_merged["improved_tp"] = df_impr["improved_tp_pixels"]
    df_merged["improved_fp"] = df_impr["improved_fp_pixels"]
    df_merged["improved_fn"] = df_impr["improved_fn_pixels"]

    # Deltas (Improved - Baseline)
    df_merged["delta_dice"] = (df_merged["improved_dice"] - df_merged["baseline_dice"]).round(4)
    df_merged["delta_iou"] = (df_merged["improved_iou"] - df_merged["baseline_iou"]).round(4)
    df_merged["delta_precision"] = (df_merged["improved_precision"] - df_merged["baseline_precision"]).round(4)
    df_merged["delta_recall"] = (df_merged["improved_recall"] - df_merged["baseline_recall"]).round(4)
    df_merged["delta_pixel_acc"] = (df_merged["improved_pixel_acc"] - df_merged["baseline_pixel_acc"]).round(4)
    df_merged["delta_fp_pixels"] = (df_merged["improved_fp"] - df_merged["baseline_fp"]).astype(int)

    df_merged.to_csv(out_comp_csv_p, index=False)
    print(f"Per-image comparison saved to: {out_comp_csv_p}")

    # Export individual improved per-image CSV
    df_impr_clean = df_impr.copy()
    # rename columns to match standard schema
    df_impr_clean = df_impr_clean.rename(columns={
        "improved_dice": "dice",
        "improved_iou": "iou",
        "improved_precision": "precision",
        "improved_recall": "recall",
        "improved_pixel_accuracy": "pixel_accuracy",
        "improved_tp_pixels": "tp_pixels",
        "improved_fp_pixels": "fp_pixels",
        "improved_fn_pixels": "fn_pixels",
        "improved_tn_pixels": "tn_pixels",
        "improved_pred_lesion_pixels": "pred_lesion_pixels",
        "improved_pred_lesion_ratio": "pred_lesion_ratio",
    })
    df_impr_clean.to_csv(out_impr_csv_p, index=False)
    print(f"Improved per-image metrics saved to: {out_impr_csv_p}")

    # 7. Disease-Level Breakdown
    disease_comparison = {}
    for disease, grp in df_merged.groupby("disease"):
        disease_comparison[disease] = {
            "sample_count": len(grp),
            "baseline": {
                "dice": round(float(grp["baseline_dice"].mean()), 4),
                "iou": round(float(grp["baseline_iou"].mean()), 4),
                "precision": round(float(grp["baseline_precision"].mean()), 4),
                "recall": round(float(grp["baseline_recall"].mean()), 4),
                "pixel_accuracy": round(float(grp["baseline_pixel_acc"].mean()), 4),
                "total_fp_pixels": int(grp["baseline_fp"].sum()),
            },
            "improved": {
                "dice": round(float(grp["improved_dice"].mean()), 4),
                "iou": round(float(grp["improved_iou"].mean()), 4),
                "precision": round(float(grp["improved_precision"].mean()), 4),
                "recall": round(float(grp["improved_recall"].mean()), 4),
                "pixel_accuracy": round(float(grp["improved_pixel_acc"].mean()), 4),
                "total_fp_pixels": int(grp["improved_fp"].sum()),
            },
            "absolute_changes": {
                "delta_dice": round(float(grp["improved_dice"].mean() - grp["baseline_dice"].mean()), 4),
                "delta_iou": round(float(grp["improved_iou"].mean() - grp["baseline_iou"].mean()), 4),
                "delta_precision": round(float(grp["improved_precision"].mean() - grp["baseline_precision"].mean()), 4),
                "delta_recall": round(float(grp["improved_recall"].mean() - grp["baseline_recall"].mean()), 4),
                "delta_pixel_accuracy": round(float(grp["improved_pixel_acc"].mean() - grp["baseline_pixel_acc"].mean()), 4),
                "fp_reduction_percent": round(
                    float((grp["baseline_fp"].sum() - grp["improved_fp"].sum()) / max(grp["baseline_fp"].sum(), 1) * 100.0), 2
                ),
            },
        }

    # 8. CPU Latency Benchmark (Under Identical Conditions)
    print("\nBenchmarking CPU Latency under identical single-thread CPU conditions...")
    base_cpu_bench = benchmark_cpu_latency(base_model, n_warmup=10, n_runs=99)
    impr_cpu_bench = benchmark_cpu_latency(impr_model, n_warmup=10, n_runs=99)
    print(f"  Baseline CPU Latency: {base_cpu_bench['mean_latency_ms']} ms ({base_cpu_bench['throughput_fps']} FPS)")
    print(f"  Improved CPU Latency: {impr_cpu_bench['mean_latency_ms']} ms ({impr_cpu_bench['throughput_fps']} FPS)")

    # 9. Multi-Panel Visual Overlays
    print("\nRendering representative 6-panel comparison overlays...")
    overlay_files = render_comparison_overlays(
        test_ds,
        preds_base,
        preds_impr,
        output_dir=out_ovl_dir_p,
        samples_per_disease=1,
    )
    print(f"Rendered {len(overlay_files)} comparison overlays in {out_ovl_dir_p}")

    # 10. Compile Results Summary
    base_macro = sum_base["macro_metrics"]
    impr_macro = sum_impr["macro_metrics"]
    base_micro = sum_base["micro_metrics"]
    impr_micro = sum_impr["micro_metrics"]
    base_conf = sum_base["confusion_pixels"]
    impr_conf = sum_impr["confusion_pixels"]

    abs_changes = {
        "macro_dice": round(impr_macro["macro_dice"] - base_macro["macro_dice"], 4),
        "macro_iou": round(impr_macro["macro_iou"] - base_macro["macro_iou"], 4),
        "macro_precision": round(impr_macro["macro_precision"] - base_macro["macro_precision"], 4),
        "macro_recall": round(impr_macro["macro_recall"] - base_macro["macro_recall"], 4),
        "macro_pixel_accuracy": round(impr_macro["macro_pixel_accuracy"] - base_macro["macro_pixel_accuracy"], 4),
        "micro_dice": round(impr_micro["micro_dice"] - base_micro["micro_dice"], 4),
        "micro_iou": round(impr_micro["micro_iou"] - base_micro["micro_iou"], 4),
        "fp_pixel_reduction_count": int(base_conf["total_fp"] - impr_conf["total_fp"]),
        "fp_pixel_reduction_percent": round(
            float((base_conf["total_fp"] - impr_conf["total_fp"]) / max(base_conf["total_fp"], 1) * 100.0), 2
        ),
    }

    comparison_results = {
        "evaluation_phase": "Phase M2 — Improved U-Net Holdout Test Split Evaluation",
        "test_dataset": "data/processed/tomato_lesions_plantseg/splits/test.csv",
        "test_sample_count": len(test_ds),
        "threshold": threshold,
        "models": {
            "baseline": {
                "checkpoint": str(base_ckpt_p),
                "epoch": base_meta["epoch"],
                "val_dice": base_meta["val_dice"],
                "macro_metrics": base_macro,
                "micro_metrics": base_micro,
                "confusion_pixels": base_conf,
                "cpu_latency": base_cpu_bench,
            },
            "improved": {
                "checkpoint": str(impr_ckpt_p),
                "epoch": impr_meta["epoch"],
                "val_dice": impr_meta["val_dice"],
                "macro_metrics": impr_macro,
                "micro_metrics": impr_micro,
                "confusion_pixels": impr_conf,
                "cpu_latency": impr_cpu_bench,
            },
        },
        "absolute_changes": abs_changes,
        "disease_level_comparison": disease_comparison,
        "overlays_directory": str(out_ovl_dir_p),
        "overlays_generated": len(overlay_files),
    }

    with open(out_comp_json_p, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Comparative evaluation summary exported to: {out_comp_json_p}")

    # Also export individual improved test evaluation json for standard schema
    improved_test_results = {
        "evaluation_phase": "Phase M2 — Improved U-Net Holdout Test Evaluation",
        "checkpoint_evaluated": str(impr_ckpt_p),
        "checkpoint_train_epoch": impr_meta["epoch"],
        "checkpoint_val_dice": impr_meta["val_dice"],
        "test_dataset": "PlantSeg v7 Cleaned Tomato Subset (data/processed/tomato_lesions_plantseg/)",
        "test_sample_count": len(test_ds),
        "fixed_classification_threshold": threshold,
        "resolution": [256, 256],
        "device": str(eval_device),
        "macro_metrics": {
            "mean_dice_score": impr_macro["macro_dice"],
            "mean_iou": impr_macro["macro_iou"],
            "mean_precision": impr_macro["macro_precision"],
            "mean_recall": impr_macro["macro_recall"],
            "mean_pixel_accuracy": impr_macro["macro_pixel_accuracy"],
        },
        "micro_metrics": {
            "micro_dice_score": impr_micro["micro_dice"],
            "micro_iou": impr_micro["micro_iou"],
            "micro_precision": impr_micro["micro_precision"],
            "micro_recall": impr_micro["micro_recall"],
            "micro_pixel_accuracy": impr_micro["micro_pixel_accuracy"],
        },
        "confusion_pixels": {
            "total_true_positives": impr_conf["total_tp"],
            "total_false_positives": impr_conf["total_fp"],
            "total_false_negatives": impr_conf["total_fn"],
            "total_true_negatives": impr_conf["total_tn"],
        },
        "per_disease_breakdown": {
            dis: {
                "test_sample_count": stats["sample_count"],
                "mean_dice": stats["improved"]["dice"],
                "mean_iou": stats["improved"]["iou"],
                "mean_precision": stats["improved"]["precision"],
                "mean_recall": stats["improved"]["recall"],
                "mean_pixel_accuracy": stats["improved"]["pixel_accuracy"],
            }
            for dis, stats in disease_comparison.items()
        },
        "latency_benchmarks": impr_cpu_bench,
        "overlays_directory": str(out_ovl_dir_p),
    }
    with open(out_impr_json_p, "w", encoding="utf-8") as f:
        json.dump(improved_test_results, f, indent=2)
    print(f"Improved model test results exported to: {out_impr_json_p}")

    # 11. Print Summary Table
    print("\n" + "=" * 80)
    print("PHASE M2: TEST SPLIT (99 IMAGES) COMPARATIVE RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<24} | {'Baseline (Phase K)':<18} | {'Improved (Phase M1)':<19} | {'Change':<12}")
    print("-" * 80)
    print(f"{'Macro Dice Score':<24} | {base_macro['macro_dice']*100:>17.2f}% | {impr_macro['macro_dice']*100:>18.2f}% | {abs_changes['macro_dice']*100:>+10.2f}%")
    print(f"{'Macro IoU (Jaccard)':<24} | {base_macro['macro_iou']*100:>17.2f}% | {impr_macro['macro_iou']*100:>18.2f}% | {abs_changes['macro_iou']*100:>+10.2f}%")
    print(f"{'Macro Precision':<24} | {base_macro['macro_precision']*100:>17.2f}% | {impr_macro['macro_precision']*100:>18.2f}% | {abs_changes['macro_precision']*100:>+10.2f}%")
    print(f"{'Macro Recall':<24} | {base_macro['macro_recall']*100:>17.2f}% | {impr_macro['macro_recall']*100:>18.2f}% | {abs_changes['macro_recall']*100:>+10.2f}%")
    print(f"{'Pixel Accuracy':<24} | {base_macro['macro_pixel_accuracy']*100:>17.2f}% | {impr_macro['macro_pixel_accuracy']*100:>18.2f}% | {abs_changes['macro_pixel_accuracy']*100:>+10.2f}%")
    print(f"{'Micro Dice Score':<24} | {base_micro['micro_dice']*100:>17.2f}% | {impr_micro['micro_dice']*100:>18.2f}% | {abs_changes['micro_dice']*100:>+10.2f}%")
    print(f"{'Micro IoU':<24} | {base_micro['micro_iou']*100:>17.2f}% | {impr_micro['micro_iou']*100:>18.2f}% | {abs_changes['micro_iou']*100:>+10.2f}%")
    print(f"{'False Positive Pixels':<24} | {base_conf['total_fp']:>18,d} | {impr_conf['total_fp']:>19,d} | {abs_changes['fp_pixel_reduction_percent']:>+9.2f}%")
    print(f"{'CPU Mean Latency':<24} | {base_cpu_bench['mean_latency_ms']:>15.2f} ms | {impr_cpu_bench['mean_latency_ms']:>16.2f} ms | {impr_cpu_bench['mean_latency_ms']-base_cpu_bench['mean_latency_ms']:>+9.2f} ms")
    print("=" * 80)

    print("\nDisease-wise Comparative Macro Dice:")
    for dis, stats in disease_comparison.items():
        b_d = stats["baseline"]["dice"] * 100.0
        i_d = stats["improved"]["dice"] * 100.0
        d_d = stats["absolute_changes"]["delta_dice"] * 100.0
        fp_red = stats["absolute_changes"]["fp_reduction_percent"]
        print(f"  - {dis.title():<30} | Base: {b_d:5.2f}% | Impr: {i_d:5.2f}% | Delta: {d_d:+5.2f}% | FP Reduction: {fp_red:+6.2f}%")
    print("=" * 80)

    return comparison_results


if __name__ == "__main__":
    run_phase_m2_evaluation()
