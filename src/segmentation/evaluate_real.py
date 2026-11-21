"""Phase L: U-Net Holdout Test Split Evaluation Engine.

Evaluates the trained real U-Net model on the untouched 99-image PlantSeg test split.

Disciplines:
- Evaluates strictly on data/processed/tomato_lesions_plantseg/splits/test.csv (99 images).
- Fixed threshold of 0.5 (strictly no threshold tuning on test data).
- Same preprocessing as training (256x256, aspect ratio padding, ImageNet normalization).
- Calculates per-image and overall metrics: Dice, IoU, Precision, Recall, Pixel Accuracy.
- Generates representative prediction vs ground-truth overlays in reports/plantseg_audit/test_prediction_overlays/.
- Exports summary metrics to reports/unet_real_test_evaluation.json and reports/unet_real_test_per_image.csv.

Author: Crop Disease Detection Project Team
Phase: Phase L
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
from PIL import Image
from torch.utils.data import DataLoader

from src.segmentation.plantseg_dataset import PlantSegTomatoLesionDataset
from src.segmentation.unet import UNet
from src.utils.device import get_device, get_device_info


def evaluate_real_unet_holdout(
    checkpoint_path: Path | str = "artifacts/checkpoints/segmentation_real/best_unet_real.pt",
    output_json: Path | str = "reports/unet_real_test_evaluation.json",
    output_csv: Path | str = "reports/unet_real_test_per_image.csv",
    overlay_dir: Path | str = "reports/plantseg_audit/test_prediction_overlays",
    threshold: float = 0.5,
    n_overlays: int = 7,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate real U-Net checkpoint on untouched 99-image holdout test split."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_p}")

    out_json_p = Path(output_json)
    out_csv_p = Path(output_csv)
    ovl_dir_p = Path(overlay_dir)

    out_json_p.parent.mkdir(parents=True, exist_ok=True)
    out_csv_p.parent.mkdir(parents=True, exist_ok=True)
    ovl_dir_p.mkdir(parents=True, exist_ok=True)

    device = get_device(preferred="mps")
    device_info = get_device_info(preferred="mps")

    print("=" * 70)
    print("PHASE L: REAL U-NET HOLDOUT TEST SPLIT EVALUATION")
    print("=" * 70)
    print(f"Loading checkpoint: {ckpt_p}")
    print(f"Device: {device} ({device_info.get('device_name', 'Host')})")
    print(f"Fixed binary threshold: {threshold} (No tuning on test data)")
    print("=" * 70)

    # 1. Load Checkpoint
    checkpoint = torch.load(ckpt_p, map_location=device)
    init_features = checkpoint.get("init_features", 16)
    in_channels = checkpoint.get("in_channels", 3)
    num_classes = checkpoint.get("num_classes", 1)
    best_train_epoch = checkpoint.get("epoch", "N/A")
    val_dice_at_ckpt = checkpoint.get("val_dice", "N/A")

    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        init_features=init_features,
        bilinear=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    print(f"Model Architecture: UNet(in={in_channels}, out={num_classes}, init_features={init_features})")
    print(f"Checkpoint Origin: Epoch {best_train_epoch} (Validation Dice: {val_dice_at_ckpt:.4f})")

    # 2. Load Holdout Test Dataset
    test_ds = PlantSegTomatoLesionDataset(
        split="test",
        image_size=(256, 256),
        as_float_mask=True,
        preserve_aspect_ratio=True,
    )
    print(f"Loaded Test Split: {len(test_ds)} images (Holdout data)\n")

    if len(test_ds) != 99:
        raise ValueError(f"Expected exactly 99 test images, found {len(test_ds)}")

    # 3. Evaluation Loop
    smooth = 1e-5
    per_image_records = []
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    inference_latencies = []

    # Store representative sample items for visual overlay generation
    overlay_candidates = {}

    with torch.no_grad():
        for i in range(len(test_ds)):
            img_tensor, mask_tensor = test_ds[i]
            meta = test_ds.get_metadata(i)

            img_input = img_tensor.unsqueeze(0).to(device)
            target = mask_tensor.unsqueeze(0).to(device)

            t0 = time.time()
            logits = model(img_input)
            probs = torch.sigmoid(logits)
            t_infer = time.time() - t0
            inference_latencies.append(t_infer * 1000.0)  # ms

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

            per_image_records.append({
                "image_name": meta["image_name"],
                "disease": meta["disease"],
                "dice": round(dice, 4),
                "iou": round(iou, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "pixel_accuracy": round(accuracy, 4),
                "tp_pixels": int(tp),
                "fp_pixels": int(fp),
                "fn_pixels": int(fn),
                "tn_pixels": int(tn),
                "gt_lesion_pixels": int(tp + fn),
                "pred_lesion_pixels": int(tp + fp),
                "gt_lesion_ratio": round(float((tp + fn) / target.numel()), 4),
                "pred_lesion_ratio": round(float((tp + fp) / target.numel()), 4),
            })

            # Pick 1 sample per disease for overlay visualization
            disease_name = meta["disease"]
            if disease_name not in overlay_candidates:
                overlay_candidates[disease_name] = {
                    "image_tensor": img_tensor.cpu(),
                    "mask_tensor": mask_tensor.cpu(),
                    "prob_map": probs[0, 0].cpu(),
                    "pred_mask": preds[0, 0].cpu(),
                    "meta": meta,
                    "dice": dice,
                    "iou": iou,
                }

    # 4. Metric Aggregations
    df_per_image = pd.DataFrame(per_image_records)
    df_per_image.to_csv(out_csv_p, index=False)
    print(f"Per-image evaluation metrics saved to: {out_csv_p}")

    # Macro averages (average of per-image scores)
    macro_dice = float(df_per_image["dice"].mean())
    macro_iou = float(df_per_image["iou"].mean())
    macro_precision = float(df_per_image["precision"].mean())
    macro_recall = float(df_per_image["recall"].mean())
    macro_pixel_accuracy = float(df_per_image["pixel_accuracy"].mean())

    # Micro averages (dataset-level pooled pixel sums)
    micro_dice = float((2.0 * total_tp + smooth) / (2.0 * total_tp + total_fp + total_fn + smooth))
    micro_iou = float((total_tp + smooth) / (total_tp + total_fp + total_fn + smooth))
    micro_precision = float((total_tp + smooth) / (total_tp + total_fp + smooth))
    micro_recall = float((total_tp + smooth) / (total_tp + total_fn + smooth))
    micro_pixel_accuracy = float((total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn))

    # Per-disease breakdown
    per_disease = {}
    for disease, grp in df_per_image.groupby("disease"):
        per_disease[disease] = {
            "test_sample_count": len(grp),
            "mean_dice": round(float(grp["dice"].mean()), 4),
            "mean_iou": round(float(grp["iou"].mean()), 4),
            "mean_precision": round(float(grp["precision"].mean()), 4),
            "mean_recall": round(float(grp["recall"].mean()), 4),
            "mean_pixel_accuracy": round(float(grp["pixel_accuracy"].mean()), 4),
        }

    # Latency statistics
    avg_latency_ms = float(np.mean(inference_latencies))
    p95_latency_ms = float(np.percentile(inference_latencies, 95))

    results = {
        "evaluation_phase": "Phase L — Holdout Test Evaluation",
        "checkpoint_evaluated": str(ckpt_p),
        "checkpoint_train_epoch": best_train_epoch,
        "checkpoint_val_dice": round(val_dice_at_ckpt, 4) if isinstance(val_dice_at_ckpt, float) else val_dice_at_ckpt,
        "test_dataset": "PlantSeg v7 Cleaned Tomato Subset (data/processed/tomato_lesions_plantseg/)",
        "test_sample_count": len(test_ds),
        "fixed_classification_threshold": threshold,
        "resolution": [256, 256],
        "device": str(device),
        "macro_metrics": {
            "mean_dice_score": round(macro_dice, 4),
            "mean_iou": round(macro_iou, 4),
            "mean_precision": round(macro_precision, 4),
            "mean_recall": round(macro_recall, 4),
            "mean_pixel_accuracy": round(macro_pixel_accuracy, 4),
        },
        "micro_metrics": {
            "micro_dice_score": round(micro_dice, 4),
            "micro_iou": round(micro_iou, 4),
            "micro_precision": round(micro_precision, 4),
            "micro_recall": round(micro_recall, 4),
            "micro_pixel_accuracy": round(micro_pixel_accuracy, 4),
        },
        "confusion_pixels": {
            "total_true_positives": int(total_tp),
            "total_false_positives": int(total_fp),
            "total_false_negatives": int(total_fn),
            "total_true_negatives": int(total_tn),
        },
        "per_disease_breakdown": per_disease,
        "latency_benchmarks": {
            "device": str(device),
            "mean_latency_ms": round(avg_latency_ms, 2),
            "p95_latency_ms": round(p95_latency_ms, 2),
            "throughput_fps": round(1000.0 / max(avg_latency_ms, 1e-3), 1),
        },
        "overlays_generated_count": len(overlay_candidates),
        "overlays_directory": str(ovl_dir_p),
    }

    with open(out_json_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Summary test evaluation results exported to: {out_json_p}")

    # 5. Generate Visual Overlays
    generate_prediction_overlays(overlay_candidates, ovl_dir_p)

    # 6. Print Report Summary
    print("\n" + "=" * 70)
    print("PHASE L: HOLDOUT TEST EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Test Samples:       {len(test_ds)} images (Holdout Split)")
    print(f"Macro Dice Score:   {macro_dice:.4f} ({macro_dice*100:.2f}%)")
    print(f"Micro Dice Score:   {micro_dice:.4f} ({micro_dice*100:.2f}%)")
    print(f"Macro IoU (Jaccard):{macro_iou:.4f} ({macro_iou*100:.2f}%)")
    print(f"Macro Precision:    {macro_precision:.4f} ({macro_precision*100:.2f}%)")
    print(f"Macro Recall:       {macro_recall:.4f} ({macro_recall*100:.2f}%)")
    print(f"Pixel Accuracy:     {macro_pixel_accuracy*100:.2f}%")
    print(f"Inference Latency:  {avg_latency_ms:.2f} ms/image ({results['latency_benchmarks']['throughput_fps']} FPS on {device})")
    print("-" * 70)
    print("Disease-wise Performance Breakdown:")
    for dis, stats in per_disease.items():
        print(f"  - {dis.title():<32} | Dice: {stats['mean_dice']:.4f} | IoU: {stats['mean_iou']:.4f} | Recall: {stats['mean_recall']:.4f}")
    print("=" * 70)

    return results


def generate_prediction_overlays(candidates: Dict[str, Dict], out_dir: Path) -> None:
    """Generate 4-panel visual verification overlays for holdout test predictions.

    Panels:
    1. Input RGB Image (denormalized ImageNet)
    2. Ground-Truth Lesion Mask
    3. Predicted Probability Heatmap (Sigmoid activation)
    4. Categorical Error Mask:
       - Green: True Positive (Correct detection)
       - Red: False Positive (Over-segmentation)
       - Blue: False Negative (Missed lesion)
    """
    print(f"\nGenerating {len(candidates)} visual test prediction overlays in: {out_dir}")

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    for disease, data in candidates.items():
        img_t = data["image_tensor"].permute(1, 2, 0).numpy()
        # Denormalize RGB image
        img_rgb = np.clip((img_t * std + mean) * 255.0, 0, 255).astype(np.uint8)

        gt_mask = data["mask_tensor"][0].numpy() > 0.5
        prob_map = data["prob_map"].numpy()
        pred_mask = data["pred_mask"].numpy() > 0.5

        # Confusion map (H, W, 3)
        # Background: Dimmed grayscale of image
        gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        diff_overlay = np.stack([gray * 0.4] * 3, axis=-1).astype(np.uint8)

        # True Positives: Green
        tp_mask = pred_mask & gt_mask
        diff_overlay[tp_mask] = [40, 220, 60]

        # False Positives: Red
        fp_mask = pred_mask & (~gt_mask)
        diff_overlay[fp_mask] = [230, 40, 40]

        # False Negatives: Blue
        fn_mask = (~pred_mask) & gt_mask
        diff_overlay[fn_mask] = [40, 100, 240]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Test Image (256x256)\n{data['meta']['image_name']}", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Ground-Truth Mask\nLesion Ratio: {gt_mask.mean():.3f}", fontsize=9)
        axes[1].axis("off")

        im_prob = axes[2].imshow(prob_map, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[2].set_title(f"U-Net Prob Map (Sigmoid)\nDice: {data['dice']:.3f} | IoU: {data['iou']:.3f}", fontsize=9)
        axes[2].axis("off")
        plt.colorbar(im_prob, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].imshow(diff_overlay)
        axes[3].set_title("Overlap Analysis\nGreen=TP, Red=FP, Blue=FN", fontsize=9)
        axes[3].axis("off")

        plt.tight_layout()
        safe_name = Path(data["meta"]["image_name"]).stem
        out_path = out_dir / f"test_eval_{safe_name}.png"
        plt.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"  Overlay rendered: {out_path.name}")


if __name__ == "__main__":
    evaluate_real_unet_holdout()
