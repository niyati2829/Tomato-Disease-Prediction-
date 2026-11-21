"""Phase M3-B: YOLOv8n-Seg Holdout Test Evaluation Engine.

Evaluates the fine-tuned YOLOv8n-seg model on the untouched 92-image test split
of TutaAbsolutaResearch v4, as well as cross-evaluating on the 12-image Kasaisa test split.

Outputs:
- Box and Mask Precision, Recall, mAP@0.5, mAP@0.5:0.95 (overall & per-class).
- Confusion matrix and detection counts.
- CPU inference latency benchmark and throughput (FPS).
- Multi-panel visual prediction overlays with predicted masks and bounding boxes.
- Exports reports/pest_detection_yolov8_evaluation.json.

Author: Crop Disease Detection Project Team
Phase: Phase M3-B
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "pest_detection_yolov8" / "best_yolov8n_seg.pt"
DATA_YAML_V4 = PROJECT_ROOT / "data" / "pest_detection" / "tuta_absoluta_research_v4" / "data.yaml"
DATA_YAML_KASAISA = PROJECT_ROOT / "data" / "pest_detection" / "real_tuta_absoluta" / "data.yaml"

OUTPUT_JSON = PROJECT_ROOT / "reports" / "pest_detection_yolov8_evaluation.json"
OVERLAY_DIR = PROJECT_ROOT / "reports" / "pest_detection_audit" / "test_prediction_overlays_yolov8"


def evaluate_yolov8_seg() -> Dict[str, Any]:
    """Execute evaluation protocol for YOLOv8n-seg."""
    print("=" * 80)
    print("PHASE M3-B: YOLOV8N-SEG HOLDOUT TEST EVALUATION")
    print("=" * 80)
    print(f"Checkpoint under test: {CHECKPOINT_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    model = YOLO(str(CHECKPOINT_PATH))
    num_params = sum(p.numel() for p in model.model.parameters())
    model_size_mb = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)

    print(f"Model Architecture: YOLOv8n-seg ({num_params:,} parameters, {model_size_mb:.2f} MB)")

    # 2. Benchmark CPU Latency deterministically
    print("\nBenchmarking CPU inference latency under deterministic conditions...")
    cpu_latencies = []
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        _ = model.predict(dummy, device="cpu", verbose=False)

    test_imgs_dir = PROJECT_ROOT / "data" / "pest_detection" / "tuta_absoluta_research_v4" / "test" / "images"
    test_img_paths = sorted(list(test_imgs_dir.glob("*.jpg")) + list(test_imgs_dir.glob("*.jpeg")) + list(test_imgs_dir.glob("*.png")))

    for p in test_img_paths:
        t0 = time.perf_counter()
        _ = model.predict(str(p), device="cpu", verbose=False, imgsz=640)
        t1 = time.perf_counter()
        cpu_latencies.append((t1 - t0) * 1000.0)

    cpu_latencies = np.array(cpu_latencies)
    mean_lat = float(np.mean(cpu_latencies))
    p50_lat = float(np.median(cpu_latencies))
    p95_lat = float(np.percentile(cpu_latencies, 95))
    fps = float(1000.0 / max(mean_lat, 1e-3))
    print(f"  CPU Latency: Mean {mean_lat:.2f} ms | Median {p50_lat:.2f} ms | 95th {p95_lat:.2f} ms | Throughput: {fps:.1f} FPS")

    # 3. Validate on Version 4 holdout test split (92 images)
    print("\nRunning full validation on 92 holdout test images (TutaAbsolutaResearch v4)...")
    val_v4 = model.val(
        data=str(DATA_YAML_V4),
        split="test",
        imgsz=640,
        device="cpu",
        save_json=True,
        verbose=True,
    )

    # Extract metrics from val_v4
    box_p = float(val_v4.box.p[0]) if len(val_v4.box.p) > 0 else float(val_v4.box.mp)
    box_r = float(val_v4.box.r[0]) if len(val_v4.box.r) > 0 else float(val_v4.box.mr)
    box_map50 = float(val_v4.box.map50)
    box_map50_95 = float(val_v4.box.map)

    mask_p = float(val_v4.seg.p[0]) if len(val_v4.seg.p) > 0 else float(val_v4.seg.mp)
    mask_r = float(val_v4.seg.r[0]) if len(val_v4.seg.r) > 0 else float(val_v4.seg.mr)
    mask_map50 = float(val_v4.seg.map50)
    mask_map50_95 = float(val_v4.seg.map)

    # Per-class metrics
    class_names = ["Healthy", "Tuta absoluta"]
    per_class_metrics = {}
    for i, cname in enumerate(class_names):
        try:
            per_class_metrics[cname] = {
                "box_precision": round(float(val_v4.box.p[i]), 4),
                "box_recall": round(float(val_v4.box.r[i]), 4),
                "box_map50": round(float(val_v4.box.maps[i, 0] if hasattr(val_v4.box, "maps") and val_v4.box.maps.ndim > 1 else val_v4.box.ap50[i]), 4),
                "box_map50_95": round(float(val_v4.box.maps[i, :].mean() if hasattr(val_v4.box, "maps") and val_v4.box.maps.ndim > 1 else val_v4.box.ap[i]), 4),
                "mask_precision": round(float(val_v4.seg.p[i]), 4),
                "mask_recall": round(float(val_v4.seg.r[i]), 4),
                "mask_map50": round(float(val_v4.seg.maps[i, 0] if hasattr(val_v4.seg, "maps") and val_v4.seg.maps.ndim > 1 else val_v4.seg.ap50[i]), 4),
                "mask_map50_95": round(float(val_v4.seg.maps[i, :].mean() if hasattr(val_v4.seg, "maps") and val_v4.seg.maps.ndim > 1 else val_v4.seg.ap[i]), 4),
            }
        except Exception as e:
            per_class_metrics[cname] = {"error": str(e)}

    # 4. Optional Cross-Evaluation on Kasaisa test split (12 images) for direct comparison
    kasaisa_eval_results = {}
    if DATA_YAML_KASAISA.exists():
        print("\nCross-evaluating YOLOv8n-seg on Kasaisa 12-image test split for baseline comparison...")
        try:
            val_kas = model.val(
                data=str(DATA_YAML_KASAISA),
                split="test",
                imgsz=640,
                device="cpu",
                verbose=False,
            )
            kasaisa_eval_results = {
                "test_images": 12,
                "box_map50": round(float(val_kas.box.map50), 4),
                "box_map50_95": round(float(val_kas.box.map), 4),
                "box_precision": round(float(val_kas.box.mp), 4),
                "box_recall": round(float(val_kas.box.mr), 4),
            }
            print(f"  Kasaisa Cross-Eval: Box mAP@0.5 = {kasaisa_eval_results['box_map50']*100:.2f}% (vs TinyYOLO 30.44%)")
        except Exception as e:
            kasaisa_eval_results = {"note": f"Could not cross-evaluate on Kasaisa: {e}"}

    # 5. Generate Visual Prediction Overlays
    print("\nRendering representative test prediction overlays with instance segmentation masks...")
    sample_test_images = test_img_paths[:8]
    rendered_overlays = []

    for img_p in sample_test_images:
        preds = model.predict(str(img_p), conf=0.25, iou=0.5, device="cpu", verbose=False)[0]

        im_bgr = preds.orig_img
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        h, w = im_rgb.shape[:2]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(im_rgb)

        # Draw masks if available
        if preds.masks is not None and len(preds.masks) > 0:
            for mask_poly, cls_id, conf in zip(preds.masks.xy, preds.boxes.cls, preds.boxes.conf):
                cid = int(cls_id.item())
                cf = float(conf.item())
                color = "deepskyblue" if cid == 0 else "crimson"
                cname = "Healthy" if cid == 0 else "Tuta Damage"

                if len(mask_poly) > 2:
                    patch = plt.Polygon(mask_poly, fill=True, facecolor=color, alpha=0.4, edgecolor=color, linewidth=2.0)
                    ax.add_patch(patch)
                    ax.text(
                        mask_poly[0, 0], max(mask_poly[0, 1] - 6, 12),
                        f"{cname} {cf:.2f}",
                        bbox=dict(facecolor=color, alpha=0.85, pad=2, edgecolor="none"),
                        fontsize=8,
                        color="white",
                        weight="bold",
                    )
        elif preds.boxes is not None and len(preds.boxes) > 0:
            # Fallback to bounding boxes
            for box, cls_id, conf in zip(preds.boxes.xyxy, preds.boxes.cls, preds.boxes.conf):
                cid = int(cls_id.item())
                cf = float(conf.item())
                color = "deepskyblue" if cid == 0 else "crimson"
                cname = "Healthy" if cid == 0 else "Tuta Damage"
                x1, y1, x2, y2 = box.tolist()
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2.0)
                ax.add_patch(rect)
                ax.text(
                    x1, max(y1 - 6, 12),
                    f"{cname} {cf:.2f}",
                    bbox=dict(facecolor=color, alpha=0.85, pad=2, edgecolor="none"),
                    fontsize=8,
                    color="white",
                    weight="bold",
                )

        n_tuta = int((preds.boxes.cls == 1).sum().item()) if preds.boxes is not None and len(preds.boxes) > 0 else 0
        n_healthy = int((preds.boxes.cls == 0).sum().item()) if preds.boxes is not None and len(preds.boxes) > 0 else 0

        ax.set_title(f"[TEST PREDICTION] {img_p.name[:32]}...\nDetections: {n_tuta} Tuta Damage | {n_healthy} Healthy", fontsize=9)
        ax.axis("off")
        plt.tight_layout()

        safe_name = f"pred_eval_{img_p.stem[:24]}.png"
        out_f = OVERLAY_DIR / safe_name
        plt.savefig(out_f, dpi=130)
        plt.close(fig)
        rendered_overlays.append(str(out_f))
        print(f"  Saved overlay: {safe_name}")

    # 6. Compile Consolidated Results Summary
    tinyyolo_baseline = {
        "architecture": "Custom TinyYOLODetector (8x8 Grid, 1 anchor/cell)",
        "test_dataset": "TomatoEbola Kasaisa Farm (12 test images, 29 boxes)",
        "box_map50": 0.3044,
        "box_map50_95": 0.0879,
        "tuta_damage_precision": 0.0784,
        "tuta_damage_recall": 0.2353,
        "tuta_damage_ap50": 0.0866,
        "cpu_latency_ms": 10.27,
        "parameters": 726_949,
        "checkpoint_size_mb": 2.77,
    }

    results = {
        "evaluation_phase": "Phase M3-B — YOLOv8n-Seg Holdout Test Evaluation",
        "model_architecture": "YOLOv8n-seg (Ultralytics)",
        "checkpoint_path": str(CHECKPOINT_PATH),
        "parameters_count": int(num_params),
        "checkpoint_size_mb": round(model_size_mb, 2),
        "test_dataset": "TutaAbsolutaResearch v4 (92 holdout images, 151 polygon instances)",
        "test_sample_count": len(test_img_paths),
        "overall_box_metrics": {
            "mean_precision": round(box_p, 4),
            "mean_recall": round(box_r, 4),
            "map50": round(box_map50, 4),
            "map50_95": round(box_map50_95, 4),
        },
        "overall_mask_metrics": {
            "mean_precision": round(mask_p, 4),
            "mean_recall": round(mask_r, 4),
            "map50": round(mask_map50, 4),
            "map50_95": round(mask_map50_95, 4),
        },
        "per_class_metrics": per_class_metrics,
        "cpu_latency_benchmark": {
            "device": "CPU",
            "samples_timed": len(cpu_latencies),
            "mean_latency_ms": round(mean_lat, 2),
            "median_latency_ms": round(p50_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "throughput_fps": round(fps, 1),
        },
        "cross_evaluation_kasaisa_baseline": kasaisa_eval_results,
        "tinyyolo_comparison": {
            "baseline": tinyyolo_baseline,
            "yolov8n_seg": {
                "tuta_damage_box_map50": per_class_metrics.get("Tuta absoluta", {}).get("box_map50", "N/A"),
                "tuta_damage_box_precision": per_class_metrics.get("Tuta absoluta", {}).get("box_precision", "N/A"),
                "tuta_damage_box_recall": per_class_metrics.get("Tuta absoluta", {}).get("box_recall", "N/A"),
                "tuta_damage_mask_map50": per_class_metrics.get("Tuta absoluta", {}).get("mask_map50", "N/A"),
            },
            "comparability_disclosure": (
                "Direct comparison notes: TinyYOLO baseline was evaluated on the 12-image Kasaisa farm holdout test split "
                "with bounding boxes. YOLOv8n-seg was evaluated on the 92-image TutaAbsolutaResearch v4 holdout test split "
                "with instance segmentation polygons. On cross-evaluation against Kasaisa, YOLOv8n-seg achieves "
                f"{kasaisa_eval_results.get('box_map50', 'N/A')} mAP@0.5 compared to TinyYOLO's 30.44% mAP@0.5."
            ),
        },
        "overlays_directory": str(OVERLAY_DIR),
        "overlays_count": len(rendered_overlays),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation summary exported to: {OUTPUT_JSON}")

    # Print Report Summary
    print("\n" + "=" * 80)
    print("PHASE M3-B: TEST EVALUATION SUMMARY RESULTS")
    print("=" * 80)
    print(f"Test Split:           92 images (TutaAbsolutaResearch v4)")
    print(f"Box mAP@0.5:          {box_map50*100:.2f}%")
    print(f"Box mAP@0.5:0.95:     {box_map50_95*100:.2f}%")
    print(f"Mask mAP@0.5:         {mask_map50*100:.2f}%")
    print(f"Mask mAP@0.5:0.95:    {mask_map50_95*100:.2f}%")
    print(f"Box Mean Precision:   {box_p*100:.2f}%")
    print(f"Box Mean Recall:      {box_r*100:.2f}%")
    print(f"CPU Mean Latency:     {mean_lat:.2f} ms ({fps:.1f} FPS)")
    print("-" * 80)
    print("Per-Class Breakdown:")
    for cname, m in per_class_metrics.items():
        if "error" not in m:
            print(f"  - {cname:<16} | Box mAP50: {m['box_map50']*100:.2f}% | Mask mAP50: {m['mask_map50']*100:.2f}% | Prec: {m['box_precision']*100:.2f}% | Rec: {m['box_recall']*100:.2f}%")
    print("=" * 80)

    return results


if __name__ == "__main__":
    evaluate_yolov8_seg()
