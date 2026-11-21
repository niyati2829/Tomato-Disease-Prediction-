"""Holdout Evaluation Engine for Pest Object Detection (REQ-08).

Evaluates the trained detector checkpoint on the untouched test split,
computing Precision, Recall, mAP@0.5, mAP@0.5:0.95, and inference latency.
Exports results to reports/pest_detection_evaluation.json.

Author: Crop Disease Detection Project Team
"""

import json
import os
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

from src.detection.dataset import PEST_CLASSES, PestDetectionDataset, generate_synthetic_pest_benchmark
from src.detection.metrics import evaluate_detections
from src.detection.model import TinyYOLODetector
from src.detection.train import collate_fn
from src.utils.device import get_device


def evaluate_pest_detector(
    checkpoint_path: Path | str = "artifacts/checkpoints/detection/best_pest_detector.pt",
    out_report_path: Path | str = "reports/pest_detection_evaluation.json",
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate detector on test split."""
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

    device = get_device(preferred="mps")
    checkpoint = torch.load(ckpt_p, map_location=device)

    model = TinyYOLODetector(
        in_channels=3,
        num_classes=checkpoint.get("num_classes", 3),
        grid_size=checkpoint.get("grid_size", 8),
        num_boxes_per_cell=checkpoint.get("num_boxes_per_cell", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    splits = generate_synthetic_pest_benchmark(n_samples=50, image_size=256, seed=seed)
    test_imgs, test_anns = splits["test"]

    test_ds = PestDetectionDataset(test_imgs, test_anns, image_size=(256, 256))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    all_preds = []
    all_gts = []
    latencies_ms = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)

            start_t = time.perf_counter()
            preds = model.predict_boxes(images, conf_threshold=0.20, iou_threshold=0.5)
            end_t = time.perf_counter()

            latencies_ms.append((end_t - start_t) * 1000.0)
            all_preds.extend(preds)
            all_gts.extend(targets)

    metrics = evaluate_detections(all_preds, all_gts, num_classes=3)
    avg_latency = float(np.mean(latencies_ms))

    results = {
        "checkpoint_evaluated": str(ckpt_p),
        "test_images": len(test_ds),
        "device": str(device),
        "average_inference_latency_ms": avg_latency,
        "mAP_0.5": metrics["mAP_0.5"],
        "mAP_0.5:0.95": metrics["mAP_0.5:0.95"],
        "per_class_metrics": {
            PEST_CLASSES[c]: {
                "AP_0.5": metrics["per_class_AP_0.5"][c],
                "Precision_0.5": metrics["per_class_Precision_0.5"][c],
                "Recall_0.5": metrics["per_class_Recall_0.5"][c],
            }
            for c in range(3)
        },
    }

    report_p = Path(out_report_path)
    report_p.parent.mkdir(parents=True, exist_ok=True)
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n================ Pest Detection Test Evaluation ================")
    print(f"Mean Average Precision (mAP@0.5):      {results['mAP_0.5']:.4f}")
    print(f"COCO Mean Average Precision (mAP@[.5:.95]): {results['mAP_0.5:0.95']:.4f}")
    print(f"Average Inference Latency:              {results['average_inference_latency_ms']:.2f} ms")
    print("\nPer-Class Breakdown:")
    for pest_name, m in results["per_class_metrics"].items():
        print(f"  • {pest_name:18s} | AP@0.5: {m['AP_0.5']:.4f} | Prec: {m['Precision_0.5']:.4f} | Rec: {m['Recall_0.5']:.4f}")
    print(f"\nResults saved to: {report_p}")

    return results


if __name__ == "__main__":
    evaluate_pest_detector()
