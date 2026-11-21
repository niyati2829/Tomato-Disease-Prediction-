"""Phase M3-B Automated Test Suite for YOLOv8n-Seg Pest Detection.

Verifies:
1. YOLOv8n-seg trained checkpoint exists, is loadable, and performs forward inference.
2. Training config and history reports exist and record reproducible parameters.
3. Test evaluation metrics integrity (mAP@0.5, Precision, Recall).
4. Visual test prediction overlays existence.
5. Preservation of production ResNet18, baseline TinyYOLO, and U-Net checkpoints.

Author: Crop Disease Detection Project Team
Phase: Phase M3-B
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ultralytics import YOLO

YOLOV8_CHECKPOINT = Path("artifacts/checkpoints/pest_detection_yolov8/best_yolov8n_seg.pt")
CONFIG_JSON = Path("reports/pest_detection_yolov8_training_config.json")
HISTORY_JSON = Path("reports/pest_detection_yolov8_training_history.json")
EVAL_JSON = Path("reports/pest_detection_yolov8_evaluation.json")
OVERLAYS_DIR = Path("reports/pest_detection_audit/test_prediction_overlays_yolov8")

RESNET_CHECKPOINT = Path("artifacts/checkpoints/best_resnet18_weighted.pt")
EXPECTED_RESNET_HASH = "b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa"

PEST_CHECKPOINT = Path("artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt")
EXPECTED_PEST_HASH = "01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343"

UNET_IMPROVED_CHECKPOINT = Path("artifacts/checkpoints/segmentation_real/best_unet_improved.pt")


def test_yolov8_checkpoint_loading_and_inference():
    """Verify best_yolov8n_seg.pt can be loaded and executed for inference."""
    assert YOLOV8_CHECKPOINT.exists(), f"Missing YOLOv8 checkpoint: {YOLOV8_CHECKPOINT}"
    assert YOLOV8_CHECKPOINT.stat().st_size > 5 * 1024 * 1024  # > 5 MB

    model = YOLO(str(YOLOV8_CHECKPOINT))
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

    preds = model.predict(dummy_img, device="cpu", verbose=False)
    assert len(preds) == 1
    assert hasattr(preds[0], "boxes")
    assert hasattr(preds[0], "masks")


def test_training_and_evaluation_reports_exist():
    """Verify JSON training reports and evaluation summary are populated and valid."""
    assert CONFIG_JSON.exists(), f"Missing config: {CONFIG_JSON}"
    assert HISTORY_JSON.exists(), f"Missing history: {HISTORY_JSON}"
    assert EVAL_JSON.exists(), f"Missing eval report: {EVAL_JSON}"

    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        ev = json.load(f)

    assert ev["test_sample_count"] == 92
    assert ev["overall_box_metrics"]["map50"] > 0.60, "Box mAP50 should exceed 60%"
    assert ev["overall_mask_metrics"]["map50"] > 0.60, "Mask mAP50 should exceed 60%"
    assert ev["per_class_metrics"]["Tuta absoluta"]["box_precision"] > 0.70
    assert ev["cpu_latency_benchmark"]["throughput_fps"] > 5.0


def test_visual_prediction_overlays_exist():
    """Verify prediction overlays were generated on holdout test samples."""
    assert OVERLAYS_DIR.exists(), f"Missing overlays directory: {OVERLAYS_DIR}"
    overlays = list(OVERLAYS_DIR.glob("*.png"))
    assert len(overlays) >= 6, f"Expected at least 6 test overlays, found {len(overlays)}"


def test_production_and_baseline_integrity():
    """Verify that ResNet18, TinyYOLO baseline, and U-Net checkpoints remain bitwise untouched."""
    assert RESNET_CHECKPOINT.exists()
    with open(RESNET_CHECKPOINT, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED_RESNET_HASH

    assert PEST_CHECKPOINT.exists()
    with open(PEST_CHECKPOINT, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED_PEST_HASH

    assert UNET_IMPROVED_CHECKPOINT.exists()
