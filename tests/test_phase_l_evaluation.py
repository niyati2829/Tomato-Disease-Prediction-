"""Phase L Automated Holdout Test Evaluation Test Suite.

Verifies:
1. Checkpoint loading and forward inference pass on real U-Net.
2. Holdout test split evaluation presence and 99-sample coverage.
3. Test metrics integrity (Dice, IoU, Precision, Recall, Pixel Accuracy).
4. Prediction vs ground-truth visual overlays in reports/plantseg_audit/test_prediction_overlays/.
5. Preservation of production checkpoints, synthetic checkpoints, and test split manifests.

Author: Crop Disease Detection Project Team
Phase: Phase L
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.segmentation.plantseg_dataset import PlantSegTomatoLesionDataset
from src.segmentation.unet import UNet

CHECKPOINT_PATH = Path("artifacts/checkpoints/segmentation_real/best_unet_real.pt")
REPORT_JSON = Path("reports/unet_real_test_evaluation.json")
REPORT_CSV = Path("reports/unet_real_test_per_image.csv")
OVERLAY_DIR = Path("reports/plantseg_audit/test_prediction_overlays")
TEST_SPLIT_CSV = Path("data/processed/tomato_lesions_plantseg/splits/test.csv")

RESNET_CHECKPOINT = Path("artifacts/checkpoints/best_resnet18_weighted.pt")
EXPECTED_RESNET_HASH = "b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa"

PEST_CHECKPOINT = Path("artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt")
EXPECTED_PEST_HASH = "01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343"

SYNTHETIC_CHECKPOINT = Path("artifacts/checkpoints/segmentation/best_unet.pt")


def test_checkpoint_loading_and_inference():
    """Verify real U-Net checkpoint can be loaded and executed for inference."""
    assert CHECKPOINT_PATH.exists(), f"Missing checkpoint: {CHECKPOINT_PATH}"

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "epoch" in ckpt
    assert "val_dice" in ckpt
    assert ckpt["in_channels"] == 3
    assert ckpt["num_classes"] == 1

    model = UNet(in_channels=3, num_classes=1, init_features=ckpt["init_features"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.sigmoid(logits)

    assert logits.shape == (1, 1, 256, 256)
    assert probs.min().item() >= 0.0
    assert probs.max().item() <= 1.0


def test_test_evaluation_reports_exist_and_match():
    """Verify JSON summary and per-image CSV reports exist and have 99 samples."""
    assert REPORT_JSON.exists(), f"Missing JSON report: {REPORT_JSON}"
    assert REPORT_CSV.exists(), f"Missing CSV report: {REPORT_CSV}"

    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        res = json.load(f)

    assert res["test_sample_count"] == 99
    assert res["fixed_classification_threshold"] == 0.5
    assert "macro_metrics" in res
    assert "micro_metrics" in res
    assert "per_disease_breakdown" in res

    df_csv = pd.read_csv(REPORT_CSV)
    assert len(df_csv) == 99

    df_test = pd.read_csv(TEST_SPLIT_CSV)
    assert set(df_csv["image_name"]) == set(df_test["image_name"])


def test_test_metric_ranges_and_values():
    """Verify macro and micro metrics are mathematically sound."""
    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        res = json.load(f)

    macro = res["macro_metrics"]
    micro = res["micro_metrics"]

    # All metrics must be in [0, 1]
    for m in (macro["mean_dice_score"], macro["mean_iou"], macro["mean_precision"], macro["mean_recall"], macro["mean_pixel_accuracy"]):
        assert 0.0 <= m <= 1.0

    for m in (micro["micro_dice_score"], micro["micro_iou"], micro["micro_precision"], micro["micro_recall"], micro["micro_pixel_accuracy"]):
        assert 0.0 <= m <= 1.0

    # Sanity checks on known achieved scores
    assert macro["mean_dice_score"] >= 0.40
    assert micro["micro_dice_score"] >= 0.45
    assert macro["mean_recall"] >= 0.70
    assert macro["mean_pixel_accuracy"] >= 0.80


def test_per_disease_breakdown_covers_all_seven():
    """Verify per-disease breakdown covers all 7 tomato foliar pathologies."""
    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        res = json.load(f)

    per_dis = res["per_disease_breakdown"]
    assert len(per_dis) == 7

    total_samples = sum(item["test_sample_count"] for item in per_dis.values())
    assert total_samples == 99


def test_visual_prediction_overlays_exist():
    """Verify representative prediction vs ground-truth overlays were generated."""
    assert OVERLAY_DIR.exists(), f"Missing overlay directory: {OVERLAY_DIR}"
    overlays = list(OVERLAY_DIR.glob("*.png"))
    assert len(overlays) >= 7

    for ovl in overlays:
        with Image.open(ovl) as im:
            assert im.size[0] > 1000  # Multi-panel figure
            assert im.size[1] > 200


def test_non_interference_preservation():
    """Verify existing production checkpoints, synthetic checkpoint, and test manifest are unmodified."""
    assert RESNET_CHECKPOINT.exists()
    assert hashlib.sha256(RESNET_CHECKPOINT.read_bytes()).hexdigest() == EXPECTED_RESNET_HASH

    assert PEST_CHECKPOINT.exists()
    assert hashlib.sha256(PEST_CHECKPOINT.read_bytes()).hexdigest() == EXPECTED_PEST_HASH

    assert SYNTHETIC_CHECKPOINT.exists()
    assert SYNTHETIC_CHECKPOINT.stat().st_size == 40303563

    assert TEST_SPLIT_CSV.exists()
    assert len(pd.read_csv(TEST_SPLIT_CSV)) == 99
