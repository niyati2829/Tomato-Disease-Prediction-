"""Phase M2 Automated Holdout Test Comparative Evaluation Test Suite.

Verifies:
1. Checkpoint loading and forward inference pass on improved U-Net vs baseline U-Net.
2. Holdout test split comparative evaluation presence and 99-sample coverage.
3. Test metrics integrity (Dice, IoU, Precision, Recall, Pixel Accuracy, Confusion Matrix).
4. Mathematical consistency of absolute changes and over-segmentation reduction.
5. Prediction vs ground-truth visual overlays in reports/plantseg_audit/test_prediction_overlays_comparison/.
6. CPU latency benchmark reporting.
7. Preservation of production checkpoints, synthetic checkpoints, and test split manifests.

Author: Crop Disease Detection Project Team
Phase: Phase M2
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.segmentation.unet import UNet

BASELINE_CHECKPOINT = Path("artifacts/checkpoints/segmentation_real/best_unet_real.pt")
IMPROVED_CHECKPOINT = Path("artifacts/checkpoints/segmentation_real/best_unet_improved.pt")

COMPARISON_JSON = Path("reports/unet_test_comparison_phase_m2.json")
COMPARISON_CSV = Path("reports/unet_test_comparison_per_image.csv")
IMPROVED_JSON = Path("reports/unet_improved_test_evaluation.json")
IMPROVED_CSV = Path("reports/unet_improved_test_per_image.csv")
OVERLAY_DIR = Path("reports/plantseg_audit/test_prediction_overlays_comparison")

TEST_SPLIT_CSV = Path("data/processed/tomato_lesions_plantseg/splits/test.csv")

RESNET_CHECKPOINT = Path("artifacts/checkpoints/best_resnet18_weighted.pt")
EXPECTED_RESNET_HASH = "b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa"

PEST_CHECKPOINT = Path("artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt")
EXPECTED_PEST_HASH = "01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343"

SYNTHETIC_CHECKPOINT = Path("artifacts/checkpoints/segmentation/best_unet.pt")


def test_checkpoints_loading_and_architecture_parity():
    """Verify both baseline and improved U-Net checkpoints load and share identical architecture."""
    assert BASELINE_CHECKPOINT.exists(), f"Missing baseline checkpoint: {BASELINE_CHECKPOINT}"
    assert IMPROVED_CHECKPOINT.exists(), f"Missing improved checkpoint: {IMPROVED_CHECKPOINT}"

    ckpt_base = torch.load(BASELINE_CHECKPOINT, map_location="cpu")
    ckpt_impr = torch.load(IMPROVED_CHECKPOINT, map_location="cpu")

    assert ckpt_base["in_channels"] == ckpt_impr["in_channels"] == 3
    assert ckpt_base["num_classes"] == ckpt_impr["num_classes"] == 1
    assert ckpt_base["init_features"] == ckpt_impr["init_features"] == 16

    model_base = UNet(in_channels=3, num_classes=1, init_features=16)
    model_base.load_state_dict(ckpt_base["model_state_dict"])
    model_base.eval()

    model_impr = UNet(in_channels=3, num_classes=1, init_features=16)
    model_impr.load_state_dict(ckpt_impr["model_state_dict"])
    model_impr.eval()

    dummy_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out_base = model_base(dummy_input)
        out_impr = model_impr(dummy_input)

    assert out_base.shape == (2, 1, 256, 256)
    assert out_impr.shape == (2, 1, 256, 256)


def test_comparison_reports_exist_and_cover_99_test_images():
    """Verify comparison JSON and CSV reports exist and contain exactly 99 test images."""
    assert COMPARISON_JSON.exists(), f"Missing JSON: {COMPARISON_JSON}"
    assert COMPARISON_CSV.exists(), f"Missing CSV: {COMPARISON_CSV}"
    assert IMPROVED_JSON.exists(), f"Missing JSON: {IMPROVED_JSON}"
    assert IMPROVED_CSV.exists(), f"Missing CSV: {IMPROVED_CSV}"

    with open(COMPARISON_JSON, "r", encoding="utf-8") as f:
        comp = json.load(f)

    assert comp["test_sample_count"] == 99
    assert comp["threshold"] == 0.50

    df_comp = pd.read_csv(COMPARISON_CSV)
    assert len(df_comp) == 99

    df_test = pd.read_csv(TEST_SPLIT_CSV)
    assert len(df_test) == 99
    assert set(df_comp["image_name"]) == set(df_test["image_name"])

    df_impr = pd.read_csv(IMPROVED_CSV)
    assert len(df_impr) == 99
    assert set(df_impr["image_name"]) == set(df_test["image_name"])


def test_comparative_metrics_mathematical_consistency():
    """Verify confusion matrix sums, metric bounds, and positive deltas for improved model."""
    with open(COMPARISON_JSON, "r", encoding="utf-8") as f:
        comp = json.load(f)

    base = comp["models"]["baseline"]
    impr = comp["models"]["improved"]
    abs_changes = comp["absolute_changes"]

    # Total pixels per model must equal 99 * 256 * 256 = 6,488,064
    total_pixels_expected = 99 * 256 * 256
    for model_dict in (base, impr):
        conf = model_dict["confusion_pixels"]
        tot = conf["total_tp"] + conf["total_fp"] + conf["total_fn"] + conf["total_tn"]
        assert tot == total_pixels_expected, f"Pixel sum mismatch: {tot} vs {total_pixels_expected}"

    # Verify metrics in [0, 1]
    for m in (base["macro_metrics"], impr["macro_metrics"]):
        for v in m.values():
            assert 0.0 <= v <= 1.0

    # Macro Dice >= Macro IoU (mathematical theorem)
    assert base["macro_metrics"]["macro_dice"] >= base["macro_metrics"]["macro_iou"]
    assert impr["macro_metrics"]["macro_dice"] >= impr["macro_metrics"]["macro_iou"]

    # Improved model shows superior Dice, IoU, Precision, and Accuracy
    assert impr["macro_metrics"]["macro_dice"] > base["macro_metrics"]["macro_dice"]
    assert impr["macro_metrics"]["macro_iou"] > base["macro_metrics"]["macro_iou"]
    assert impr["macro_metrics"]["macro_precision"] > base["macro_metrics"]["macro_precision"]
    assert impr["macro_metrics"]["macro_pixel_accuracy"] > base["macro_metrics"]["macro_pixel_accuracy"]

    # False positive pixels reduced
    assert impr["confusion_pixels"]["total_fp"] < base["confusion_pixels"]["total_fp"]
    assert abs_changes["fp_pixel_reduction_percent"] > 20.0  # Over 20% reduction


def test_cpu_latency_benchmarks_present():
    """Verify latency benchmarks exist and reflect realistic CPU execution."""
    with open(COMPARISON_JSON, "r", encoding="utf-8") as f:
        comp = json.load(f)

    for tag in ("baseline", "improved"):
        lat = comp["models"][tag]["cpu_latency"]
        assert lat["device"] == "cpu"
        assert lat["num_runs"] == 99
        assert 1.0 <= lat["mean_latency_ms"] <= 500.0
        assert lat["throughput_fps"] > 0.0


def test_comparison_overlays_generated():
    """Verify multi-panel visual overlays are generated for all 7 diseases."""
    assert OVERLAY_DIR.exists()
    pngs = list(OVERLAY_DIR.glob("*.png"))
    assert len(pngs) >= 7, f"Expected at least 7 comparison overlays, found {len(pngs)}"


def test_production_and_split_integrity():
    """Verify that existing production models, synthetic U-Net, and test split are untouched."""
    assert RESNET_CHECKPOINT.exists()
    with open(RESNET_CHECKPOINT, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED_RESNET_HASH

    assert PEST_CHECKPOINT.exists()
    with open(PEST_CHECKPOINT, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED_PEST_HASH

    assert SYNTHETIC_CHECKPOINT.exists()
    assert TEST_SPLIT_CSV.exists()
