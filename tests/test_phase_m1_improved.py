"""Phase M1 Test Suite: Verification of Improved Real U-Net Checkpoint & Artifacts.

Verifies:
1. Loading and inference pass for the improved real U-Net checkpoint.
2. Improved model architecture configuration (in=3, out=1, init_features=16).
3. Presence and validity of validation prediction overlays.
4. Training history and configuration JSON files.
5. Preservation of Phase K baseline checkpoint (best_unet_real.pt) and test split.

Author: Crop Disease Detection Project Team
Phase: Phase M1
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from src.segmentation.unet import UNet

IMPROVED_CHECKPOINT = Path("artifacts/checkpoints/segmentation_real/best_unet_improved.pt")
BASELINE_CHECKPOINT = Path("artifacts/checkpoints/segmentation_real/best_unet_real.pt")
HISTORY_JSON = Path("reports/unet_improved_training_history.json")
CONFIG_JSON = Path("reports/unet_improved_training_config.json")
VAL_OVERLAY_DIR = Path("reports/plantseg_audit/val_prediction_overlays_improved")
TEST_SPLIT_CSV = Path("data/processed/tomato_lesions_plantseg/splits/test.csv")


def test_improved_checkpoint_loading_and_inference():
    """Verify improved checkpoint can be loaded and outputs probability map."""
    assert IMPROVED_CHECKPOINT.exists(), f"Missing improved checkpoint: {IMPROVED_CHECKPOINT}"

    ckpt = torch.load(IMPROVED_CHECKPOINT, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "val_dice" in ckpt
    assert "val_precision" in ckpt
    assert ckpt["in_channels"] == 3
    assert ckpt["num_classes"] == 1
    assert ckpt["init_features"] == 16

    model = UNet(in_channels=3, num_classes=1, init_features=16, bilinear=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.sigmoid(logits)

    assert logits.shape == (2, 1, 256, 256)
    assert probs.min().item() >= 0.0
    assert probs.max().item() <= 1.0


def test_improved_training_artifacts():
    """Verify training history and configuration files exist with expected structure."""
    assert HISTORY_JSON.exists(), f"Missing history JSON: {HISTORY_JSON}"
    assert CONFIG_JSON.exists(), f"Missing config JSON: {CONFIG_JSON}"

    with open(CONFIG_JSON, "r", encoding="utf-8") as f:
        conf = json.load(f)

    assert conf["model_architecture"] == "UNet"
    assert conf["test_split_touched"] is False
    assert conf["best_val_dice"] >= 0.55
    assert conf["best_val_precision"] >= 0.50

    with open(HISTORY_JSON, "r", encoding="utf-8") as f:
        hist = json.load(f)

    assert len(hist["train_loss"]) == conf["completed_epochs"]
    assert len(hist["val_dice"]) == conf["completed_epochs"]


def test_validation_overlays_exist():
    """Verify representative validation prediction overlays were generated."""
    assert VAL_OVERLAY_DIR.exists(), f"Missing overlay directory: {VAL_OVERLAY_DIR}"
    overlays = list(VAL_OVERLAY_DIR.glob("*.png"))
    assert len(overlays) == 7

    for ovl in overlays:
        with Image.open(ovl) as im:
            assert im.size[0] > 1000
            assert im.size[1] > 200


def test_baseline_and_test_split_preservation():
    """Verify Phase K baseline checkpoint and test split are untouched."""
    assert BASELINE_CHECKPOINT.exists()
    assert BASELINE_CHECKPOINT.stat().st_size > 0

    assert TEST_SPLIT_CSV.exists()
    df_test = pd.read_csv(TEST_SPLIT_CSV)
    assert len(df_test) == 99
