"""Phase M3-A Automated Unit Test Suite for Tuta absoluta Dataset & Annotation Audit.

Verifies:
1. TomatoEbola Kasaisa dataset split coverage (98 train, 12 valid, 12 test = 122 images).
2. Annotation file parity (122 images, 122 labels, 0 orphaned files).
3. Class distribution (319 total boxes: 138 class 0, 181 class 1).
4. Bounding box validity (0 degenerate, 0 overflow).
5. Image uniqueness and zero cross-split data leakage.
6. EDEN Library verification status.
7. Production checkpoints and baseline detector integrity.

Author: Crop Disease Detection Project Team
Phase: Phase M3-A
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch

DATASET_ROOT = Path("data/pest_detection/real_tuta_absoluta")
AUDIT_JSON = Path("reports/tuta_dataset_audit_phase_m3a.json")
OVERLAY_DIR = Path("reports/pest_detection_audit")

PEST_CHECKPOINT = Path("artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt")
EXPECTED_PEST_HASH = "01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343"

RESNET_CHECKPOINT = Path("artifacts/checkpoints/best_resnet18_weighted.pt")
EXPECTED_RESNET_HASH = "b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa"


def test_tuta_dataset_split_counts_and_parity():
    """Verify that the dataset contains exactly 122 images and 122 label files across splits."""
    assert DATASET_ROOT.exists(), f"Dataset directory missing: {DATASET_ROOT}"

    expected_splits = {"train": 98, "valid": 12, "test": 12}
    total_imgs = 0
    total_lbls = 0

    for split, exp_count in expected_splits.items():
        img_dir = DATASET_ROOT / split / "images"
        lbl_dir = DATASET_ROOT / split / "labels"

        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
        lbls = list(lbl_dir.glob("*.txt"))

        assert len(imgs) == exp_count, f"{split} images count mismatch: {len(imgs)} vs {exp_count}"
        assert len(lbls) == exp_count, f"{split} labels count mismatch: {len(lbls)} vs {exp_count}"

        img_stems = {f.stem for f in imgs}
        lbl_stems = {f.stem for f in lbls}
        assert img_stems == lbl_stems, f"Orphaned files detected in split {split}"

        total_imgs += len(imgs)
        total_lbls += len(lbls)

    assert total_imgs == 122
    assert total_lbls == 122


def test_tuta_annotations_validity_and_classes():
    """Verify bounding boxes have valid normalized bounds and expected classes (0 and 1)."""
    assert AUDIT_JSON.exists(), f"Audit report missing: {AUDIT_JSON}"

    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["total_images"] == 122
    assert audit["total_boxes"] == 319
    assert audit["integrity"]["total_degenerate_boxes"] == 0
    assert audit["integrity"]["total_overflow_boxes"] == 0
    assert audit["integrity"]["intra_split_duplicates_count"] == 0
    assert audit["integrity"]["cross_split_leakage_count"] == 0

    box_stats = audit["box_statistics"]
    assert box_stats["class_0_healthy_count"] == 138
    assert box_stats["class_1_tuta_count"] == 181
    assert box_stats["class_0_healthy_count"] + box_stats["class_1_tuta_count"] == 319


def test_eden_library_and_other_farms_audit_status():
    """Verify that EDEN Library absence and upstream farm restrictions are properly recorded."""
    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["eden_library_investigation"]["local_files_present"] is False
    assert "restricted" in audit["other_farm_subsets_status"]["dikumari_farm"].lower()
    assert "restricted" in audit["other_farm_subsets_status"]["kukareta_farm"].lower()


def test_inspection_overlays_exist():
    """Verify visual inspection overlays have been rendered."""
    assert OVERLAY_DIR.exists()
    overlays = list(OVERLAY_DIR.glob("*.png"))
    assert len(overlays) >= 6, f"Expected at least 6 inspection overlays, found {len(overlays)}"


def test_checkpoint_integrity_preserved():
    """Verify that baseline pest detector and ResNet18 checkpoints are bitwise untouched."""
    assert PEST_CHECKPOINT.exists()
    with open(PEST_CHECKPOINT, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED_PEST_HASH

    assert RESNET_CHECKPOINT.exists()
    with open(RESNET_CHECKPOINT, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED_RESNET_HASH
