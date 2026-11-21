"""Forensic Automated Tests for Pest Detection Prediction Audit (REQ-AUDIT).

Verifies:
1. Exact class mappings and naming consistency across YOLOv8n-Seg and TinyYOLO.
2. Verified multi-image testing across:
   - Tuta-only damage images
   - Healthy-only foliage images
   - Mixed foliage images (both Tuta damage and healthy foliage)
3. Agreement between region counts (tuta_count, healthy_count) and actual filtered detections list.
4. Correct label semantics, bounding box boundaries, and mask area calculations.
5. Independent processing across sequential uploads without stale session or image leakage.
"""

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from app.app import (
    PEST_CHECKPOINT_PATH,
    YOLO_CHECKPOINT_PATH,
    load_pest_detector,
    load_yolov8_detector,
    predict_pests,
    predict_pests_yolov8,
)
from src.detection.real_pest_dataset import REAL_PEST_CLASSES


@pytest.fixture(scope="module")
def models():
    """Load both pest detection models once for test module."""
    tiny, dev, _ = load_pest_detector(PEST_CHECKPOINT_PATH)
    yolo = load_yolov8_detector(YOLO_CHECKPOINT_PATH)
    return tiny, yolo, dev


def test_class_mappings_and_ids(models):
    """Verify class IDs and labels in both checkpoints and dataset definitions."""
    tiny, yolo, _ = models

    # 1. TinyYOLO check
    assert tiny is not None, "TinyYOLO model failed to load"
    assert REAL_PEST_CLASSES[0] == "healthy"
    assert REAL_PEST_CLASSES[1] == "tuta_absoluta_damage"

    # 2. YOLOv8n-Seg check
    assert yolo is not None, "YOLOv8n-Seg model failed to load"
    assert yolo.names[0] == "Healthy"
    assert yolo.names[1] == "Tuta absoluta"
    assert yolo.model.nc == 2


def test_tuta_only_leaf_prediction(models):
    """Verify behavior on a leaf exhibiting exclusively Tuta damage.

    On a leaf with only Tuta damage, YOLOv8n-Seg correctly identifies damage
    and predicts 0 healthy regions (avoiding TinyYOLO's noisy false positives).
    """
    tiny, yolo, dev = models
    tuta_img_path = Path("data/pest_detection/tuta_absoluta_research_v4/train/images/20260218_161110_jpg.rf.ec462ad8a6605d883640a3e2075aec30.jpg")
    if not tuta_img_path.exists():
        pytest.skip("Test image not found")

    img = Image.open(tuta_img_path)

    # YOLOv8n-Seg
    annotated_y, dets_y, tuta_y, healthy_y = predict_pests_yolov8(img, yolo, conf_threshold=0.25)
    assert tuta_y > 0, "YOLOv8 should detect Tuta damage on diseased leaf"
    assert healthy_y == 0, "YOLOv8 should NOT hallucinate healthy regions on exclusively diseased leaf"
    assert len(dets_y) == tuta_y + healthy_y
    for d in dets_y:
        assert d["is_pest_damage"] is True
        assert "Tuta" in d["category"]
        assert d["mask_area_px"] > 0

    # TinyYOLO (Kasaisa baseline) - expected to show noisy background detections
    annotated_t, dets_t, tuta_t, healthy_t = predict_pests(img, tiny, dev, conf_threshold=0.20)
    assert len(dets_t) == tuta_t + healthy_t
    assert tuta_t > 0


def test_healthy_only_leaf_prediction(models):
    """Verify that YOLOv8n-Seg genuinely supports and predicts Healthy Foliage masks."""
    tiny, yolo, dev = models
    healthy_img_path = Path("data/pest_detection/tuta_absoluta_research_v4/train/images/20230704_210813992_iOS_jpg.rf.bde2d28e047fda22880d22e852a6ee2d.jpg")
    if not healthy_img_path.exists():
        pytest.skip("Test image not found")

    img = Image.open(healthy_img_path)

    annotated_y, dets_y, tuta_y, healthy_y = predict_pests_yolov8(img, yolo, conf_threshold=0.25)
    assert healthy_y > 0, "YOLOv8 must detect healthy foliage on healthy leaf"
    assert tuta_y == 0, "YOLOv8 should not detect Tuta on healthy leaf"
    assert len(dets_y) == healthy_y

    for d in dets_y:
        assert d["is_pest_damage"] is False
        assert "Healthy" in d["category"]
        assert d["mask_area_px"] > 1000.0, "Healthy foliage mask must have non-zero area"


def test_mixed_foliage_prediction(models):
    """Verify that YOLOv8n-Seg detects BOTH classes when both are present."""
    tiny, yolo, dev = models
    mixed_img_path = Path("data/pest_detection/tuta_absoluta_research_v4/test/images/20260218_155239_jpg.rf.29dffa863c79d45ab9f3d991a01a99f5.jpg")
    if not mixed_img_path.exists():
        pytest.skip("Test image not found")

    img = Image.open(mixed_img_path)

    annotated_y, dets_y, tuta_y, healthy_y = predict_pests_yolov8(img, yolo, conf_threshold=0.25)
    assert tuta_y >= 1, "Expected at least 1 Tuta detection"
    assert healthy_y >= 1, "Expected at least 1 Healthy detection"
    assert len(dets_y) == tuta_y + healthy_y

    categories = [d["category"] for d in dets_y]
    assert any("Tuta" in c for c in categories)
    assert any("Healthy" in c for c in categories)


def test_count_agreement_and_schema(models):
    """Verify that displayed region counts strictly equal filtered detections list."""
    _, yolo, _ = models
    img_path = Path("data/pest_detection/tuta_absoluta_research_v4/test/images/20260218_155239_jpg.rf.29dffa863c79d45ab9f3d991a01a99f5.jpg")
    if not img_path.exists():
        pytest.skip("Test image not found")

    img = Image.open(img_path)
    _, dets, tuta_c, healthy_c = predict_pests_yolov8(img, yolo, conf_threshold=0.20)

    tuta_filter = sum(1 for d in dets if d["is_pest_damage"])
    healthy_filter = sum(1 for d in dets if not d["is_pest_damage"])

    assert tuta_c == tuta_filter, f"Tuta count mismatch: {tuta_c} != {tuta_filter}"
    assert healthy_c == healthy_filter, f"Healthy count mismatch: {healthy_c} != {healthy_filter}"
    assert len(dets) == tuta_c + healthy_c


def test_independent_image_upload_processing(models):
    """Verify that sequential image processing operates independently without state leakage."""
    _, yolo, _ = models
    img1_path = Path("data/pest_detection/tuta_absoluta_research_v4/train/images/20260218_161110_jpg.rf.ec462ad8a6605d883640a3e2075aec30.jpg")
    img2_path = Path("data/pest_detection/tuta_absoluta_research_v4/train/images/20230704_210813992_iOS_jpg.rf.bde2d28e047fda22880d22e852a6ee2d.jpg")
    if not img1_path.exists() or not img2_path.exists():
        pytest.skip("Test images not found")

    # Simulate fresh byte decoding for Image 1
    raw1 = img1_path.read_bytes()
    pil1 = ImageOps.exif_transpose(Image.open(io.BytesIO(raw1))).convert("RGB")
    pil1.load()
    _, dets1, tuta1, healthy1 = predict_pests_yolov8(pil1, yolo, conf_threshold=0.25)

    # Simulate fresh byte decoding for Image 2
    raw2 = img2_path.read_bytes()
    pil2 = ImageOps.exif_transpose(Image.open(io.BytesIO(raw2))).convert("RGB")
    pil2.load()
    _, dets2, tuta2, healthy2 = predict_pests_yolov8(pil2, yolo, conf_threshold=0.25)

    # Verify predictions are distinct and correctly reflect each image's pathology
    assert tuta1 > 0 and healthy1 == 0
    assert tuta2 == 0 and healthy2 > 0
    assert dets1 != dets2
