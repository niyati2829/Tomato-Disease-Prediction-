"""Automated Unit Tests for Real Tomato Pest Detection (Phase G - REQ-G2).

Verifies:
1. RealTomatoPestDataset loading and sample integrity.
2. YOLO coordinate parser bounds validation (0 <= xmin < xmax <= W, 0 <= ymin < ymax <= H).
3. TinyYOLODetector forward pass and output shape on 2-class setup.
4. NMS-filtered predict_boxes output schema.
5. Real pest detector checkpoint loadability and parameter integrity.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.detection.model import TinyYOLODetector
from src.detection.real_pest_dataset import (
    REAL_PEST_CLASSES,
    RealTomatoPestDataset,
    parse_real_yolo_labels,
)


@pytest.fixture
def sample_dataset():
    """Fixture providing initialized RealTomatoPestDataset."""
    root = Path("data/pest_detection/real_tuta_absoluta")
    if not root.exists():
        pytest.skip("Real pest dataset directory not present")
    return RealTomatoPestDataset(split="test", dataset_root=root, image_size=(256, 256))


def test_real_pest_dataset_structure(sample_dataset):
    """Test that dataset contains samples and returns valid tensor types."""
    assert len(sample_dataset) > 0, "Dataset should have at least one sample"

    img_tensor, target = sample_dataset[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 256, 256)
    assert "boxes" in target
    assert "labels" in target
    assert "image_id" in target


def test_parse_real_yolo_labels_bounds(tmp_path):
    """Test YOLO label parsing ensures non-negative coordinates bounded by image size."""
    label_file = tmp_path / "sample.txt"
    # Format: class cx cy w h
    label_file.write_text("1 0.5 0.5 0.2 0.4\n0 0.8 0.8 0.3 0.3\n", encoding="utf-8")

    boxes, labels = parse_real_yolo_labels(label_file, image_width=640, image_height=480)
    assert boxes.shape == (2, 4)
    assert labels.shape == (2,)

    # Check bounds
    for b in boxes:
        xmin, ymin, xmax, ymax = b.tolist()
        assert 0.0 <= xmin < xmax <= 640.0
        assert 0.0 <= ymin < ymax <= 480.0


def test_tiny_yolo_real_pest_forward_shape():
    """Test TinyYOLODetector output tensor dimensions for num_classes=2."""
    model = TinyYOLODetector(in_channels=3, num_classes=2, grid_size=8, num_boxes_per_cell=1)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    # Shape: (B, S, S, B*5 + C) = (2, 8, 8, 1*5 + 2) = (2, 8, 8, 7)
    assert out.shape == (2, 8, 8, 7)


def test_tiny_yolo_predict_boxes_schema():
    """Test predict_boxes returns expected dict schema."""
    model = TinyYOLODetector(in_channels=3, num_classes=2, grid_size=8, num_boxes_per_cell=1)
    x = torch.randn(2, 3, 256, 256)
    preds = model.predict_boxes(x, conf_threshold=0.1, iou_threshold=0.5)

    assert len(preds) == 2
    for p in preds:
        assert "boxes" in p
        assert "scores" in p
        assert "labels" in p
        assert p["boxes"].ndim == 2
        assert p["scores"].ndim == 1
        assert p["labels"].ndim == 1


def test_real_pest_detector_checkpoint():
    """Verify saved checkpoint exists and contains valid weights."""
    ckpt_path = Path("artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt")
    if not ckpt_path.exists():
        pytest.skip("Real pest detector checkpoint has not been trained yet")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "val_mAP_0.5" in ckpt
    assert ckpt["num_classes"] == 2
    assert ckpt["val_mAP_0.5"] > 0.0, "Validation mAP should be positive"


def test_baseline_pest_detector_checkpoint_hash_invariance():
    """Verify that baseline pest detector checkpoint hash is strictly preserved."""
    import hashlib
    ckpt_path = Path("artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt")
    if not ckpt_path.exists():
        pytest.skip("Checkpoint not found")

    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    expected_hash = "01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343"
    assert hasher.hexdigest() == expected_hash, "Baseline pest detector checkpoint hash altered!"


def test_kasaisa_dataset_integrity_and_box_validity():
    """Verify Kasaisa dataset has exactly 122 images, 319 boxes, and zero coordinate overflows."""
    root = Path("data/pest_detection/real_tuta_absoluta")
    if not root.exists():
        pytest.skip("Kasaisa dataset directory not present")

    total_images = len(list(root.rglob("images/*.jpg")))
    label_files = list(root.rglob("labels/*.txt"))
    assert total_images == 122, f"Expected 122 Kasaisa images, got {total_images}"
    assert len(label_files) == 122, f"Expected 122 Kasaisa label files, got {len(label_files)}"

    total_boxes = 0
    for lf in label_files:
        lines = lf.read_text().strip().split("\n")
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            c = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            assert c in (0, 1), f"Unexpected class index {c} in {lf.name}"
            assert 0.0 <= cx <= 1.0, f"cx out of bounds in {lf.name}"
            assert 0.0 <= cy <= 1.0, f"cy out of bounds in {lf.name}"
            assert 0.0 < w <= 1.0, f"width out of bounds in {lf.name}"
            assert 0.0 < h <= 1.0, f"height out of bounds in {lf.name}"
            total_boxes += 1

    assert total_boxes == 319, f"Expected exactly 319 bounding boxes, got {total_boxes}"

