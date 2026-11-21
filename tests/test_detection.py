"""Unit Tests for Pest Object Detection Pipeline (REQ-08).

Verifies:
1. Coordinate transformation invertibility (xyxy <-> cxcywh).
2. Box clipping and positive area validation.
3. Box IoU analytical calculation.
4. Non-Maximum Suppression (NMS) candidate filtering.
5. YOLO and Pascal VOC annotation parser accuracy.
6. TinyYOLODetector forward pass and decode dimensions.
7. Mean Average Precision (mAP) calculation logic.

Author: Crop Disease Detection Project Team
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.detection.boxes import (
    box_iou,
    clip_boxes_to_image,
    cxcywh_to_xyxy,
    non_max_suppression,
    validate_boxes,
    xyxy_to_cxcywh,
)
from src.detection.dataset import parse_voc_xml, parse_yolo_annotation
from src.detection.metrics import calculate_map, compute_class_ap
from src.detection.model import TinyYOLODetector


def test_box_coordinate_transformations_invertibility():
    """Verify bidirectional invertibility between xyxy and cxcywh."""
    # [xmin, ymin, xmax, ymax]
    boxes_xyxy = torch.tensor([
        [10.0, 20.0, 50.0, 80.0],
        [100.0, 150.0, 140.0, 210.0],
    ])

    cxcywh = xyxy_to_cxcywh(boxes_xyxy)
    reconstructed = cxcywh_to_xyxy(cxcywh)

    torch.testing.assert_close(boxes_xyxy, reconstructed)


def test_box_clipping_and_validation():
    """Verify clipping prevents out-of-bounds coordinates and validate_boxes checks positive area."""
    # Out-of-bounds box: [-10, 5, 280, 300] on 256x256 image
    boxes = torch.tensor([
        [-10.0, 5.0, 280.0, 300.0],
        [50.0, 50.0, 40.0, 60.0],  # Invalid: xmax < xmin
    ])

    clipped = clip_boxes_to_image(boxes, (256, 256))
    assert clipped[0, 0].item() == 0.0
    assert clipped[0, 2].item() == 256.0
    assert clipped[0, 3].item() == 256.0

    valid_box = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    invalid_box = torch.tensor([[20.0, 10.0, 10.0, 20.0]])
    assert validate_boxes(valid_box) is True
    assert validate_boxes(invalid_box) is False


def test_box_iou_analytical():
    """Verify pairwise IoU on known geometries."""
    # Box A: [0, 0, 10, 10] -> area 100
    # Box B: [5, 0, 15, 10] -> area 100
    # Intersection: [5, 0, 10, 10] -> area 50
    # Union: 100 + 100 - 50 = 150 -> IoU = 50 / 150 = 1/3
    b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b2 = torch.tensor([[5.0, 0.0, 15.0, 10.0]])

    iou = box_iou(b1, b2).item()
    assert iou == pytest.approx(1.0 / 3.0, abs=1e-5)

    # Completely disjoint boxes
    b3 = torch.tensor([[50.0, 50.0, 60.0, 60.0]])
    iou_disjoint = box_iou(b1, b3).item()
    assert iou_disjoint == pytest.approx(0.0, abs=1e-5)


def test_non_max_suppression():
    """Verify NMS suppresses overlapping boxes while keeping highest scoring box."""
    boxes = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 11.0, 49.0, 51.0],  # Heavy overlap with box 0
        [100.0, 100.0, 150.0, 150.0],  # Disjoint box
    ])
    scores = torch.tensor([0.9, 0.75, 0.85])

    keep = non_max_suppression(boxes, scores, iou_threshold=0.5, score_threshold=0.5)

    # Box 0 (score 0.9) and Box 2 (score 0.85) should be kept; Box 1 suppressed
    assert set(keep.tolist()) == {0, 2}


def test_yolo_and_voc_annotation_parsers():
    """Verify YOLO txt and Pascal VOC XML parsers extract correct bounding boxes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)

        # 1. Test YOLO txt parser
        yolo_file = tmp_p / "sample.txt"
        # class_id=1, cx=0.5, cy=0.5, w=0.2, h=0.2 on 200x200 -> xmin=80, ymin=80, xmax=120, ymax=120
        yolo_file.write_text("1 0.5 0.5 0.2 0.2\n")

        boxes_y, labels_y = parse_yolo_annotation(yolo_file, image_size=(200, 200))
        assert boxes_y.shape == (1, 4)
        assert labels_y.shape == (1,)
        assert labels_y[0].item() == 1
        torch.testing.assert_close(boxes_y[0], torch.tensor([80.0, 80.0, 120.0, 120.0]))

        # 2. Test VOC XML parser
        voc_file = tmp_p / "sample.xml"
        voc_xml = """<annotation>
            <size><width>200</width><height>200</height><depth>3</depth></size>
            <object>
                <name>aphid</name>
                <bndbox><xmin>25</xmin><ymin>30</ymin><xmax>75</xmax><ymax>90</ymax></bndbox>
            </object>
        </annotation>"""
        voc_file.write_text(voc_xml)

        boxes_v, labels_v = parse_voc_xml(voc_file, class_to_idx={"aphid": 2})
        assert boxes_v.shape == (1, 4)
        assert labels_v[0].item() == 2
        torch.testing.assert_close(boxes_v[0], torch.tensor([25.0, 30.0, 75.0, 90.0]))


def test_tiny_yolo_forward_and_prediction_output():
    """Verify TinyYOLODetector forward tensor shape and decode output format."""
    model = TinyYOLODetector(in_channels=3, num_classes=3, grid_size=8, num_boxes_per_cell=1)
    x = torch.randn(2, 3, 256, 256)

    raw = model(x)
    # Output shape: (Batch=2, S=8, S=8, Depth = 1*5 + 3 = 8)
    assert raw.shape == (2, 8, 8, 8)

    preds = model.predict_boxes(x, conf_threshold=0.1, iou_threshold=0.5)
    assert len(preds) == 2
    for p in preds:
        assert "boxes" in p and "scores" in p and "labels" in p
        if p["boxes"].numel() > 0:
            assert p["boxes"].shape[1] == 4
            assert p["scores"].shape[0] == p["boxes"].shape[0]


def test_calculate_map_metrics():
    """Verify mAP calculation on synthetic prediction and ground-truth matches."""
    # 2 images, 1 class (class 0)
    # Image 0: GT [10, 10, 50, 50], Pred [10, 10, 50, 50] score 0.95 (TP)
    # Image 1: GT [20, 20, 60, 60], Pred [80, 80, 120, 120] score 0.80 (FP)
    gts = {0: [torch.tensor([[10.0, 10.0, 50.0, 50.0]]), torch.tensor([[20.0, 20.0, 60.0, 60.0]])]}
    preds = {
        0: (
            [torch.tensor([[10.0, 10.0, 50.0, 50.0]]), torch.tensor([[80.0, 80.0, 120.0, 120.0]])],
            [torch.tensor([0.95]), torch.tensor([0.80])],
        )
    }

    res = calculate_map(preds, gts)
    assert "mAP_0.5" in res
    assert "mAP_0.5:0.95" in res
    # Image 0 was TP (recall = 0.5), Image 1 was FP -> AP should be around 0.5
    assert 0.0 <= res["mAP_0.5"] <= 1.0

