"""Automated Integration & Verification Tests for Phase M3-C / Phase N/O.

Verifies:
1. YOLOv8n-Seg checkpoint loading & parameter integrity.
2. Dual model availability: YOLOv8n-Seg (Production) and TinyYOLO (Baseline).
3. `predict_pests_yolov8` inference structure, box/mask generation, and schema consistency.
4. Edge cases: Empty/blank image, miniature image (32x32).
5. `SurveillanceFrameAnalyzer` with `use_yolov8=True` multi-task video frame analysis.
6. `VideoAnnotator` rendering with YOLOv8 segmentation masks and HUD telemetry.
7. Real PlantSeg U-Net checkpoint availability and `predict_unet_lesions` function.
8. Latency benchmark telemetry artifact integrity.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from app.app import (
    CHECKPOINT_PATH,
    PEST_CHECKPOINT_PATH,
    UNET_CHECKPOINT_PATH,
    YOLO_CHECKPOINT_PATH,
    load_pest_detector,
    load_production_model,
    load_unet_lesion_segmenter,
    load_yolov8_detector,
    predict_pests,
    predict_pests_yolov8,
    predict_unet_lesions,
)
from src.surveillance.annotator import VideoAnnotator
from src.surveillance.frame_analyzer import FrameAnalysisResult, SurveillanceFrameAnalyzer


def test_checkpoints_exist_and_non_empty():
    """Verify all production and baseline checkpoints exist."""
    assert CHECKPOINT_PATH.exists(), f"ResNet18 checkpoint missing: {CHECKPOINT_PATH}"
    assert PEST_CHECKPOINT_PATH.exists(), f"TinyYOLO checkpoint missing: {PEST_CHECKPOINT_PATH}"
    assert YOLO_CHECKPOINT_PATH.exists(), f"YOLOv8n-Seg checkpoint missing: {YOLO_CHECKPOINT_PATH}"
    assert UNET_CHECKPOINT_PATH.exists(), f"U-Net checkpoint missing: {UNET_CHECKPOINT_PATH}"

    assert YOLO_CHECKPOINT_PATH.stat().st_size > 1_000_000, "YOLOv8 checkpoint is unexpectedly small"
    assert PEST_CHECKPOINT_PATH.stat().st_size > 1_000_000, "TinyYOLO checkpoint is unexpectedly small"


def test_yolov8_detector_loader():
    """Verify load_yolov8_detector loads Ultralytics model without conflicts."""
    model = load_yolov8_detector(YOLO_CHECKPOINT_PATH)
    assert model is not None, "Failed to load YOLOv8n-Seg model"
    assert hasattr(model, "predict"), "Model does not have predict method"


def test_predict_pests_yolov8_inference():
    """Verify predict_pests_yolov8 output schema on a synthetic foliage image."""
    model = load_yolov8_detector(YOLO_CHECKPOINT_PATH)
    assert model is not None

    # Synthetic leaf image
    dummy_img = Image.new("RGB", (640, 640), color=(40, 140, 50))
    annotated, detections, tuta_cnt, healthy_cnt = predict_pests_yolov8(
        dummy_img, model, conf_threshold=0.10, iou_threshold=0.45
    )

    assert isinstance(annotated, Image.Image)
    assert annotated.size == (640, 640)
    assert isinstance(detections, list)
    assert isinstance(tuta_cnt, int)
    assert isinstance(healthy_cnt, int)

    for det in detections:
        assert "class_id" in det
        assert "category" in det
        assert "confidence" in det
        assert "box" in det
        assert "mask_area_px" in det
        assert "is_pest_damage" in det
        assert len(det["box"]) == 4


def test_predict_pests_yolov8_blank_image():
    """Verify predict_pests_yolov8 handles completely black/empty images cleanly."""
    model = load_yolov8_detector(YOLO_CHECKPOINT_PATH)
    assert model is not None

    black_img = Image.new("RGB", (300, 300), color=(0, 0, 0))
    annotated, detections, tuta_cnt, healthy_cnt = predict_pests_yolov8(
        black_img, model, conf_threshold=0.90
    )

    assert isinstance(annotated, Image.Image)
    assert len(detections) == 0
    assert tuta_cnt == 0
    assert healthy_cnt == 0


def test_predict_pests_yolov8_small_image():
    """Verify predict_pests_yolov8 handles small images without crashing."""
    model = load_yolov8_detector(YOLO_CHECKPOINT_PATH)
    assert model is not None

    small_img = Image.new("RGB", (32, 32), color=(50, 120, 40))
    annotated, detections, tuta_cnt, healthy_cnt = predict_pests_yolov8(
        small_img, model, conf_threshold=0.25
    )
    assert isinstance(annotated, Image.Image)
    assert annotated.size == (32, 32)


def test_surveillance_frame_analyzer_yolov8_mode():
    """Verify SurveillanceFrameAnalyzer operates in YOLOv8 mode."""
    analyzer = SurveillanceFrameAnalyzer(
        temperature=1.3021,
        use_yolov8=True,
        yolov8_checkpoint=YOLO_CHECKPOINT_PATH,
    )

    assert analyzer.use_yolov8 is True
    assert analyzer.yolov8_detector is not None

    dummy_frame = np.full((480, 640, 3), (35, 120, 40), dtype=np.uint8)
    res = analyzer.analyze_frame(dummy_frame, frame_idx=5, timestamp=0.16)

    assert isinstance(res, FrameAnalysisResult)
    assert res.frame_idx == 5
    assert res.timestamp_seconds == 0.16
    assert res.pest_detector_type == "yolov8n-seg"
    assert isinstance(res.pest_boxes, list)
    assert isinstance(res.pest_scores, list)
    assert isinstance(res.pest_labels, list)


def test_video_annotator_yolov8_masks():
    """Verify VideoAnnotator renders polygon masks when present."""
    annotator = VideoAnnotator()
    dummy_frame = np.full((480, 640, 3), 40, dtype=np.uint8)

    # Synthetic polygon mask: triangle
    poly_mask = np.array([[100, 100], [200, 100], [150, 200]], dtype=np.float32)

    result = FrameAnalysisResult(
        frame_idx=1,
        timestamp_seconds=0.033,
        disease_class_id=9,
        disease_label="Tomato Healthy",
        disease_confidence=0.98,
        disease_probabilities={"Tomato Healthy": 0.98},
        pest_count=1,
        pest_boxes=[[100.0, 100.0, 200.0, 200.0]],
        pest_scores=[0.88],
        pest_labels=["Tuta absoluta Feeding Damage"],
        segmentation_status="Unavailable (Blocked by Dataset Availability)",
        pest_masks=[poly_mask],
        pest_detector_type="yolov8n-seg",
    )

    annotated = annotator.annotate(dummy_frame, result)
    assert annotated.shape == (480, 640, 3)
    assert annotated.dtype == np.uint8
    assert not np.array_equal(annotated, dummy_frame)


def test_unet_lesion_segmentation_integration():
    """Verify real PlantSeg U-Net loader and inference."""
    unet_model, dev, _ = load_unet_lesion_segmenter(UNET_CHECKPOINT_PATH)
    assert unet_model is not None, "Failed to load improved U-Net checkpoint"

    dummy_leaf = Image.new("RGB", (256, 256), color=(40, 120, 50))
    overlay, lesion_pct, mask = predict_unet_lesions(dummy_leaf, unet_model, dev)

    assert isinstance(overlay, Image.Image)
    assert isinstance(lesion_pct, float)
    assert 0.0 <= lesion_pct <= 100.0
    assert mask.shape == (256, 256)


def test_latency_benchmark_telemetry_exists():
    """Verify reports/pest_detection_yolov8_latency_benchmark.json is populated."""
    bench_file = Path("reports/pest_detection_yolov8_latency_benchmark.json")
    assert bench_file.exists(), f"Benchmark telemetry file missing: {bench_file}"

    with open(bench_file) as f:
        data = json.load(f)

    assert "yolov8n_seg_cpu" in data
    assert "tinyyolo_cpu" in data
    assert data["yolov8n_seg_cpu"]["mean_latency_ms"] > 0.0
    assert data["tinyyolo_cpu"]["mean_latency_ms"] > 0.0
    assert data["yolov8n_seg_cpu"]["throughput_fps"] > 0.0
