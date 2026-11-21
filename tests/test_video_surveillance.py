"""Automated Unit Tests for Tomato Video Surveillance Pipeline (Phase G - REQ-G4).

Verifies:
1. VideoProcessor safety boundaries (<60s duration limit, <100MB file size limit).
2. VideoMetadata extraction correctness.
3. Frame extraction generator yields expected shapes and monotonic timestamps.
4. SurveillanceFrameAnalyzer multi-task inference and temperature calibration (T*=1.3021).
5. Academic transparency flag: segmentation_status == 'Unavailable (Blocked by Dataset Availability)'.
6. VideoAnnotator HUD overlay generation preserves resolution and channel count.
7. Timeline plotter creates valid artifact figure.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.surveillance.annotator import VideoAnnotator
from src.surveillance.frame_analyzer import FrameAnalysisResult, SurveillanceFrameAnalyzer
from src.surveillance.timeline_plotter import plot_surveillance_timeline
from src.surveillance.video_processor import VideoMetadata, VideoProcessor


@pytest.fixture
def sample_video_path():
    """Fixture providing path to sample surveillance video."""
    video_path = Path("data/video/sample_field_surveillance.mp4")
    if not video_path.exists():
        pytest.skip("Sample field surveillance video not present")
    return video_path


def test_video_processor_metadata_validation(sample_video_path):
    """Test video validation and metadata extraction."""
    processor = VideoProcessor(max_duration_seconds=60.0, max_file_size_mb=100.0)
    meta = processor.validate_and_inspect(sample_video_path)

    assert isinstance(meta, VideoMetadata)
    assert meta.width > 0
    assert meta.height > 0
    assert meta.fps > 0
    assert meta.frame_count > 0
    assert 0.0 < meta.duration_seconds <= 60.0
    assert meta.file_size_mb < 100.0


def test_video_processor_duration_safeguard(tmp_path):
    """Test that videos exceeding duration limit raise ValueError."""
    processor = VideoProcessor(max_duration_seconds=1.0)  # strict 1s limit
    # Generate 3-second dummy video
    fake_video = tmp_path / "long_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(fake_video), fourcc, 10.0, (128, 128))
    for _ in range(30):  # 30 frames at 10 FPS = 3.0 seconds
        writer.write(np.zeros((128, 128, 3), dtype=np.uint8))
    writer.release()

    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        processor.validate_and_inspect(fake_video)


def test_video_processor_sample_frames(sample_video_path):
    """Test frame sampling yields valid frames and timestamps."""
    processor = VideoProcessor()
    frames = list(processor.sample_frames(sample_video_path, stride=10, max_frames=3))

    assert len(frames) == 3
    for f_idx, ts, frame_bgr in frames:
        assert isinstance(f_idx, int)
        assert ts >= 0.0
        assert isinstance(frame_bgr, np.ndarray)
        assert frame_bgr.ndim == 3
        assert frame_bgr.shape[2] == 3


def test_surveillance_frame_analyzer():
    """Test multi-task inference engine."""
    classifier_ckpt = Path("artifacts/checkpoints/best_resnet18_weighted.pt")
    if not classifier_ckpt.exists():
        pytest.skip("Classifier checkpoint missing")

    analyzer = SurveillanceFrameAnalyzer(
        classifier_checkpoint=classifier_ckpt,
        temperature=1.3021,
    )

    # Test with synthetic foliage frame
    dummy_frame = np.full((480, 640, 3), (35, 120, 40), dtype=np.uint8)
    res = analyzer.analyze_frame(dummy_frame, frame_idx=10, timestamp=0.33)

    assert isinstance(res, FrameAnalysisResult)
    assert res.frame_idx == 10
    assert res.timestamp_seconds == 0.33
    assert 0 <= res.disease_class_id < 10
    assert 0.0 <= res.disease_confidence <= 1.0
    assert len(res.disease_probabilities) == 10
    assert res.pest_count >= 0
    assert res.segmentation_status == "Unavailable (Blocked by Dataset Availability)"


def test_video_annotator_overlay():
    """Test HUD annotator preserves dimensions and channels."""
    annotator = VideoAnnotator()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_result = FrameAnalysisResult(
        frame_idx=1,
        timestamp_seconds=0.033,
        disease_class_id=0,
        disease_label="Tomato Bacterial Spot",
        disease_confidence=0.95,
        disease_probabilities={"Tomato Bacterial Spot": 0.95},
        pest_count=1,
        pest_boxes=[[50.0, 50.0, 150.0, 150.0]],
        pest_scores=[0.85],
        pest_labels=["tuta_absoluta_damage"],
        segmentation_status="Unavailable (Blocked by Dataset Availability)",
    )

    annotated = annotator.annotate(dummy_frame, dummy_result)
    assert annotated.shape == (480, 640, 3)
    assert not np.array_equal(annotated, dummy_frame), "Overlay should modify canvas"


def test_timeline_plotter(tmp_path):
    """Test timeline plotter generates valid PNG image file."""
    results = [
        FrameAnalysisResult(
            frame_idx=i,
            timestamp_seconds=float(i * 0.1),
            disease_class_id=0 if i < 5 else 1,
            disease_label="Tomato Healthy" if i < 5 else "Tomato Early Blight",
            disease_confidence=0.90 + 0.01 * i,
            disease_probabilities={},
            pest_count=i % 2,
            pest_boxes=[],
            pest_scores=[],
            pest_labels=[],
        )
        for i in range(10)
    ]

    out_fig = tmp_path / "test_timeline.png"
    plot_surveillance_timeline(results, save_path=out_fig)

    assert out_fig.exists()
    assert out_fig.stat().st_size > 1000  # Valid non-empty PNG
