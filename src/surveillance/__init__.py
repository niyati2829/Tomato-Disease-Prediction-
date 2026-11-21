"""Video Surveillance Pipeline for Tomato Disease and Pest Monitoring (REQ-Phase G).

Integrates:
1. Modular video ingestion and validation (<60s, <100MB, MP4/AVI).
2. Frame-level tomato disease classification using calibrated ResNet18 (T*=1.3021).
3. Real tomato pest detection (Tuta absoluta) using trained detector.
4. Professional video overlay, frame logs, and disease/pest temporal timeline analytics.
"""

from src.surveillance.video_processor import VideoProcessor, VideoMetadata
from src.surveillance.frame_analyzer import SurveillanceFrameAnalyzer, FrameAnalysisResult
from src.surveillance.annotator import VideoAnnotator
from src.surveillance.timeline_plotter import plot_surveillance_timeline

__all__ = [
    "VideoProcessor",
    "VideoMetadata",
    "SurveillanceFrameAnalyzer",
    "FrameAnalysisResult",
    "VideoAnnotator",
    "plot_surveillance_timeline",
]
