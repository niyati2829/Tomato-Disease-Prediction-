"""Modular Video Processor for Tomato Field Video Surveillance.

Provides:
- Video ingestion and strict boundary validation (<60s duration, <100MB file size).
- Metadata extraction (resolution, FPS, duration, file size, codec).
- Frame sampling with configurable temporal stride.
- Video output writing for annotated surveillance exports.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Metadata container for surveillance video."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    file_size_mb: float
    codec: str


class VideoProcessor:
    """Manages video decoding, validation, sampling, and encoding."""

    def __init__(
        self,
        max_duration_seconds: float = 60.0,
        max_file_size_mb: float = 100.0,
        supported_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
    ) -> None:
        self.max_duration_seconds = max_duration_seconds
        self.max_file_size_mb = max_file_size_mb
        self.supported_extensions = tuple(ext.lower() for ext in supported_extensions)

    def validate_and_inspect(self, video_path: Path | str) -> VideoMetadata:
        """Validate safety constraints and extract video metadata.

        Args:
            video_path: Path to video file.

        Returns:
            VideoMetadata instance.

        Raises:
            FileNotFoundError: If video file does not exist.
            ValueError: If format is unsupported or violates safety limits.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"Unsupported video extension '{path.suffix}'. Supported: {self.supported_extensions}"
            )

        file_size_mb = path.stat().st_size / (1024.0 * 1024.0)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(
                f"Video file size ({file_size_mb:.2f} MB) exceeds maximum allowed limit ({self.max_file_size_mb:.1f} MB)."
            )

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV failed to open video file: {path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

            if fps <= 0:
                fps = 30.0  # Safe default fallback

            duration_seconds = frame_count / fps if fps > 0 else 0.0

            if duration_seconds > self.max_duration_seconds:
                raise ValueError(
                    f"Video duration ({duration_seconds:.2f} s) exceeds maximum allowed limit ({self.max_duration_seconds:.1f} s)."
                )

            return VideoMetadata(
                path=str(path),
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration_seconds=duration_seconds,
                file_size_mb=file_size_mb,
                codec=codec,
            )
        finally:
            cap.release()

    def sample_frames(
        self,
        video_path: Path | str,
        stride: int = 1,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yield sampled frames from video.

        Args:
            video_path: Path to video.
            stride: Extract every stride-th frame (default 1).
            max_frames: Optional cap on total yielded frames.

        Yields:
            (frame_idx, timestamp_seconds, frame_bgr)
        """
        metadata = self.validate_and_inspect(video_path)
        cap = cv2.VideoCapture(str(video_path))
        fps = metadata.fps if metadata.fps > 0 else 30.0

        frame_idx = 0
        yielded_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % stride == 0:
                    timestamp = frame_idx / fps
                    yield frame_idx, timestamp, frame
                    yielded_count += 1
                    if max_frames is not None and yielded_count >= max_frames:
                        break

                frame_idx += 1
        finally:
            cap.release()

    @staticmethod
    def create_writer(
        output_path: Path | str,
        width: int,
        height: int,
        fps: float,
        codec: str = "mp4v",
    ) -> cv2.VideoWriter:
        """Create OpenCV VideoWriter instance."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))
        if not writer.isOpened():
            # Fallback codec
            fourcc_fallback = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(out_p), fourcc_fallback, fps, (width, height))
        return writer
