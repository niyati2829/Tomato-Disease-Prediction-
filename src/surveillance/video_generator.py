"""Sample Video Generator for Field Surveillance Validation.

Synthesizes a realistic tomato surveillance video sequence from real field images
(PlantVillage tomato foliage and real Tuta absoluta pest images).
Simulates an agricultural surveillance camera scanning across tomato crop rows.
"""

import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

import numpy as np
import pandas as pd
from PIL import Image

from src.surveillance.video_processor import VideoProcessor


def generate_field_surveillance_video(
    output_path: Path | str = "data/video/sample_field_surveillance.mp4",
    duration_seconds: float = 10.0,
    fps: float = 30.0,
    resolution: tuple[int, int] = (640, 480),  # (W, H)
) -> Path:
    """Generate surveillance video sequence using real tomato images."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    W, H = resolution
    total_frames = int(duration_seconds * fps)

    # 1. Gather source leaf images from test split
    test_csv = Path("data/splits/test.csv")
    candidate_images: List[Path] = []
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        # Sample across multiple distinct classes
        for _, group in df.groupby("class_name"):
            for p in group["image_path"].head(2):
                img_path = Path(p)
                if img_path.exists():
                    candidate_images.append(img_path)

    # Also add real Tuta absoluta pest images
    pest_test_dir = Path("data/pest_detection/real_tuta_absoluta/test/images")
    if pest_test_dir.exists():
        for p in sorted(pest_test_dir.glob("*.jpg"))[:4]:
            candidate_images.append(p)

    if not candidate_images:
        # Fallback if no images found: create clean synthetic foliage frames
        source_frames = [
            np.full((H, W, 3), (35, 120, 40), dtype=np.uint8) for _ in range(5)
        ]
    else:
        source_frames = []
        for p in candidate_images:
            try:
                bgr = cv2.imread(str(p))
                if bgr is not None:
                    resized = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
                    source_frames.append(resized)
            except Exception:
                continue

    if not source_frames:
        source_frames = [np.full((H, W, 3), (35, 120, 40), dtype=np.uint8)]

    # 2. Write video using VideoWriter
    writer = VideoProcessor.create_writer(out_p, W, H, fps, codec="mp4v")

    frames_per_shot = max(1, total_frames // len(source_frames))
    current_shot_idx = 0

    for i in range(total_frames):
        src_idx = (i // frames_per_shot) % len(source_frames)
        next_idx = (src_idx + 1) % len(source_frames)

        base_frame = source_frames[src_idx]
        next_frame = source_frames[next_idx]

        # Smooth cross-fade transition between leaf scenes
        shot_progress = (i % frames_per_shot) / float(frames_per_shot)
        if shot_progress > 0.8:
            alpha = (shot_progress - 0.8) / 0.2
            frame = cv2.addWeighted(base_frame, 1.0 - alpha, next_frame, alpha, 0)
        else:
            frame = base_frame.copy()

        # Add slight pan motion to simulate camera tracking
        pan_dx = int(12.0 * np.sin(2.0 * np.pi * i / 60.0))
        M = np.float32([[1, 0, pan_dx], [0, 1, 0]])
        frame = cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REFLECT)

        writer.write(frame)

    writer.release()
    print(f"Generated sample field surveillance video: {out_p} ({total_frames} frames, {duration_seconds}s, {fps} FPS)")
    return out_p


if __name__ == "__main__":
    generate_field_surveillance_video()
