"""End-to-End Tomato Video Surveillance Runner (Phase G).

Executes the complete surveillance workflow:
1. Video validation and metadata extraction (<60s, <100MB).
2. Frame sampling and multi-task inference (Calibrated ResNet18 + Real Pest Detection).
3. High-quality HUD annotation and export to artifacts/video/annotated_surveillance.mp4.
4. Frame-level analytics export to reports/video_surveillance_frames.csv.
5. Dual-axis timeline plotting to artifacts/figures/video/disease_timeline.png.
6. Summary reporting to reports/video_surveillance_summary.json.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd

from src.surveillance.annotator import VideoAnnotator
from src.surveillance.frame_analyzer import FrameAnalysisResult, SurveillanceFrameAnalyzer
from src.surveillance.timeline_plotter import plot_surveillance_timeline
from src.surveillance.video_processor import VideoProcessor


def run_surveillance_pipeline(
    video_path: Path | str = "data/video/sample_field_surveillance.mp4",
    output_video_path: Path | str = "artifacts/video/annotated_surveillance.mp4",
    csv_report_path: Path | str = "reports/video_surveillance_frames.csv",
    timeline_figure_path: Path | str = "artifacts/figures/video/disease_timeline.png",
    summary_report_path: Path | str = "reports/video_surveillance_summary.json",
    stride: int = 2,  # Process every 2nd frame for 15 FPS surveillance throughput
) -> Dict[str, Any]:
    """Run full automated surveillance analysis on video."""
    video_p = Path(video_path)
    processor = VideoProcessor(max_duration_seconds=60.0, max_file_size_mb=100.0)

    print(f"\n[Phase G] Inspecting video: {video_p}")
    metadata = processor.validate_and_inspect(video_p)
    print(
        f"  Metadata: {metadata.width}x{metadata.height} | "
        f"{metadata.fps:.1f} FPS | "
        f"{metadata.frame_count} frames | "
        f"{metadata.duration_seconds:.2f}s | "
        f"{metadata.file_size_mb:.2f} MB"
    )

    analyzer = SurveillanceFrameAnalyzer()
    annotator = VideoAnnotator()

    # Create annotated video writer
    out_vid_p = Path(output_video_path)
    effective_fps = metadata.fps / float(stride)
    writer = processor.create_writer(
        out_vid_p, metadata.width, metadata.height, fps=effective_fps, codec="mp4v"
    )

    results: List[FrameAnalysisResult] = []
    frame_latencies: List[float] = []

    print(f"[Phase G] Processing video with temporal stride = {stride}...")
    t_start = time.perf_counter()

    for frame_idx, timestamp, frame in processor.sample_frames(video_p, stride=stride):
        t0 = time.perf_counter()
        res = analyzer.analyze_frame(frame, frame_idx=frame_idx, timestamp=timestamp)
        t1 = time.perf_counter()
        frame_latencies.append((t1 - t0) * 1000.0)

        annotated_frame = annotator.annotate(frame, res)
        writer.write(annotated_frame)
        results.append(res)

    writer.release()
    total_time = time.perf_counter() - t_start

    print(f"Processed {len(results)} frames in {total_time:.2f}s (Average latency: {np.mean(frame_latencies):.1f} ms/frame)")

    # 1. Export CSV Log
    csv_rows = []
    for r in results:
        csv_rows.append({
            "frame_idx": r.frame_idx,
            "timestamp_seconds": round(r.timestamp_seconds, 3),
            "disease_class_id": r.disease_class_id,
            "disease_label": r.disease_label,
            "disease_confidence": round(r.disease_confidence, 4),
            "pest_count": r.pest_count,
            "pest_labels": ";".join(r.pest_labels) if r.pest_labels else "none",
            "pest_scores": ";".join([f"{s:.2f}" for s in r.pest_scores]) if r.pest_scores else "none",
            "segmentation_status": r.segmentation_status,
        })
    df_csv = pd.DataFrame(csv_rows)
    csv_p = Path(csv_report_path)
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    df_csv.to_csv(csv_p, index=False)
    print(f"Frame log exported: {csv_p}")

    # 2. Export Timeline Plot
    plot_surveillance_timeline(results, save_path=timeline_figure_path)
    print(f"Timeline plot exported: {timeline_figure_path}")

    # 3. Export Summary JSON
    disease_counts = df_csv["disease_label"].value_counts().to_dict()
    dominant_disease = df_csv["disease_label"].mode()[0] if not df_csv.empty else "N/A"
    max_pests = int(df_csv["pest_count"].max()) if not df_csv.empty else 0
    total_pests_detected = int(df_csv["pest_count"].sum()) if not df_csv.empty else 0

    summary = {
        "video_input": str(video_p),
        "video_annotated": str(out_vid_p),
        "metadata": {
            "resolution": f"{metadata.width}x{metadata.height}",
            "original_fps": metadata.fps,
            "sampled_fps": effective_fps,
            "total_frames_sampled": len(results),
            "duration_seconds": metadata.duration_seconds,
            "file_size_mb": round(metadata.file_size_mb, 2),
        },
        "performance": {
            "total_processing_time_s": round(total_time, 2),
            "mean_latency_ms_per_frame": round(float(np.mean(frame_latencies)), 2),
            "surveillance_fps": round(1000.0 / float(np.mean(frame_latencies)), 1),
        },
        "pathology_findings": {
            "dominant_disease": dominant_disease,
            "disease_distribution": disease_counts,
            "mean_confidence": round(float(df_csv["disease_confidence"].mean()), 4),
        },
        "entomology_findings": {
            "total_tuta_absoluta_detections": total_pests_detected,
            "max_pests_in_single_frame": max_pests,
            "frames_with_pests": int((df_csv["pest_count"] > 0).sum()),
        },
        "segmentation_status": "Unavailable (Blocked by Dataset Availability)",
    }

    sum_p = Path(summary_report_path)
    sum_p.parent.mkdir(parents=True, exist_ok=True)
    with open(sum_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary report exported: {sum_p}")

    return summary


if __name__ == "__main__":
    run_surveillance_pipeline()
