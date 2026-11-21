"""Standardized Latency and Throughput Benchmark: YOLOv8n-Seg vs TinyYOLO.

Under identical warm-up and timing conditions:
- 10 warm-up passes.
- 50 timed iterations on holdout test images.
- Detailed metrics: Mean, Median, Std Dev, 95th Percentile Latency (ms), and FPS.
- Evaluates CPU and MPS (if hardware acceleration available).
- Exports results to reports/pest_detection_yolov8_latency_benchmark.json.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.detection.model import TinyYOLODetector


def get_test_images(limit: int = 50) -> List[Path]:
    """Retrieve test images for benchmarking."""
    img_dir = PROJECT_ROOT / "data" / "pest_detection" / "tuta_absoluta_research_v4" / "test" / "images"
    if not img_dir.exists():
        raise FileNotFoundError(f"Test image directory missing: {img_dir}")
    imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not imgs:
        raise FileNotFoundError(f"No test images found in {img_dir}")
    return imgs[:limit]


def benchmark_yolov8(
    model_path: Path,
    image_paths: List[Path],
    device_str: str = "cpu",
    warmup_iters: int = 10,
    timed_iters: int = 50,
) -> Dict[str, Any]:
    """Benchmark YOLOv8n-Seg model."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))

    # Preload images as PIL
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]
    n_imgs = len(pil_images)

    # 1. Warm-up
    for i in range(warmup_iters):
        img = pil_images[i % n_imgs]
        _ = model.predict(img, device=device_str, verbose=False)

    # 2. Timed Iterations
    latencies: List[float] = []
    for i in range(timed_iters):
        img = pil_images[i % n_imgs]
        t0 = time.perf_counter()
        _ = model.predict(img, device=device_str, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    arr = np.array(latencies)
    mean_ms = float(np.mean(arr))
    median_ms = float(np.median(arr))
    std_ms = float(np.std(arr))
    p95_ms = float(np.percentile(arr, 95))
    fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

    return {
        "model": "YOLOv8n-Seg (Production / Recommended)",
        "device": device_str.upper(),
        "warmup_iterations": warmup_iters,
        "timed_iterations": timed_iters,
        "mean_latency_ms": round(mean_ms, 2),
        "median_latency_ms": round(median_ms, 2),
        "std_latency_ms": round(std_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "throughput_fps": round(fps, 2),
    }


def benchmark_tinyyolo(
    checkpoint_path: Path,
    image_paths: List[Path],
    device_str: str = "cpu",
    warmup_iters: int = 10,
    timed_iters: int = 50,
) -> Dict[str, Any]:
    """Benchmark TinyYOLO baseline detector."""
    device = torch.device(device_str)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = TinyYOLODetector(
        in_channels=3,
        num_classes=ckpt.get("num_classes", 2),
        grid_size=ckpt.get("grid_size", 8),
        num_boxes_per_cell=ckpt.get("num_boxes_per_cell", 1),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pil_images = [Image.open(p).convert("RGB") for p in image_paths]
    n_imgs = len(pil_images)
    tensors = [transform(img).unsqueeze(0).to(device) for img in pil_images]

    # 1. Warm-up
    with torch.no_grad():
        for i in range(warmup_iters):
            t_img = tensors[i % n_imgs]
            _ = model.predict_boxes(t_img, conf_threshold=0.20, iou_threshold=0.50)

    # 2. Timed Iterations
    latencies: List[float] = []
    with torch.no_grad():
        for i in range(timed_iters):
            t_img = tensors[i % n_imgs]
            t0 = time.perf_counter()
            _ = model.predict_boxes(t_img, conf_threshold=0.20, iou_threshold=0.50)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    arr = np.array(latencies)
    mean_ms = float(np.mean(arr))
    median_ms = float(np.median(arr))
    std_ms = float(np.std(arr))
    p95_ms = float(np.percentile(arr, 95))
    fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

    return {
        "model": "TinyYOLO (Experimental Baseline)",
        "device": device_str.upper(),
        "warmup_iterations": warmup_iters,
        "timed_iterations": timed_iters,
        "mean_latency_ms": round(mean_ms, 2),
        "median_latency_ms": round(median_ms, 2),
        "std_latency_ms": round(std_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "throughput_fps": round(fps, 2),
    }


def main():
    yolo_ckpt = PROJECT_ROOT / "artifacts" / "checkpoints" / "pest_detection_yolov8" / "best_yolov8n_seg.pt"
    tiny_ckpt = PROJECT_ROOT / "artifacts" / "checkpoints" / "pest_detection_real" / "best_pest_detector_real.pt"

    if not yolo_ckpt.exists():
        raise FileNotFoundError(f"YOLOv8 checkpoint missing: {yolo_ckpt}")
    if not tiny_ckpt.exists():
        raise FileNotFoundError(f"TinyYOLO checkpoint missing: {tiny_ckpt}")

    test_imgs = get_test_images(limit=50)
    print(f"Loaded {len(test_imgs)} test images for benchmarking.")

    benchmarks = {}

    # CPU Benchmarks
    print("\n--- Running YOLOv8n-Seg CPU Benchmark ---")
    benchmarks["yolov8n_seg_cpu"] = benchmark_yolov8(yolo_ckpt, test_imgs, device_str="cpu")
    print(f"  YOLOv8n-Seg CPU: {benchmarks['yolov8n_seg_cpu']['mean_latency_ms']} ms ({benchmarks['yolov8n_seg_cpu']['throughput_fps']} FPS)")

    print("\n--- Running TinyYOLO CPU Benchmark ---")
    benchmarks["tinyyolo_cpu"] = benchmark_tinyyolo(tiny_ckpt, test_imgs, device_str="cpu")
    print(f"  TinyYOLO CPU: {benchmarks['tinyyolo_cpu']['mean_latency_ms']} ms ({benchmarks['tinyyolo_cpu']['throughput_fps']} FPS)")

    # MPS Benchmarks (if available)
    mps_available = torch.backends.mps.is_available()
    benchmarks["environment"] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": mps_available,
    }

    if mps_available:
        print("\n--- Running YOLOv8n-Seg MPS Benchmark ---")
        try:
            benchmarks["yolov8n_seg_mps"] = benchmark_yolov8(yolo_ckpt, test_imgs, device_str="mps")
            print(f"  YOLOv8n-Seg MPS: {benchmarks['yolov8n_seg_mps']['mean_latency_ms']} ms ({benchmarks['yolov8n_seg_mps']['throughput_fps']} FPS)")
        except Exception as e:
            benchmarks["yolov8n_seg_mps"] = {"error": str(e)}

        print("\n--- Running TinyYOLO MPS Benchmark ---")
        try:
            benchmarks["tinyyolo_mps"] = benchmark_tinyyolo(tiny_ckpt, test_imgs, device_str="mps")
            print(f"  TinyYOLO MPS: {benchmarks['tinyyolo_mps']['mean_latency_ms']} ms ({benchmarks['tinyyolo_mps']['throughput_fps']} FPS)")
        except Exception as e:
            benchmarks["tinyyolo_mps"] = {"error": str(e)}
    else:
        print("\nMPS hardware acceleration not available in current environment; CPU benchmarks recorded.")

    out_file = PROJECT_ROOT / "reports" / "pest_detection_yolov8_latency_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"\nBenchmark successfully saved to: {out_file}")


if __name__ == "__main__":
    main()
