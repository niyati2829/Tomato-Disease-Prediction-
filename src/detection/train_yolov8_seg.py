"""Phase M3-B: YOLOv8n-Seg Training Pipeline for Tuta absoluta Foliar Pest Detection.

Fine-tunes pretrained YOLOv8n-seg on the verified TutaAbsolutaResearch v4 dataset.

Disciplines:
- Uses pretrained weights: artifacts/checkpoints/pest_detection_yolov8/yolov8n-seg.pt.
- Ingests verified dataset: data/pest_detection/tuta_absoluta_research_v4/data.yaml.
- Checkpoints best model to: artifacts/checkpoints/pest_detection_yolov8/best_yolov8n_seg.pt.
- Preserves existing TinyYOLO baseline and Kasaisa dataset.
- Evaluates validation performance for model selection.
- Exports reproducible training telemetry to reports/.

Author: Crop Disease Detection Project Team
Phase: Phase M3-B
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from ultralytics import YOLO

CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints" / "pest_detection_yolov8"
PRETRAINED_WEIGHTS = CHECKPOINT_DIR / "yolov8n-seg.pt"
DATA_YAML = PROJECT_ROOT / "data" / "pest_detection" / "tuta_absoluta_research_v4" / "data.yaml"

BEST_MODEL_TARGET = CHECKPOINT_DIR / "best_yolov8n_seg.pt"
CONFIG_JSON = PROJECT_ROOT / "reports" / "pest_detection_yolov8_training_config.json"
HISTORY_JSON = PROJECT_ROOT / "reports" / "pest_detection_yolov8_training_history.json"


def train_yolov8n_seg(
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 8,
    seed: int = 42,
    patience: int = 10,
    workers: int = 2,
) -> Dict[str, Any]:
    """Execute YOLOv8n-seg fine-tuning."""
    print("=" * 80)
    print("PHASE M3-B: YOLOV8N-SEG TRAINING ON TUTA ABSOLUTA RESEARCH V4")
    print("=" * 80)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing data.yaml at {DATA_YAML}")

    if not PRETRAINED_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing pretrained weights at {PRETRAINED_WEIGHTS}")

    # Determine hardware acceleration
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    print(f"Hardware execution device: {device}")
    print(f"Pretrained weights: {PRETRAINED_WEIGHTS}")
    print(f"Dataset config: {DATA_YAML}")
    print(f"Target epochs: {epochs} | Batch size: {batch_size} | Image size: {imgsz}")
    print("=" * 80)

    # 1. Initialize YOLOv8n-seg model
    model = YOLO(str(PRETRAINED_WEIGHTS))

    # 2. Record configuration
    config = {
        "model_architecture": "YOLOv8n-seg",
        "task": "segment",
        "pretrained_weights": str(PRETRAINED_WEIGHTS),
        "data_yaml": str(DATA_YAML),
        "dataset_name": "TutaAbsolutaResearch v4 (Roboflow Universe)",
        "classes": {0: "Healthy", 1: "Tuta absoluta"},
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "seed": seed,
        "patience": patience,
        "device": device,
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "box_loss_gain": 7.5,
        "cls_loss_gain": 0.5,
        "dfl_loss_gain": 1.5,
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    t0 = time.time()

    # 3. Train
    runs_dir = PROJECT_ROOT / "runs" / "pest_detection_yolov8"
    run_name = "train_yolov8n_seg"

    train_results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        seed=seed,
        patience=patience,
        workers=workers,
        project=str(runs_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
    )

    elapsed_time = round(time.time() - t0, 2)
    print(f"\nTraining completed in {elapsed_time:.1f} s ({elapsed_time/60:.2f} min).")

    # 4. Locate best model weights and copy to persistent checkpoint target
    run_save_dir = runs_dir / run_name
    best_weights_src = run_save_dir / "weights" / "best.pt"
    if not best_weights_src.exists():
        # Fallback to last.pt if best.pt not found
        best_weights_src = run_save_dir / "weights" / "last.pt"

    if best_weights_src.exists():
        shutil.copy2(best_weights_src, BEST_MODEL_TARGET)
        print(f"Saved best model checkpoint to: {BEST_MODEL_TARGET} ({BEST_MODEL_TARGET.stat().st_size / (1024*1024):.2f} MB)")
    else:
        raise FileNotFoundError(f"Weights file not found in {run_save_dir}")

    # 5. Extract and save epoch-by-epoch history
    csv_results = run_save_dir / "results.csv"
    history_records = []
    if csv_results.exists():
        df_res = pd.read_csv(csv_results)
        df_res.columns = [c.strip() for c in df_res.columns]
        history_records = df_res.to_dict(orient="records")

    history_payload = {
        "training_time_seconds": elapsed_time,
        "epochs_trained": len(history_records),
        "best_checkpoint_path": str(BEST_MODEL_TARGET),
        "metrics_per_epoch": history_records,
    }
    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    print(f"Training history telemetry exported to: {HISTORY_JSON}")
    return history_payload


if __name__ == "__main__":
    train_yolov8n_seg()
