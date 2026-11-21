# Phase H — Pre-Integration Technical Audit & Architecture Plan

**Project:** Tomato Disease and Pest Surveillance System  
**Task:** Phase H — Unified Streamlit Application Integration  
**Date:** November 2026  
**Status:** **AUDITED & PLANNED** (Zero Destructive Changes)  

---

## 1. Executive Summary & Audit Context

Phase H unifies all independently validated project components into one polished, modular, robust Streamlit application (`app/app.py`). Rather than rewriting the working application or developing new models, Phase H connects the existing production disease classifier, the experimental *Tuta absoluta* pest detector, and the field video surveillance pipeline under a single navigation framework.

This pre-integration audit thoroughly documents the real interfaces, checkpoint locations, device handling, annotation semantics, and known risks of all project components before application code modifications.

---

## 2. Component Audits & Interfaces

### Component A: Disease Classification (Validated Base)
- **Model Architecture:** `TomatoResNet18` (subclass of torchvision ResNet18) in `src/models/transfer_learning.py`.
- **Checkpoint Location:** `artifacts/checkpoints/best_resnet18_weighted.pt`
- **Expected SHA-256:** `b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa`
- **Pre-Integration Verification:** Verified identical (`b86d...a3fa`).
- **Class Mapping:** `data/splits/class_mapping.json` (10 classes: 9 diseases + 1 healthy).
- **Preprocessing:** Resize to $224 \times 224$, `ToTensor()`, Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- **Post-Hoc Calibration:** Temperature scaling with $T^* = 1.3021$ via `ModelWithTemperature` (`src/calibration/temperature_scaling.py`).
- **Interpretability (Grad-CAM):** Supported via `GradCAM(model, model.layer4[-1])` in `src/interpretability/grad_cam.py` and `overlay_heatmap` in `src/interpretability/visualization.py`. Can be safely invoked without mutating model weights.
- **Display Status:** **Evaluated on PlantVillage Holdout Test Set** (99.23% Accuracy, 0.9885 Macro F1, 0.364% Test ECE).

### Component B: Real Pest Detection (Experimental Module)
- **Model Architecture:** `TinyYOLODetector` (8x8 grid, 1 box/cell, 7-channel depth) in `src/detection/model.py`.
- **Checkpoint Location:** `artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt`
- **Trained Classes:** `0: "healthy"`, `1: "tuta_absoluta_damage"` (from Shehu et al. 2025 / Zenodo `13324917`).
- **Preprocessing:** Resize to $256 \times 256$, `ToTensor()`, Normalize with ImageNet mean/std.
- **Inference Method:** `predict_boxes(input_tensor, conf_threshold=0.20, iou_threshold=0.50)`.
- **Output Schema:** List of dicts per image: `{"boxes": Tensor(N, 4), "scores": Tensor(N,), "labels": Tensor(N,)}`.

#### Critical Class-Semantics Audit: What "healthy" Represents
In the underlying Shehu et al. (2025) field dataset (`data/pest_detection/real_tuta_absoluta/`):
- **Class 1 (`infected` / `tuta_absoluta_damage`):** Bounding boxes placed around active leafminer mines, serpentine feeding channels, and necrotic puncture wounds caused by *Tuta absoluta* larvae.
- **Class 0 (`healthy`):** Bounding boxes placed by annotators around unblemished, clean green leaf lamina regions to serve as baseline contrast against infested areas.
- **Finding:** **"Healthy" is NOT a pest object.** Displaying healthy foliar tissue patches as "pests" would be botanically and technically false.
- **UI Policy:** The application strictly displays Class 1 as **"Tuta absoluta Infestation Lesions"** and Class 0 as **"Healthy Foliage Regions (Reference)"**. Detections of Class 0 are never totaled in pest counts.
- **Holdout Test Benchmark:** 12 images, 29 boxes, mAP@0.5 = 0.3044, mAP@0.5:0.95 = 0.0879, Tuta damage recall = 0.2353, CPU latency = 10.27 ms.
- **Display Status:** **Experimental Research Module** (not for pesticide decision-making).

### Component C: Video Surveillance (Multi-Task Pipeline)
- **Ingestion & Validation:** `VideoProcessor` (`src/surveillance/video_processor.py`).
  - Strict guardrails: rejects videos $>60$ seconds duration or $>100$ MB file size.
  - Supports `.mp4`, `.avi`, `.mov`.
- **Multi-Task Frame Analyzer:** `SurveillanceFrameAnalyzer` (`src/surveillance/frame_analyzer.py`).
  - Combines calibrated ResNet18 ($T^*=1.3021$) and TinyYOLODetector.
- **Video HUD Annotator:** `VideoAnnotator` (`src/surveillance/annotator.py`).
  - Draws telemetry header, pathology diagnosis badge, calibrated confidence meter, pest counter, and bounding boxes.
- **Analytics & Plotting:** `plot_surveillance_timeline` (`src/surveillance/timeline_plotter.py`).
- **Benchmark Performance:** 10-second video ($640 \times 480$, 30 FPS) analyzed at temporal stride $k=2$ (150 sampled frames) in 4.43 s $\to$ 26.67 ms/frame (**37.5 FPS** on sampled frames).
- **Display Status:** **Implemented; Subject to Model Limitations**.

### Component D: Lesion Segmentation (Blocked Capability)
- **Architecture:** `UNet` in `src/segmentation/unet.py`.
- **Evaluation Status:** Evaluated strictly on synthetic benchmark foliage sprites (Phase D).
- **Limitation:** `PlantVillage/segmented` contains binary silhouettes separating entire leaves from lab backgrounds, not disease lesions. No open-access dataset currently provides genuine human pixel-level lesion annotations for tomato leaves.
- **UI Policy:** Explicitly declared:
  > *"Real tomato lesion segmentation is currently unavailable. The existing U-Net was evaluated using synthetic data and has not been validated on genuine tomato lesion annotations."*
- **Display Status:** **Unavailable — Dataset Limitation**.

---

## 3. Interface Compatibility & Integration Matrix

| Module | Python Source | Key Classes / Functions | Primary Inputs | Primary Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **Classifier** | `src/models/transfer_learning.py` | `TomatoResNet18`, `load_production_model` | RGB Image (PIL) | Top-3 classes, calibrated probabilities |
| **Calibration** | `src/calibration/temperature_scaling.py` | `ModelWithTemperature` | Logits tensor, $T^*=1.3021$ | Scaled probabilities |
| **Interpretability** | `src/interpretability/grad_cam.py` | `GradCAM`, `overlay_heatmap` | Image tensor, `model.layer4[-1]` | 2D Heatmap & blended RGB image |
| **Pest Detector** | `src/detection/model.py` | `TinyYOLODetector` | RGB Image (PIL), thresholds | Bounding boxes, scores, labels |
| **Video Ingestion**| `src/surveillance/video_processor.py` | `VideoProcessor` | Video path | `VideoMetadata`, sampled frames |
| **Frame Analysis** | `src/surveillance/frame_analyzer.py`| `SurveillanceFrameAnalyzer` | BGR frame array | `FrameAnalysisResult` |
| **Video HUD** | `src/surveillance/annotator.py` | `VideoAnnotator` | BGR frame, `FrameAnalysisResult`| Annotated BGR frame |
| **Timeline** | `src/surveillance/timeline_plotter.py` | `plot_surveillance_timeline`| Frame analysis list | Matplotlib figure / PNG file |

---

## 4. Known Integration Risks & Mitigations

1. **Streamlit Execution Rerun Overhead:**
   - *Risk:* Re-instantiating neural network weights on every user interaction causes latency spikes and GPU/RAM memory leaks.
   - *Mitigation:* Wrap `load_production_model` and `load_pest_detector` with `@st.cache_resource(show_spinner=False)`.
2. **Video Upload Temporary File Collisions:**
   - *Risk:* Processing simultaneous or consecutive video uploads overwriting the same temporary path on disk.
   - *Mitigation:* Use `tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")` and enforce cleanup inside `finally:` blocks.
3. **Hardware Acceleration Failure Modes:**
   - *Risk:* Incompatible MPS operations or headless server environments without Apple Silicon GPU.
   - *Mitigation:* Utilize `get_device(preferred="mps")` with automatic CPU fallback.
4. **Regression of Existing Tests:**
   - *Risk:* Modifying `app/app.py` breaks `tests/test_app.py` which imports `CHECKPOINT_PATH`, `load_production_model`, etc.
   - *Mitigation:* Preserve all existing signatures and constants in `app/app.py` while adding modular tab/page rendering functions.

---

## 5. Proposed Application Architecture & Navigation Plan

Navigation will be driven via a clear sidebar radio selector:

1. **🏠 Home / Overview:** System scope, capability status matrix, limitations, and quick-start guide.
2. **🔬 Disease Classification:** Leaf upload, calibrated ResNet18 diagnosis, top-3 probabilities, uncertainty alerts, optional Grad-CAM activation heatmap, decision disclaimer.
3. **🐛 Pest Detection (Experimental):** Image upload, TinyYOLODetector inference, bounding-box overlays, distinct tuta vs healthy labeling, holdout metrics, prominent experimental warning.
4. **📹 Video Surveillance:** Field video upload, duration/size safeguards, progress bar frame processing, annotated video playback, disease/pest timeline plot, CSV telemetry export.
5. **🧩 Lesion Segmentation (Unavailable):** Transparent academic explanation of the dataset blocker, leaf silhouettes vs lesion masks, future annotation roadmap.
6. **📊 Analytics & Model Performance:** Consolidated benchmarking dashboard presenting Phase F holdout metrics, Phase G pest detection results, and video throughput metrics with strict dataset labeling.
7. **ℹ️ About the Project:** Technical specifications, 10-requirement lifecycle summary, author attribution, and ethical AI in precision agriculture.
