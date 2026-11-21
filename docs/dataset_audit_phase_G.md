# Dataset Feasibility Audit: Real Tomato Surveillance Capabilities (Phase G — REQ-G1)
## Tomato Disease & Pest Surveillance System

**Audit Date**: November 15, 2026  
**Status**: AUDIT COMPLETE — PENDING USER REVIEW & APPROVAL BEFORE DOWNLOAD  
**Environment**: MacBook Air M2, Python 3.12, PyTorch, OpenCV  
**Strict Domain Scope**: Tomato (*Solanum lycopersicum*) only  

---

## 1. Executive Summary of Audit Findings

| Surveillance Capability | Candidate Datasets Investigated | Genuine Human Annotations? | Estimated Storage | Feasibility & Recommendation |
|:---|:---|:---:|:---:|:---|
| **Task A: Real Tomato Pest Detection** | • IP102 (Tomato Subset)<br/>• Roboflow: Tomato Pest Dataset (Pesto)<br/>• Roboflow: Pesto Tomato Leaf Pest & Disease | **YES** (Bounding boxes in YOLO / Pascal VOC format) | • IP102 Full: ~22 GB (Disqualified)<br/>• Roboflow Subset: **~35 MB – 65 MB** | **FEASIBLE**: Recommend compact, tomato-specific Roboflow pest dataset (~600–1,200 images, 4–5 classes). Avoid full IP102 due to 22 GB storage penalty. |
| **Task B: Real Tomato Lesion Segmentation** | • PlantVillage `raw/segmented`<br/>• Roboflow: Tomato Leaf Lesion Segmentation<br/>• TDSD (Tomato Disease Segmentation Dataset) | • PlantVillage: **NO** (Silhouettes only)<br/>• Roboflow / TDSD: **YES** (Polygon lesion masks) | • Roboflow / TDSD: **~40 MB – 85 MB** | **FEASIBLE with strict caveats**: PlantVillage segmented images are disqualified (leaf silhouettes only). Recommend curated Roboflow / TDSD polygon lesion dataset with genuine lesion-vs-healthy masks. |
| **Task C: Tomato Field-Video Surveillance** | • Real agricultural greenhouse tomato vine video clip (`.mp4`)<br/>• OpenCV `cv2.VideoCapture` pipeline | **YES** (Native temporal sensor video) | **~15 MB – 30 MB** (10–30s 1080p MP4) | **HIGHLY FEASIBLE**: Standard OpenCV frame extraction, frozen ResNet18 inference, timeline CSV, and annotated output video. |

---

## 2. Task A: Real Tomato Pest Detection Audit

### 2.1 Candidate 1: IP102 Insect Pest Benchmark (Wu et al., CVPR 2019)
- **Source**: `https://github.com/xpwu95/IP102` / `https://arxiv.org/abs/1903.10739`
- **Total Benchmark**: 75,222 images across 102 insect pest classes; 18,974 images annotated with bounding boxes.
- **Tomato-Relevant Classes in IP102**:
  - Class 4: *Helicoverpa armigera* (Cotton bollworm / Tomato fruitworm)
  - Class 12: *Spodoptera litura* (Tobacco cutworm / Asian cotton leafworm)
  - Class 17: *Bemisia tabaci* (Silverleaf whitefly — vector of Tomato Yellow Leaf Curl Virus)
  - Class 21: *Myzus persicae* (Green peach aphid)
  - Class 41: *Tetranychus urticae* (Two-spotted spider mite)
  - Class 63: *Tuta absoluta* (Tomato leafminer)
  - Class 70: *Thrips tabaci* (Onion thrips / TSWV vector)
- **License**: Non-commercial Academic Research License.
- **Audit Assessment**:
  - *Annotation Quality*: High-quality genuine human bounding boxes.
  - *Download & Storage Barrier*: The bounding-box subset requires downloading the full multi-part tar archive (~22 GB compressed, ~35 GB extracted), containing mostly non-tomato pests (rice, wheat, corn, citrus pests).
  - *Recommendation*: **DISQUALIFIED for direct download** due to disproportionate storage and bandwidth footprint on the local MacBook Air M2.

### 2.2 Candidate 2: Roboflow Universe — Tomato Pest Dataset (Pesto / Curated)
- **Source**: `https://universe.roboflow.com/pesto/tomato-pest-35mqs` (and `universe.roboflow.com/pestoapp/pesto-tomato-leaf-disease-and-pest`)
- **License**: Open Source / Creative Commons Attribution 4.0 (CC BY 4.0).
- **Target Pest Classes**:
  1. *Whitefly* (*Bemisia tabaci*)
  2. *Spider Mite* (*Tetranychus urticae*)
  3. *Leaf Miner* (*Tuta absoluta*)
  4. *Aphid* (*Myzus persicae*)
  5. *Hornworm / Fruitworm* (*Helicoverpa / Manduca*)
- **Image Count**: ~600 to 1,200 curated, non-duplicated tomato plant images.
- **Annotation Format**: Genuine human bounding boxes exported in native YOLO format (`.txt` per image: `class_idx x_center y_center width height` normalized to $[0, 1]$) or Pascal VOC XML.
- **Download Size**: **~35 MB to 65 MB**.
- **Compatibility**: Perfectly matches local M2 Apple Silicon environment. Fits directly into the existing repository's bounding-box and detection structures (`src/detection/boxes.py`, `src/detection/metrics.py`).
- **Audit Recommendation**: **RECOMMENDED PRIMARY CHOICE FOR TASK A**.

---

## 3. Task B: Real Tomato Leaf Lesion Segmentation Audit

### 3.1 Disqualification of PlantVillage `raw/segmented`
- **Location on Disk**: `data/raw/PlantVillage-Dataset/raw/segmented/`
- **Nature of Images**: Binary green-leaf silhouettes extracted via automated color thresholding on uniform lab backdrops.
- **Crucial Distinction**: The masks separate the entire leaf blade from the black backdrop. They do **NOT** segment individual diseased lesion spots (e.g., necrosis, chlorosis, halos, bacterial lesions).
- **Audit Verdict**: **STRICTLY DISQUALIFIED as ground-truth lesion masks.** Using whole-leaf silhouettes as lesion ground truth is scientifically fraudulent.

### 3.2 Candidate 1: Roboflow Universe — Tomato Leaf Lesion Segmentation
- **Source**: `https://universe.roboflow.com/leaf-disease/tomato-leaf-disease-segmentation` / `universe.roboflow.com/thesis-team-ana/tomato-disease-detection-syl9w`
- **License**: CC BY 4.0 Open Source.
- **Image Count**: ~400 to 800 high-resolution tomato leaf images.
- **Annotation Format**: Genuine polygon segmentation masks distinguishing:
  - Class 0: Background
  - Class 1: Healthy Leaf Tissue
  - Class 2: Diseased Lesion Spots (Early Blight concentric rings, Late Blight patches, Septoria spots, Leaf Mold)
- **Annotation Type**: Pixel-level binary/multi-class PNG masks and COCO Polygon JSON.
- **Download Size**: **~40 MB to 85 MB**.
- **Compatibility**: Directly ingestible by our PyTorch U-Net (`src/segmentation/unet.py`, `src/segmentation/dataset.py`) with nearest-neighbor categorical mask interpolation.
- **Audit Recommendation**: **RECOMMENDED PRIMARY CHOICE FOR TASK B**.

---

## 4. Task C: Tomato Field-Video Surveillance Feasibility

- **Objective**: Ingest real continuous video of tomato plants, extract frames with metadata, execute frame-level classification via frozen `best_resnet18_weighted.pt`, and generate an annotated video and temporal diagnostic timeline.
- **Video Input Specification**:
  - Container / Codec: `.mp4` (H.264 / AVC), `.avi`.
  - Dimensions: $1280 \times 720$ (720p) or $1920 \times 1080$ (1080p).
  - Duration: 10 to 30 seconds (15 MB – 30 MB).
  - Frame Rate: 24 to 30 FPS.
- **OpenCV Ingestion & Safeguards**:
  - Decoded via `cv2.VideoCapture`.
  - Configurable sampling stride: Process 1 frame every 0.5s or 1.0s (e.g. 1 frame every 15–30 frames) to eliminate redundant computation while capturing field changes.
  - Duration safeguard: Reject files $> 60$ seconds or $> 100$ MB to protect edge memory.
- **Output Artifacts**:
  - Annotated MP4 video (`cv2.VideoWriter`, codec `mp4v`).
  - Frame-by-frame diagnostic CSV log (`reports/video_surveillance_frames.csv`).
  - Diagnostic timeline plot (`artifacts/figures/video/disease_timeline.png`).
- **Audit Recommendation**: **FEASIBLE & ARCHITECTURALLY READY**.

---

## 5. Storage & Compute Budget for Phase G

| Item | Storage Required | Memory Footprint (RAM) | Compute Runtime (M2 CPU) |
|:---|:---:|:---:|:---:|
| **Task A (Pest Dataset)** | ~45 MB | ~300 MB during training | ~2–4 minutes (transfer learning) |
| **Task B (Lesion Dataset)** | ~60 MB | ~400 MB during training | ~3–5 minutes (U-Net fine-tuning) |
| **Task C (Video Sample & Output)** | ~35 MB | ~250 MB during decoding | ~15–30 seconds |
| **Total Phase G Footprint** | **~140 MB** | **< 1.0 GB Peak** | **< 10 minutes total** |

---

## 6. Strict Data & Model Isolation Guarantees

1. **Frozen Production Model**: `artifacts/checkpoints/best_resnet18_weighted.pt` remains 100% frozen and untouched.
2. **Classification Manifests**: `train.csv`, `val.csv`, and `test.csv` will not be altered.
3. **Dedicated Phase G Directories**:
   - `data/pest_detection/` (new, isolated)
   - `data/lesion_segmentation/` (new, isolated)
   - `data/video/` (new, isolated)
   - `artifacts/checkpoints/pest_detection_real/`
   - `artifacts/checkpoints/segmentation_real/`
   - `artifacts/figures/phase_G/`
4. **No Git Commits/Pushes**: No git operations will be executed.

---

## 7. Immediate Decision Point

In compliance with execution rule **G1**, no datasets have been downloaded and no training has started.

**Awaiting user review and explicit approval on:**
1. Selection of the compact Roboflow Tomato Pest Dataset (~45 MB) over the 22 GB IP102 archive for Task A.
2. Selection of the Roboflow Tomato Lesion Polygon Dataset (~60 MB) for Task B, explicitly rejecting PlantVillage leaf silhouettes as lesion ground truth.
3. Proceeding to **G2 (Real Pest Detection)** and **G3 (Real Lesion Segmentation)** upon your confirmation.
