# Phase G — Real-World Tomato Surveillance Capability Development

**Project:** Tomato Disease and Pest Surveillance System  
**Phase:** **PHASE G**  
**Date:** November 2026  
**Status:** **COMPLETED & AUDITED** (140/140 Unit Tests Passing)  

---

## 1. Executive Summary & Context

Following the completion and faculty approval of **Phases A through F** (REQ-01 through REQ-10), which established mathematical foundations, neural fundamentals, optimization dynamics, classical learning paradigms, multi-task vision, temporal modeling, generative modeling, and post-hoc calibration, **Phase G** was authorized to bridge the remaining gaps between controlled benchmark evaluation and real-world agricultural field surveillance.

Phase G focused on four key milestones:
1. **G1 — Dataset Feasibility & Safety Audit:** Systematic verification of candidate real-world datasets for tomato pest detection and lesion segmentation.
2. **G2 — Real Tomato Pest Detection:** Training, evaluating, and checkpointing an object detector exclusively on genuine field photography of tomato leaves damaged by *Tuta absoluta* (Tomato Leafminer).
3. **G3 — Lesion Segmentation Dataset Verification:** Rigorous botanical and technical audit of candidate segmentation datasets, culminating in a transparent declaration of blocker status due to the absence of genuine human ground-truth lesion masks in open-access literature.
4. **G4 — Tomato Field-Video Surveillance:** A modular, high-throughput video surveillance pipeline integrating calibrated disease diagnosis ($T^* = 1.3021$), real pest detection, HUD visual overlays, telemetry logging, and temporal analytics.

---

## 2. Milestone Summary & Technical Accomplishments

### Milestone G1: Dataset Feasibility & Safety Audit
- **Audit Findings:** Evaluated candidate datasets (IP102, Roboflow Pesto `tomato-pest-35mqs`, Mendeley Data `s62zm6djd2`, and Shehu et al. 2025).
- **Discrepancy Resolution:** Resolved the public 567 vs. 1,351 image discrepancy on Roboflow Pesto (original 567 raw images expanded via 3x synthetic augmentation). Disqualified IP102 due to prohibitive 22 GB storage overhead and predominance of non-tomato hosts.
- **Selected Real Pest Dataset:** Sourced from peer-reviewed study by Shehu et al. (2025), *Frontiers in Plant Science* (DOI: `10.3389/fpls.2025.1524630`) / Zenodo `13324917`, licensed under CC BY 4.0. Contains 122 field images ($640 \times 640$) with 319 human-annotated bounding boxes across healthy tissue and *Tuta absoluta* foliar mines.
- **Audit Documentation:** Detailed in `docs/dataset_audit_phase_G.md`.

### Milestone G2: Real Tomato Pest Detection
- **Model Topology:** Single-stage `TinyYOLODetector` with 5 convolutional downsampling blocks ($256 \to 8$), $8 \times 8$ grid resolution, 1 box anchor per cell, and 2 target classes (Class 0: `healthy`, Class 1: `tuta_absoluta_damage`). Spatial analysis verified maximum box count per cell is exactly 1.
- **Training Trajectory:** 35 epochs with AdamW, Cosine Annealing learning rate schedule ($\eta = 10^{-3} \to 10^{-5}$), and custom multi-task grid loss.
- **Validation Outcome:** Validation mAP@0.5 climbed from 0.0083 to **0.3460**, checkpointed to `artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt`.
- **Holdout Test Benchmark:**
  - Overall mAP@0.5: **0.3044** (30.4%)
  - Overall mAP@0.5:0.95: **0.0879** (8.8%)
  - Class 0 (`healthy`): AP@0.5 = 0.5222, Precision = 0.1860, Recall = 0.6667
  - Class 1 (`tuta_absoluta_damage`): AP@0.5 = 0.0866, Precision = 0.0784, Recall = 0.2353
  - Inference Latency: **10.27 ms** per image (**97.4 FPS**) on CPU.
- **Technical Report:** Documented in `docs/pest_detection_report.md`.

### Milestone G3: Lesion Segmentation Dataset Verification
- **Audit Findings:** Exhaustive review of candidate datasets revealed that `PlantVillage/segmented` contains binary silhouettes of entire leaves against uniform backgrounds (not diseased lesions), `thesis-team-ana` contains bounding boxes only, and community Kaggle sets use Otsu thresholding or SAM zero-shot pseudo-masks.
- **Compliance with Non-Negotiable Integrity Rules:** In accordance with the mandate prohibiting the fabrication or substitution of pseudo-masks for genuine ground truth, Milestone G3 was formally declared:
  $$\mathbf{BLOCKED\ BY\ DATASET\ AVAILABILITY}$$
- **Architectural Preservation:** The verified U-Net implementation in `src/segmentation/` was preserved. Video overlays and CSV logs explicitly surface this status.
- **Audit Report:** Documented in `docs/lesion_segmentation_report.md`.

### Milestone G4: Tomato Field-Video Surveillance Pipeline
- **Modular Pipeline:** Implemented `src/surveillance/` comprising `VideoProcessor`, `SurveillanceFrameAnalyzer`, `VideoAnnotator`, and `plot_surveillance_timeline`.
- **Safety Boundaries:** Validates container (.mp4, .avi), strictly enforcing duration $<60$ s and file size $<100$ MB.
- **Multi-Task Integration:** Combines calibrated ResNet18 ($T^* = 1.3021$) across 10 tomato diseases with the trained real *Tuta absoluta* detector.
- **Field Benchmark Execution:**
  - Analyzed 150 frames (temporal stride $k=2$, 10-second $640 \times 480$ field video) in **4.43 seconds**.
  - Mean latency: **26.67 ms / frame** (**37.5 FPS**), exceeding real-time requirements ($>30$ FPS).
  - Detected 94 *Tuta absoluta* instances across 41 infested frames (peak 5 pests/frame).
  - Mean calibrated disease diagnostic confidence: **96.15%** (dominant: Tomato Late Blight).
- **Technical Report:** Documented in `docs/video_surveillance_report.md`.

---

## 3. Quantitative Milestone Scorecard

| Milestone | Capability | Evaluation Dataset / Input | Key Metric | Target / Benchmark | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **G1** | Dataset Safety & Audit | Roboflow / Zenodo / PlantVillage | 122 images verified | CC BY 4.0 & GT verified | **Completed** |
| **G2** | Real Pest Detection | Real *Tuta absoluta* Holdout Test | **0.3044 mAP@0.5** | $>0.25$ baseline mAP | **Completed** |
| **G2** | Real Pest Latency | 256x256 image feed | **10.27 ms (97.4 FPS)** | $<33$ ms ($>30$ FPS) | **Completed** |
| **G3** | Lesion Segmentation | Global open-access repositories | Blocker declaration | Academic integrity rule | **Blocked (Declared)** |
| **G4** | Video Surveillance Latency | 640x480 surveillance video | **26.67 ms (37.5 FPS)** | $<33$ ms ($>30$ FPS) | **Completed** |
| **G4** | Video Safety Guardrails | MP4/AVI $<60$s, $<100$MB | 100% boundary checks | Reject non-conforming | **Completed** |
| **System**| Automated Test Suite | `tests/` (21 test suites) | **140 / 140 passing** | 100% green | **Passed** |

---

## 4. Repository Preservation & Quality Assurance

1. **Production ResNet18 Checkpoint:**
   - Path: `artifacts/checkpoints/best_resnet18_weighted.pt`
   - Verified SHA-256: `b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa` (Identical, zero modifications).
2. **Dataset Manifests:**
   - Files: `data/splits/train.csv`, `val.csv`, `test.csv` (100% untouched).
3. **Application Layer:**
   - Streamlit application (`app/app.py`) preserved and running at `http://localhost:8501`.
4. **Git Discipline:**
   - No git commits or git pushes executed.

---

## 5. Artifact Directory Map

```
artifacts/
├── checkpoints/
│   ├── best_resnet18_weighted.pt              # Frozen production classifier (SHA-256: b86d...a3fa)
│   └── pest_detection_real/
│       └── best_pest_detector_real.pt         # Best real Tuta absoluta detector (Val mAP: 0.3460)
├── figures/
│   ├── phase_G/
│   │   └── pest_detection_real_predictions.png# Ground truth vs predicted bounding boxes
│   └── video/
│       └── disease_timeline.png               # Disease probability & pest count timeline
└── video/
    └── annotated_surveillance.mp4             # Processed surveillance video with HUD overlay

reports/
├── pest_detection_real_evaluation.json        # Quantitative holdout test metrics
├── pest_detection_real_training_history.json  # 35-epoch loss and mAP progression
├── video_surveillance_frames.csv              # Frame-by-frame telemetry log (150 frames)
└── video_surveillance_summary.json            # Surveillance execution summary

docs/
├── dataset_audit_phase_G.md                   # Milestone G1 audit report
├── pest_detection_report.md                   # Milestone G2 technical report
├── lesion_segmentation_report.md              # Milestone G3 blocker analysis
├── video_surveillance_report.md               # Milestone G4 technical report
└── phase_G_real_world_surveillance.md         # Complete Phase G synthesis
```
