# Real Tomato Pest Detection Technical Report (Phase G — REQ-G2)

**Project:** Tomato Disease and Pest Surveillance System  
**Task:** Milestone G2 — Real Tomato Pest Detection  
**Target Pest:** *Tuta absoluta* (Tomato Leafminer / Tomato Pinworm)  
**Date:** November 2026  
**Status:** **COMPLETED & VERIFIED** (Holdout Test mAP@0.5: **0.3044**, 97.4 FPS)  

---

## 1. Executive Summary & Objective

In Phase D (REQ-08), an educational single-stage object detector was trained and benchmarked on synthetic foliage sprites to validate bounding-box mechanics, non-maximum suppression (NMS), and Average Precision (AP) metrics. Milestone G2 transitions the surveillance system to **real-world agricultural field imagery** by training, evaluating, and checkpointing an object detector exclusively on genuine tomato plants with human-annotated foliar damage caused by *Tuta absoluta* (Tomato Leafminer), one of the most devastating invasive pests impacting commercial tomato cultivation globally.

---

## 2. Dataset Curation & Attribution

Following the exhaustive dataset feasibility audit conducted in Milestone G1 (`docs/dataset_audit_phase_G.md`), candidate repositories were evaluated for genuine bounding-box annotations, tomato crop exclusivity, and open-access licensing.

- **Primary Source:** Shehu et al. (2025), *"Deep learning-based detection of Tuta absoluta infestation in tomato leaves"*, *Frontiers in Plant Science*, DOI: `10.3389/fpls.2025.1524630`.
- **Open-Access Repository:** Zenodo `13324917` / Roboflow Universe (`nairs/nairs`).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Physical Characteristics:**
  - Image Resolution: $640 \times 640$ pixels, 24-bit RGB.
  - Image Modality: Natural daylight field photography of tomato leaves exhibiting real serpentine leaf mines and tissue necrosis.
  - Annotation Format: Normalized YOLO coordinates (`class_id, cx, cy, w, h`).

### Dataset Partitions and Bounding Box Statistics

| Partition | Images | Label Files | Total Bounding Boxes | Class 0 (`healthy`) | Class 1 (`tuta_absoluta_damage`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 98 | 98 | 262 | 116 | 146 |
| **Validation** | 12 | 12 | 28 | 10 | 18 |
| **Test (Holdout)** | 12 | 12 | 29 | 12 | 17 |
| **Total** | **122** | **122** | **319** | **138** | **181** |

Bounding box dimensions range from minute initial mines ($w=0.034, h=0.032$) to extensive foliar necrosis ($w=0.907, h=0.895$), with mean normalized width $\bar{w}=0.357$ and mean height $\bar{h}=0.326$.

---

## 3. Model Architecture & Loss Formulation

### 3.1 Network Topology: TinyYOLODetector
The detector utilizes an 8-stage convolutional grid architecture:
- **Input Dimension:** $(B, 3, 256, 256)$ with standard ImageNet normalization.
- **Backbone:** 5 sequential downsampling stages (16 $\to$ 32 $\to$ 64 $\to$ 128 $\to$ 256 channels), each comprising `Conv2d(3x3)`, `BatchNorm2d`, `LeakyReLU(0.1)`, and `MaxPool2d(2)`.
- **Grid Resolution:** $S \times S = 8 \times 8 = 64$ spatial anchor cells (effective receptive field: $32 \times 32$ pixels per cell). Spatial collision analysis verified that maximum ground-truth box count per cell across all 122 images is exactly 1.
- **Head:** 2-layer convolutional prediction head (`Conv2d(3x3) -> LeakyReLU -> Conv2d(1x1)`) outputting $S \times S \times (B \cdot 5 + C) = 8 \times 8 \times 7$ tensor ($B=1$ box anchor: $t_x, t_y, t_w, t_h, \text{objectness}$; $C=2$ class logits).

### 3.2 Loss Function
$$\mathcal{L}_{\text{total}} = \lambda_{\text{coord}} \mathcal{L}_{\text{coord}} + \lambda_{\text{obj}} \mathcal{L}_{\text{obj}} + \lambda_{\text{noobj}} \mathcal{L}_{\text{noobj}} + \mathcal{L}_{\text{class}}$$

Where:
- $\lambda_{\text{coord}} = 5.0$, penalizing bounding-box center sigmoid offsets and dimensions via Mean Squared Error (MSE).
- $\lambda_{\text{obj}} = 5.0$, penalizing false negative cell objectness via Binary Cross-Entropy (BCE).
- $\lambda_{\text{noobj}} = 1.0$, stabilizing background cell predictions via BCE.
- $\mathcal{L}_{\text{class}}$ evaluates cross-entropy over the two target classes.

---

## 4. Training Dynamics

- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 10^{-4}$).
- **Learning Rate Schedule:** Cosine Annealing LR starting at $\eta_{\text{max}} = 10^{-3}$ and decaying to $\eta_{\text{min}} = 10^{-5}$ over 35 epochs.
- **Batch Size:** 8 with custom collator handling variable box counts.
- **Device:** Standard CPU / Apple Silicon unified memory.

### Training Trajectory Summary

| Epoch | Train Loss | Validation mAP@0.5 | Learning Rate | Checkpoint Event |
| :---: | :---: | :---: | :---: | :--- |
| **01** | 8.7970 | 0.0083 | 0.000998 | Initial Checkpoint |
| **05** | 4.2750 | 0.1104 | 0.000951 | Checkpointed |
| **10** | 2.6924 | 0.3118 | 0.000814 | Checkpointed |
| **20** | 0.9801 | 0.2842 | 0.000395 | — |
| **30** | 0.4571 | 0.3396 | 0.000059 | Checkpointed |
| **35** | **0.3910** | **0.3460** | **0.000010** | **Best Checkpoint (\*)** |

The final checkpoint was serialized to `artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt`.

---

## 5. Quantitative Holdout Test Evaluation

Evaluation was executed on the untouched holdout `test` split (12 images, 29 ground-truth boxes) using standard COCO/PASCAL VOC 101-point interpolation at IoU threshold $\ge 0.50$.

### 5.1 Detection Metrics

| Metric | Holdout Test Set | Notes |
| :--- | :---: | :--- |
| **mAP@0.50** | **0.3044** (30.4%) | Mean AP across both classes |
| **mAP@0.50:0.95** | **0.0879** (8.8%) | COCO multi-threshold average |
| **Class 0 (`healthy`) AP@0.5** | **0.5222** (52.2%) | Precision: 0.1860, Recall: 0.6667 |
| **Class 1 (`tuta_absoluta`) AP@0.5** | **0.0866** (8.7%) | Precision: 0.0784, Recall: 0.2353 |

### 5.2 Latency and Throughput Profiling

| Hardware Target | Mean Latency (ms) | Std Dev (ms) | Throughput (FPS) |
| :--- | :---: | :---: | :---: |
| **CPU (Apple Silicon M-series)** | **10.27 ms** | 1.46 ms | **97.4 FPS** |

The model achieves near 100 FPS throughput, fulfilling the edge real-time surveillance requirement ($>30$ FPS).

---

## 6. Artifact Inventory

1. **Model Checkpoint:** `artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt`
2. **Evaluation Metrics JSON:** `reports/pest_detection_real_evaluation.json`
3. **Training History JSON:** `reports/pest_detection_real_training_history.json`
4. **Prediction Visualization Figure:** `artifacts/figures/phase_G/pest_detection_real_predictions.png`
5. **Data Loader Implementation:** `src/detection/real_pest_dataset.py`
6. **Training Pipeline:** `src/detection/train_real_pest_detector.py`
7. **Evaluation Pipeline:** `src/detection/evaluate_real_pest_detector.py`
8. **Automated Unit Tests:** `tests/test_pest_detection_real.py` (5 passing tests)
