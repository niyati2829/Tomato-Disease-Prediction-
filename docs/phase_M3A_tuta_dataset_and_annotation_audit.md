# Phase M3-A — Tuta absoluta Dataset & Annotation Forensic Audit Report

**Project**: Tomato Disease and Pest Surveillance System  
**Phase**: Phase M3-A  
**Target Pest**: *Tuta absoluta* (Tomato Leafminer / Tomato Pinworm) foliar feeding damage  
**Dataset Under Audit**: TomatoEbola Kasaisa Farm Dataset (`data/pest_detection/real_tuta_absoluta/`)  
**Audit Status**: **AUDIT COMPLETED — DATASET VERIFIED & YOLOv8-COMPATIBLE**

---

## 1. Executive Summary & Audit Mandate

Before starting any YOLOv8 transfer-learning or fine-tuning experiments, an exhaustive forensic audit was performed to inspect all available pest-detection datasets, evaluate bounding-box quality, verify EDEN Library availability, determine whether ordinary disease annotations can be repurposed, and assess experimental feasibility.

### Core Audit Findings:
1. **Kasaisa Farm Dataset (122 images, 319 bounding boxes)**:
   - **Fully verified & accessible** in `data/pest_detection/real_tuta_absoluta/`.
   - **Quality**: Zero coordinate overflows, zero degenerate boxes, zero duplicate images, zero cross-split leakage, and 100% image-label parity (0 orphaned files).
   - **License**: Creative Commons Attribution 4.0 International (CC BY 4.0) via Roboflow and Shehu et al. (2025).
2. **EDEN Library Dataset**:
   - **Not present locally**: An exhaustive search across the repository root, `data/`, `docs/`, `reports/`, and `src/` confirmed that no EDEN Library dataset files or archives exist locally in the project workspace.
   - EDEN Library (`edenlibrary.ai`) is an external agricultural AI platform, but no local files have been downloaded to this repository.
3. **Upstream TomatoEbola Farm Subsets (Dikumari & Kukareta Farms)**:
   - Still restricted on Zenodo (Record 13324917, `"files": "restricted"`).
   - The authors' secondary GitHub repository hosts raw JPGs only for whole-image classification (`HEALTHY` vs `INFECTED`) with **zero bounding-box annotations**.
4. **Biological Specificity & Non-Repurposing of Disease Annotations**:
   - Ordinary disease bounding boxes (e.g., Early Blight, Late Blight, Septoria from PlantVillage or PlantSeg) **cannot and must not be treated as Tuta feeding-damage annotations**.
   - *Tuta absoluta* is an insect pest whose larvae mine blotch-type tunnels between epidermal layers with visible frass; conflating fungal necrotic lesions with pest larval mines would corrupt the biological integrity of the detector.
5. **YOLOv8 Compatibility & Experimental Feasibility**:
   - The Kasaisa dataset is already formatted in standard YOLO format (`class_id x_center y_center width height` normalized) with a valid `data.yaml` pointing to `train`, `valid`, and `test` splits.
   - **Feasibility**: 98 training images (262 boxes) is sufficient for **fine-tuning a pretrained YOLOv8 nano (`yolov8n.pt`)** detector using transfer learning, though insufficient for training a large model from scratch.

---

## 2. Dataset & Annotation Metrics Breakdown

### 2.1 Split Coverage & Image-Label Parity

| Split Partition | Images | Label Files | Orphaned Images | Orphaned Labels | Total Bounding Boxes | Healthy Patches (Class 0) | Tuta Damage (Class 1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 98 | 98 | 0 | 0 | 262 | 116 (44.3%) | 146 (55.7%) |
| **Validation** | 12 | 12 | 0 | 0 | 28 | 10 (35.7%) | 18 (64.3%) |
| **Test (Holdout)** | 12 | 12 | 0 | 0 | 29 | 12 (41.4%) | 17 (58.6%) |
| **Total** | **122** | **122** | **0** | **0** | **319** | **138 (43.3%)** | **181 (56.7%)** |

- **Image Dimensions**: All 122 images are $640 \times 640$ pixels, 3-channel RGB JPEG format (preprocessed via Roboflow stretch with EXIF auto-orientation).
- **Class Balance**: 181 *Tuta absoluta* damage lesions ($56.7\%$) vs 138 healthy foliar reference patches ($43.3\%$).

---

## 3. Bounding-Box Geometry & Annotation Quality

Geometric analysis of all 319 bounding boxes across the dataset:

| Geometric Metric | Mean | Median | Minimum | Maximum | Standard Deviation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normalized Width ($w$)** | $0.3566$ ($228$ px) | $0.3281$ ($210$ px) | $0.0344$ ($22$ px) | $0.9844$ ($630$ px) | $0.1834$ |
| **Normalized Height ($h$)** | $0.3265$ ($209$ px) | $0.2750$ ($176$ px) | $0.0375$ ($24$ px) | $0.9672$ ($619$ px) | $0.1866$ |
| **Area Fraction ($w \times h$)**| $0.1427$ ($14.3\%$) | $0.0902$ ($9.0\%$) | $0.0021$ ($0.2\%$) | $0.6876$ ($68.8\%$) | $0.1378$ |
| **Aspect Ratio ($w / h$)** | $1.3421$ | $1.1341$ | $0.2647$ | $4.7647$ | $0.7512$ |

### Quality Audit Checks:
- **Degenerate Boxes ($w \le 0$ or $h \le 0$)**: **0** (0.0%).
- **Boundary Overflows ($x_{\min} < 0$ or $x_{\max} > 1$)**: **0** (0.0%).
- **Class Label Values**: All labels are strictly in $\{0, 1\}$.
- **Scale Distribution**: Lesions range from small initial punctures ($22 \times 24$ pixels) to extensive larval mines covering up to $68.8\%$ of the leaf area.

---

## 4. Integrity, Uniqueness & Leakage Verification

1. **Intra-Split Duplicate Images**:
   - Computed SHA-256 cryptographic hashes for all 122 image files.
   - **Result**: Exactly 122 unique hashes. Zero duplicate images within train, valid, or test splits.
2. **Cross-Split Data Leakage**:
   - Tested intersection between sets of hashes: $\text{Train} \cap \text{Valid} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Valid} \cap \text{Test} = \emptyset$.
   - **Result**: Zero cross-split leakage. The 12-image holdout test split is strictly isolated.
3. **Class Mapping Consistency**:
   - `data.yaml`: `names: ['healthy', 'infected']` (`nc: 2`).
   - Project code (`src/detection/real_pest_dataset.py`): `REAL_PEST_CLASSES = {0: "healthy", 1: "tuta_absoluta_damage"}`.
   - Both define class 0 as healthy foliar patch and class 1 as Tuta feeding damage.

---

## 5. Visual Inspection Overlays

6 representative visual inspection overlays have been generated in [reports/pest_detection_audit/](file:///Users/niyatijha/Desktop/crop-disease-detection/reports/pest_detection_audit/):
- `audit_overlay_train_DCS_2121-38-_jpg.rf.af05.png` (Train split: multi-box leaf mine with healthy comparison patch)
- `audit_overlay_train_DCS_2121-40-_jpg.rf.198e.png` (Train split: extensive Tuta blotch mine)
- `audit_overlay_valid_DCS_2121-20-_jpg.rf.f295.png` (Validation split: multiple Tuta larval feeding tunnels)
- `audit_overlay_valid_DCS_2121-23-_jpg.rf.844b.png` (Validation split: clear healthy vs infested leaf patches)
- `audit_overlay_test_DCS_2121-1-_jpg.rf.f31a4.png` (Holdout test split: leafminer damage verification)
- `audit_overlay_test_DCS_2121-15-_jpg.rf.ab99.png` (Holdout test split: dual-class reference patch)

Each overlay displays the original RGB image with color-coded bounding boxes (Crimson for *Tuta absoluta* damage, Deep Sky Blue for healthy leaf patch) and pixel dimensions.

---

## 6. Compatibility with YOLOv8 & Existing Pipeline

1. **Directory Structure Parity**:
   The existing directory structure in `data/pest_detection/real_tuta_absoluta/` matches Ultralytics YOLOv8 requirements exactly:
   ```
   real_tuta_absoluta/
   ├── data.yaml
   ├── train/
   │   ├── images/  (98 .jpg)
   │   └── labels/  (98 .txt)
   ├── valid/
   │   ├── images/  (12 .jpg)
   │   └── labels/  (12 .txt)
   └── test/
       ├── images/  (12 .jpg)
       └── labels/  (12 .txt)
   ```
2. **Annotation Format**:
   Each `.txt` file contains standard YOLO annotations:
   $$\langle \text{class\_id} \rangle \quad \langle c_x \rangle \quad \langle c_y \rangle \quad \langle w \rangle \quad \langle h \rangle$$
   Normalized to $[0.0, 1.0]$.
3. **TinyYOLO vs YOLOv8**:
   - Existing custom `TinyYOLODetector`: Custom single-scale $8 \times 8$ grid detector with a single anchor box, which suffered from low capacity ($8.66\%$ Tuta AP in Phase G).
   - `YOLOv8`: Modern anchor-free multi-scale feature pyramid (P3, P4, P5) with Task-Aligned Assigner and CIoU/DFL loss, which is far better equipped to detect overlapping and multi-scale pest mines.

---

## 7. Audit Verdict & Experimental Feasibility

| Question | Verdict | Rationale |
| :--- | :---: | :--- |
| **Is the Kasaisa dataset usable?** | **YES** | 122 images, 319 boxes, 100% verified, clean coordinates, zero leakage. |
| **Are EDEN Library files available?** | **NO** | No EDEN Library files exist locally in the project workspace. |
| **Can disease boxes be repurposed?** | **NO** | Biological violation. Fungal blight lesions are not insect larval mines. |
| **Is available data sufficient for YOLOv8?** | **YES (Transfer Learning)** | 98 training images are sufficient to fine-tune a pretrained `yolov8n.pt` backbone. It is not sufficient to train a detector from scratch without pretraining. |
| **What would be needed to scale further?** | **External Access** | Gaining approved access to Zenodo Record 13324917 (Dikumari & Kukareta farms) to expand dataset to 326 images (+167%). |

---

## 8. Verification & Project Integrity

- **Zero Training Executed**: Strict adherence to the constraint prohibiting premature YOLOv8 training.
- **Checkpoints Preserved**:
  - `best_pest_detector_real.pt` (SHA-256: `01d8...1343`) is intact.
  - `best_resnet18_weighted.pt` (SHA-256: `b86d...a3fa`) is intact.
- **Automated Tests**:
  - `tests/test_phase_m3a_audit.py` ($5/5$ passing).
  - Entire project test suite: **187 / 187 tests green ($100\%$)** in 38.62s.
