# Phase I: Real-World Tuta absoluta Pest Damage Dataset Investigation & Blocker Audit Report

**Project:** Tomato Disease and Pest Surveillance System  
**Phase:** Phase I — Real-World Tuta absoluta Pest Damage Detection Improvement  
**Target Pest:** *Tuta absoluta* (Tomato Leafminer / Tomato Pinworm) foliar feeding damage  
**Date:** November 19, 2026  
**Status:** **AUDIT COMPLETED — DATASET EXPANSION BLOCKED BY UPSTREAM DATASET ACCESS RESTRICTIONS**  

---

## 1. Executive Summary & Audit Mandate

In Phase I, the objective was to improve the existing experimental *Tuta absoluta* foliar feeding-damage detection module using the complete, verified **TomatoEbola** benchmark dataset described in Shehu et al. (2025). 

Per strict Phase I protocol:
> *"If the complete dataset cannot be accessed, document the blocker and stop before training. Do not silently substitute another dataset. Do not invent missing annotations or automatically treat unverified pseudo-labels as ground truth. If any important dataset, annotation or evaluation requirement cannot be verified, stop and report the blocker instead of proceeding with assumptions."*

An exhaustive, multi-source investigation was performed across academic registries, author code repositories, Zenodo records, Roboflow Universe workspaces, and Hugging Face repositories. 

### Core Audit Verdict
- **Kasaisa Farm Subset (122 images, 319 bounding boxes)** is **VERIFIED & ACCESSIBLE** with genuine, human-annotated bounding boxes (CC BY 4.0). This constitutes the current working baseline dataset in `data/pest_detection/real_tuta_absoluta/`.
- **Dikumari Farm Subset (101 images, 217 bounding boxes)** and **Kukareta Farm Subset (103 images, 248 bounding boxes)** are **STRICTLY BLOCKED / RESTRICTED**:
  1. The official Zenodo archive ([Record 13324917](https://zenodo.org/records/13324917), DOI `10.5281/zenodo.13324917`) has its files explicitly marked as **`"access": {"files": "restricted"}`** (`"files": []` via REST API), requiring private author-approved access requests and login.
  2. The authors' official GitHub repository (`harisushehu/YOLO-for-detection-of-tomato-leaf-diseases`) explicitly acknowledges this in its README: *"A sample dataset from one farm in the TomatoEbola dataset has been uploaded... To download the full datasets from the other two farms, please refer to this link (https://zenodo.org/records/13324917)"*.
  3. The authors' secondary repository (`harisushehu/Transformers-for-Tomato-Disease-Detection`) hosts raw JPG photos for Dikumari and Kukareta, but organized exclusively as whole-image classification directories (`HEALTHY` and `INFECTED`) with **ZERO bounding-box annotations**.
  4. The Hugging Face archive (`Project-AgML/tuta_absoluta_tomato_leaf_classification`) provides image-level classification Parquet tables, omitting bounding boxes.
- **Decision**: In strict accordance with user non-negotiable directives, **training has been safely halted**. The baseline model, existing checkpoints, and evaluation results are completely preserved without speculative or unverified substitutions.

---

## 2. Baseline Model & Checkpoint Verification

Before dataset investigation, the existing Phase G baseline detector was audited and verified:

| Parameter | Baseline Value | Verification Method |
| :--- | :--- | :--- |
| **Architecture** | Custom `TinyYOLODetector` (8×8 Grid, 1 anchor/cell) | `src/detection/model.py` |
| **Trained Checkpoint** | `artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt` | File on disk (2.77 MB) |
| **Checkpoint SHA-256** | `01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343` | `shasum -a 256` verified identical |
| **Holdout Test Images** | 12 unseen field images | `data/pest_detection/real_tuta_absoluta/test/images` |
| **Holdout Ground-Truth Boxes** | 29 bounding boxes | `data/pest_detection/real_tuta_absoluta/test/labels` |
| **Overall mAP@0.5** | **30.44%** (0.3044) | `reports/pest_detection_real_evaluation.json` |
| **Overall mAP@0.5:0.95** | **8.79%** (0.0879) | `reports/pest_detection_real_evaluation.json` |
| **Class 0 (Healthy Ref) AP** | 52.22% (0.5222) | Precision: 18.60% \| Recall: 66.67% |
| **Class 1 (Tuta Damage) AP** | 8.66% (0.0866) | Precision: 7.84% \| Recall: 23.53% |
| **CPU Latency / Throughput** | 10.27 ms / image (97.35 FPS) | Apple Silicon / CPU inference |

The baseline checkpoint has been strictly preserved and remained unaltered.

---

## 3. Detailed Investigation of the 3 Farm Subsets (Shehu et al., 2025)

### 3.1 Publication & Metadata Attribution
- **Article:** Shehu, H. A., Ackley, A., Marvellous, M., Eteng, O. E., Sharif, M. H., & Kusetogullari, H. (2025). *"Deep learning-based detection of Tuta absoluta infestation in tomato leaves"*. *Frontiers in Plant Science*, 16, 1524630. DOI: `10.3389/fpls.2025.1524630`.
- **Primary Host DOI:** `10.5281/zenodo.13324917`.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Physical Context:** Daylight outdoor field photography from smallholder farms in Yobe State, Nigeria, capturing *Tuta absoluta* (Tomato Leafminer / "Tomato Ebola") larval leaf-mining damage.

### 3.2 Breakdown of Claimed Subsets vs. Actual Accessibility

| Farm Location | Claimed Images | Claimed Boxes | Average Boxes/Image | Mean Image Resolution | Bounding-Box Availability | Accessibility Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Kasaisa Farm** | 122 | 319 | 2.6 | $667 \times 906$ | **Available** (YOLO `.txt`) | **ACCESSIBLE** (in local repo & authors' GitHub) |
| **2. Dikumari Farm** | 101 | 217 | 2.1 | $1345 \times 1673$ | **Restricted / Missing** | **BLOCKED** (Zenodo files restricted; only raw JPGs on GitHub) |
| **3. Kukareta Farm** | 103 | 248 | 2.4 | $929 \times 943$ | **Restricted / Missing** | **BLOCKED** (Zenodo files restricted; only raw JPGs on GitHub) |
| **Total Benchmark** | **326** | **784** | **2.4** | — | — | **62.6% of annotations inaccessible** |

---

## 4. Evidence of Upstream Data Restrictions (Technical Audit)

### 4.1 Zenodo Record 13324917 REST API Audit
A direct query to the Zenodo REST API (`https://zenodo.org/api/records/13324917`) returned:
```json
{
  "id": 13324917,
  "title": "TomatoEbola dataset",
  "doi": "10.5281/zenodo.13324917",
  "access": {
    "embargo": {"active": false, "reason": null},
    "files": "restricted",
    "record": "public",
    "status": "restricted"
  },
  "files": []
}
```
*Finding:* The record metadata is publicly readable, but the actual file payload is marked `"files": "restricted"`. Public unauthenticated downloads return an empty file list (`[]`).

### 4.2 Zenodo Web Portal Inspection
Visiting `https://zenodo.org/records/13324917` reveals:
> *"The record is publicly accessible, but files are restricted. Log in to check if you have access."*

Users attempting to download files are redirected to an authentication and access-request grant flow requiring manual approval from author Harisu Shehu.

### 4.3 Authors' GitHub Repository Audit
In [harisushehu/YOLO-for-detection-of-tomato-leaf-diseases](https://github.com/harisushehu/YOLO-for-detection-of-tomato-leaf-diseases):
The `TomatoEbola/` directory contains exclusively the Kasaisa subset (`train: 98`, `valid: 12`, `test: 12` = 122 images).
The repository `README.md` explicitly discloses:
> *"A sample dataset from one farm in the TomatoEbola dataset has been uploaded. The TomatoEbola dataset follows this structure... To download the full datasets from the other two farms, please refer to this link (https://zenodo.org/records/13324917)"*

### 4.4 Authors' Classification Repository Audit
In [harisushehu/Transformers-for-Tomato-Disease-Detection](https://github.com/harisushehu/Transformers-for-Tomato-Disease-Detection):
The authors committed folders for `DIKUMARI WARD FARM` (101 JPGs) and `KUKARETA WARD FARM` (103 JPGs). However, inspection confirms these folders contain **raw `.jpg` files only**, partitioned into binary subfolders (`HEALTHY` and `INFECTED`). There are **zero bounding-box annotation files** (`.txt`, `.xml`, `.json`, or `.csv`).

---

## 5. Inspection of Accessible Kasaisa Subset (Quality & Leakage Audit)

The accessible Kasaisa dataset (`data/pest_detection/real_tuta_absoluta/`) was audited using automated scripts:

### 5.1 Split Distribution & Instance Counts
- **Train Split:** 98 images, 262 boxes (116 healthy reference, 146 tuta damage).
- **Validation Split:** 12 images, 28 boxes (10 healthy reference, 18 tuta damage).
- **Test Split (Holdout):** 12 images, 29 boxes (12 healthy reference, 17 tuta damage).
- **Total Images:** 122 images.
- **Total Bounding Boxes:** 319 boxes.
  - Class 0 (`healthy` reference foliar patch): **138 boxes (43.3%)**
  - Class 1 (`tuta_absoluta_damage` foliar mine): **181 boxes (56.7%)**

### 5.2 Bounding Box Validity
- Coordinates verified: normalized $(c_x, c_y, w, h) \in [0.0, 1.0]$.
- Coordinate overflow count: **0 boxes outside valid bounds**.
- Degenerate box count ($w \le 0$ or $h \le 0$): **0 boxes**.

### 5.3 Duplicate and Leakage Audit
- Computed MD5 and SHA-256 hashes across all 122 images.
- Duplicate images detected: **0 duplicates**.
- Split separation: Train, validation, and test partitions contain disjoint image sets with zero image overlap.

---

## 6. Farm-Wise Combination Feasibility Analysis

Can the subsets be combined if Dikumari and Kukareta annotations become accessible in the future?

| Feature | Kasaisa Farm | Dikumari Farm | Kukareta Farm | Combination Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Resolution** | $667 \times 906$ (0.62 MP) | $1345 \times 1673$ (2.24 MP) | $929 \times 943$ (0.87 MP) | Resolvable via letterbox resize to $640 \times 640$ |
| **Aspect Ratio** | 0.74 (Portrait) | 0.80 (Portrait) | 0.99 (Square) | Resolvable via padding/stride |
| **Annotation Density** | 2.6 boxes/image | 2.1 boxes/image | 2.4 boxes/image | Comparable density across farms |
| **Class Imbalance** | 56.7% Tuta / 43.3% Healthy | 46.1% Tuta / 53.9% Healthy | 33.9% Tuta / 66.1% Healthy | Healthy class dominates Kukareta; requires weighted loss |
| **Cross-Farm Generalization** | — | — | — | **Highly desirable**: Training on Kasaisa + Dikumari and holding out Kukareta would provide the first genuine cross-farm out-of-distribution evaluation. |

---

## 7. Compliance with Non-Negotiable Instructions

| Instruction | Compliance Status | Evidence |
| :--- | :---: | :--- |
| *"If the complete dataset cannot be accessed, document the blocker and stop before training."* | **Compliant** | Training halted immediately after Zenodo access restriction was confirmed. |
| *"Do not silently substitute another dataset."* | **Compliant** | No external or unrelated datasets were substituted. |
| *"Do not invent missing annotations or automatically treat unverified pseudo-labels as ground truth."* | **Compliant** | Refused to fabricate pseudo-labels for Dikumari and Kukareta images. |
| *"Do not modify or overwrite existing baseline checkpoint."* | **Compliant** | `best_pest_detector_real.pt` (`01d8...1343`) remains 100% untouched. |
| *"Do not modify production ResNet18 model or its checkpoint."* | **Compliant** | `best_resnet18_weighted.pt` (`b86d...a3fa`) remains 100% untouched. |
| *"Do not modify the Streamlit UI until model evaluation is complete."* | **Compliant** | Streamlit UI preserved in Phase H validated state. |

---

## 8. Summary of Automated Verification Tests

Added regression tests in `tests/test_pest_detection_real.py`:
1. `test_baseline_pest_detector_checkpoint_hash_invariance`: Confirms SHA-256 hash `01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343`.
2. `test_kasaisa_dataset_integrity_and_box_validity`: Verifies exactly 122 images, 319 bounding boxes, and 0 out-of-bounds coordinates.

**Full Test Suite Status:** **153 / 153 tests passing (100%)** across 22 test suites in 8.78 seconds.

---

## 9. Conclusion & Actionable Recommendation

1. **Academic Honesty Verdict**: The current repository represents the maximum achievable verified real-world *Tuta absoluta* detection possible with publicly accessible open-access data.
2. **Next Steps for Dataset Access**: To proceed with multi-farm YOLO training, a formal data request must be submitted to the authors (Harisu Shehu et al., Victoria University of Wellington / NITDA Nigeria) via Zenodo Record 13324917 to grant access to the restricted Dikumari and Kukareta YOLO annotation archives.
3. **Current System Readiness**: The existing application is fully functional with the verified Kasaisa-trained `TinyYOLODetector` baseline properly labeled as experimental research.
