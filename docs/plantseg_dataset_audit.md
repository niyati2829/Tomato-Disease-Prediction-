# PlantSeg v7 Dataset Audit Report: Tomato Disease Lesion Segmentation (U-Net)

**Project:** Tomato Disease and Pest Surveillance System  
**Task:** PlantSeg Dataset Audit & U-Net Feasibility Investigation  
**Audit Date:** November 19, 2026  
**Auditor:** Crop Disease Detection Project Team  
**Evaluation Target:** Semantic segmentation of tomato leaf disease lesions using custom U-Net  
**Audit Verdict:** **B. CONDITIONALLY SUITABLE** (Genuine pixel-level lesion annotations verified; requires deduplication, split re-partitioning, and transposed dimension correction prior to model training)  

---

## 1. Executive Summary & Audit Mandate

In Phase G and Phase H, real-world tomato lesion segmentation was paused because all open-access candidates (PlantVillage `raw/segmented`, bounding-box datasets, and automated Otsu/SAM pseudo-masks) lacked genuine human pixel-level lesion annotations.

This audit evaluates the newly acquired **PlantSeg (Version 7)** dataset (`data/raw/PlantSeg/`) to determine whether it provides genuine, verified pixel-level tomato lesion masks suitable for training and evaluating a production U-Net segmentation network.

### Key Audit Findings
1. **Genuine Lesion Annotations Verified**: Unlike PlantVillage silhouettes (which mask entire leaf perimeters), PlantSeg provides **genuine pixel-level lesion contours**. Over 84.9% of tomato masks feature multiple disconnected lesion components (averaging 9.55 distinct lesion spots per image, up to 112 lesions on Septoria / Bacterial Spot).
2. **Tomato Representation**: The dataset contains **667 tomato images** across **7 distinct tomato foliar pathologies** (8.58% of the 7,774-image benchmark).
3. **Image-Mask Correspondence**: All 667 tomato metadata records correspond to existing images and mask files on disk (**100% file presence**, zero missing masks, zero empty/all-black masks).
4. **Data Quality Caveats Identified**:
   - **Transposed Dimensions**: 5 images have transposed dimensions ($W \times H$ swapped) due to EXIF orientation discrepancies.
   - **Duplicate Web Scrapes**: 10 duplicate image files exist, of which 5 cause direct cross-split leakage across the original train, val, and test splits.
   - **Cross-Category Label Ambiguity**: Web scraping caused identical images to be cataloged under conflicting disease labels (e.g., Septoria leaf spot in Training vs. Bacterial leaf spot in Validation).
5. **Final Verdict**: **CONDITIONALLY SUITABLE**. The dataset provides the first verified real-world lesion annotations in the project, but requires deduplication, orientation correction, and fresh leakage-safe split generation before any model training is initiated.

---

## 2. Dataset Identity, Source & Integrity

| Property | Value / Verification Detail |
| :--- | :--- |
| **Dataset Name** | PlantSeg: A Large-Scale In-the-wild Dataset for Plant Disease Segmentation |
| **Version** | Version 7 (v7) |
| **Official Repository** | `https://github.com/tqwei05/PlantSeg` |
| **Origin Location** | `/Users/niyatijha/Downloads/plantseg` (Safely mirrored to `data/raw/PlantSeg/`) |
| **Directory Size** | **1.10 GB** (1,180,483,584 bytes) on disk |
| **License** | Creative Commons Attribution-NonCommercial (**CC BY-NC 4.0**) |
| **Permitted Use** | Non-commercial academic research, educational evaluation, and benchmarking |
| **Total Images in Benchmark** | 7,780 image files (7,774 indexed in `Metadata.csv`) |
| **Total Masks in Benchmark** | 7,780 PNG label mask files |
| **Metadata File** | `Metadata.csv` (1.75 MB, SHA-256: `394fa915...`) |
| **Annotation Manifests** | `annotation_train.json` (47.5 MB), `annotation_test.json` (14.0 MB), `annotation_val.json` (7.9 MB) |

---

## 3. Dataset Folder Hierarchy

The dataset follows a standardized multi-split folder structure:

```
data/raw/PlantSeg/
├── Metadata.csv              # Primary metadata (filename, plant, disease, resolution, split, URL)
├── annotation_train.json     # COCO-format polygon annotations (training split)
├── annotation_val.json       # COCO-format polygon annotations (validation split)
├── annotation_test.json      # COCO-format polygon annotations (test split)
├── images/
│   ├── train/                # 5,369 RGB images (.jpg, .png)
│   ├── val/                  # 848 RGB images (.jpg, .png)
│   └── test/                 # 1,563 RGB images (.jpg, .png)
└── annotations/
    ├── train/                # 5,369 single-channel categorical PNG masks
    ├── val/                  # 848 single-channel categorical PNG masks
    └── test/                 # 1,563 single-channel categorical PNG masks
```

---

## 4. Total Benchmark Statistics

- **Total Host Plant Species**: 34 crops (Wheat: 1,166, Tomato: 667, Soybean: 500, Corn: 419, Citrus: 416, Apple: 402, etc.).
- **Total Unique Disease Categories**: 115 plant diseases.
- **Image Modality**: In-the-wild field and greenhouse photography under diverse natural daylight, complex soil/foliar backgrounds, and varied orientations.
- **Mask Encoding**: 8-bit single-channel PNG images where pixel value `0` represents background/healthy leaf tissue and integer `Index` represents the categorical disease class.

---

## 5. Tomato-Specific Category Statistics

Tomato (*Solanum lycopersicum*) is the **second-largest crop subset** in PlantSeg, comprising **667 images (8.58%)**:

| Disease Category | Class Index | Total Images | Total Masks | Valid Pairs | Transposed Dims | Empty Masks | Mean Lesion Ratio | Train | Val | Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tomato Early Blight** | 97 | 153 | 153 | 153 | 3 | 0 | 10.58% | 103 | 16 | 34 |
| **Tomato Late Blight** | 98 | 117 | 117 | 117 | 0 | 0 | 15.15% | 84 | 14 | 19 |
| **Tomato Leaf Mold** | 99 | 102 | 102 | 102 | 0 | 0 | 16.01% | 71 | 13 | 18 |
| **Tomato Septoria Leaf Spot** | 101 | 99 | 99 | 99 | 1 | 0 | 26.41% | 66 | 13 | 20 |
| **Tomato Bacterial Leaf Spot** | 96 | 90 | 90 | 90 | 1 | 0 | 22.11% | 64 | 9 | 17 |
| **Tomato Yellow Leaf Curl Virus** | 102 | 63 | 63 | 63 | 0 | 0 | 21.05% | 43 | 4 | 16 |
| **Tomato Mosaic Virus** | 100 | 43 | 43 | 43 | 0 | 0 | 16.31% | 29 | 5 | 9 |
| **Total / Summary** | — | **667** | **667** | **667** | **5** | **0** | **17.48%** | **460** | **74** | **133** |

*Summary CSV saved at:* [reports/plantseg_audit/tomato_dataset_summary.csv](file:///Users/niyatijha/Desktop/crop-disease-detection/reports/plantseg_audit/tomato_dataset_summary.csv)

---

## 6. Mask Semantics & Visual Inspection Findings

Visual inspection and connected-component analysis confirmed that PlantSeg masks represent **genuine foliar lesions**:

1. **Pixel Value Distribution**:
   - Background / Healthy foliar tissue: `0`
   - Lesion pixels: Strictly integer equal to the categorical class `Index` (e.g., `97` for Early Blight, `98` for Late Blight).
2. **Disconnected Lesion Topologies**:
   - **84.9% (566 / 667)** of masks contain **multiple disconnected lesion components** (median: 6.0 components, mean: 9.55 components per leaf).
   - Only 15.1% contain a single large necrotic patch.
   - Septoria leaf spot and Bacterial spot exhibit up to **112 minute disconnected necrotic speckles** precisely delineated across the blade.
3. **Lesion Area Ratios**:
   - Minimum lesion coverage: **0.57%** (early subtle pinpoints).
   - Maximum lesion coverage: **80.30%** (severe late-stage blighted necrosis).
   - Mean lesion coverage: **17.48%** (median: 13.10%).
4. **Visual Overlays Generated**:
   - 14 high-resolution 3-panel audit figures (Original Image, Ground-Truth Mask, Blended Overlay) saved under [reports/plantseg_audit/overlays/](file:///Users/niyatijha/Desktop/crop-disease-detection/reports/plantseg_audit/overlays/).
   - Verified that masks tightly hug irregular necrotic halos, concentric target spot rings, and chlorotic patches without spilling into the background or covering healthy green foliage.

---

## 7. Data Quality, Duplicates & Leakage Audit

### 7.1 Image-Mask Dimension Mismatches
Of the 667 pairs, **662 pairs (99.25%)** have exact pixel-for-pixel dimension matches. Exactly **5 pairs** exhibit transposed dimensions ($W \times H$ swapped due to unhandled EXIF rotation):
1. `tomato_bacterial_leaf_spot_42.jpg`: Image $(438 \times 364)$ vs Mask $(364 \times 438)$
2. `tomato_early_blight_google_0043.jpg`: Image $(600 \times 450)$ vs Mask $(450 \times 600)$
3. `tomato_early_blight_google_0000.jpg`: Image $(450 \times 600)$ vs Mask $(600 \times 450)$
4. `tomato_early_blight_google_0056.jpg`: Image $(450 \times 600)$ vs Mask $(600 \times 450)$
5. `tomato_septoria_leaf_spot_google_0033.jpg`: Image $(450 \times 600)$ vs Mask $(600 \times 450)$

*Remediation:* Simple $90^\circ$ transpose alignment (`np.transpose`) resolves all 5 cases without data loss.

### 7.2 Duplicate Web Scrapes & Cross-Split Leakage
MD5 hash auditing discovered **10 duplicate image instances** (representing 657 unique images across 667 files):
- **Cross-Split Leakage**: Exactly **5 duplicate instances** bridge the original training split and the validation/test splits:
  - `tomato_bacterial_leaf_spot_google_0163.jpg` (Test) is identical to `tomato_septoria_leaf_spot_google_0052.jpg` (Training).
  - `tomato_septoria_leaf_spot_google_0205.jpg` (Test) is identical to `tomato_early_blight_google_0260.jpg` (Training).
  - `tomato_yellow_leaf_curl_virus_4.jpg` (Test) is identical to `tomato_mosaic_virus_13.jpg` (Training).
  - `tomato_septoria_leaf_spot_6.jpg` (Training) is identical to `tomato_bacterial_leaf_spot_28.jpg` (Validation).
  - `tomato_yellow_leaf_curl_virus_11.jpg` (Training) is identical to `tomato_mosaic_virus_1.jpg` (Validation).
- **Label Conflict Discrepancy**: Notice that identical scraped leaf photographs were assigned conflicting diagnoses (e.g., Bacterial Spot vs Septoria Spot) by different source websites, and their binary mask IoU is only **0.5008** due to independent human annotation styles.

*Remediation:* The original splits cannot be used as-is. Deduplication and hash-based disjoint splitting must precede training.

---

## 8. U-Net Compatibility & Training Feasibility (MacBook Air M2)

| Engineering Dimension | PlantSeg Characteristics | Compatibility with Existing U-Net Pipeline |
| :--- | :--- | :--- |
| **Input Dimensions** | Variable ($219 \times 294$ to $1280 \times 1600$) | Resize with Bilinear interpolation to $(256 \times 256)$ matches `src/segmentation/unet.py` directly. |
| **Mask Encoding** | Categorical integers (`0` vs `96..102`) | Convert to binary target $(256 \times 256)$ with Nearest-Neighbor interpolation: `target = (mask > 0).float()`. |
| **Segmentation Formulation** | Class-agnostic lesion localization | Highly recommended: Binary segmentation (0: Healthy/Background, 1: Lesion) circumvents the web-scraped label noise between visually overlapping foliar spots. |
| **Class Imbalance** | Mean foreground ratio: 17.48% | Existing combined focal Dice + BCE loss (`src/segmentation/losses_metrics.py`) handles this 1:5 ratio effectively. |
| **Compute Footprint** | ~500 training images after deduplication | Batch size 8 at $256 \times 256$ requires $<450$ MB VRAM on Apple Silicon MPS (`torch.backends.mps`). |
| **Training Duration** | 50 epochs on MacBook Air M2 | Estimated at **3.5 to 5.0 minutes total** on Apple Silicon M2 GPU. |

---

## 9. Final Decision & Justification

### Final Verdict: **B. CONDITIONALLY SUITABLE**

### Supporting Justification
1. **Why NOT "Not Suitable" (Option C)?**
   Unlike PlantVillage `raw/segmented` (which contained zero lesion annotations and only leaf silhouettes), PlantSeg v7 provides **genuine, high-quality pixel-level foliar lesion masks** across 667 tomato images (84.9% multi-lesion). It completely resolves the primary scientific barrier that prevented real lesion segmentation in Phase G.
2. **Why NOT "Suitable" (Option A)?**
   The dataset cannot be piped into a training loop out-of-the-box:
   - 10 duplicate images cause 5 cross-split data leakages between the original train, val, and test partitions.
   - 5 image-mask pairs have transposed dimensions.
   - Conflicting disease labels on duplicate web scrapes mandate class-agnostic binary lesion segmentation rather than fine-grained 7-class segmentation.
3. **Mandatory Conditions Before Phase J Training**:
   - Deduplicate images using perceptual/MD5 hashing.
   - Re-align the 5 transposed image-mask pairs.
   - Formulate training as **binary lesion localization** (Background/Healthy Leaf vs. Disease Lesion).
   - Generate reproducible, leakage-safe train/val/test splits (e.g., 70/15/15).

---

## 10. Final Project Integrity Verification

All existing production checkpoints, manifests, and test suites remain **100% verified, uncorrupted, and preserved**:

| Component | Path | Verified Checksum | Status |
| :--- | :--- | :--- | :---: |
| **Disease Classifier Checkpoint** | `artifacts/checkpoints/best_resnet18_weighted.pt` | `b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa` | **Strictly Preserved** |
| **Pest Detector Checkpoint** | `artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt` | `01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343` | **Strictly Preserved** |
| **Dataset Train Split** | `data/splits/train.csv` | `528898f3c17bd7efb486c027694c15b2be7a1d371780ce00110236dbc6b1af30` | **Strictly Preserved** |
| **Dataset Val Split** | `data/splits/val.csv` | `92c4b22ba874a841cd4dc5ee73bbd475ed9d7826e295c30c34facd7e2981fa70` | **Strictly Preserved** |
| **Dataset Test Split** | `data/splits/test.csv` | `7d152f32dafc55899733b22a2077fbe7ad45a3c0779c73810e99eb20aff0844b` | **Strictly Preserved** |
| **Production UI** | `app/app.py` | Untouched Phase H state | **Strictly Preserved** |
| **Automated Test Suite** | `pytest tests/` | **153 / 153 tests passing (100%)** | **Verified** |
| **Git Status** | Zero commits, zero pushes | Working directory preserved | **Verified** |
