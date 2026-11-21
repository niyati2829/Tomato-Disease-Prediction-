# Phase M2 — Improved U-Net Holdout Test Split Evaluation & Benchmark

**Project**: Tomato Disease and Pest Surveillance System  
**Phase**: Phase M2  
**Dataset**: Cleaned PlantSeg v7 Tomato Lesion Subset (`data/processed/tomato_lesions_plantseg/`)  
**Holdout Split**: `splits/test.csv` (Exactly 99 images, strictly untouched during training and model selection)  
**Binary Classification Threshold**: Fixed at $0.50$ (no threshold tuning on test data)

---

## 1. Executive Summary & Core Results

In Phase M2, the improved real U-Net model trained in Phase M1 (`best_unet_improved.pt`) was benchmarked against the original Phase K baseline (`best_unet_real.pt`) on the untouched 99-image holdout test split. Both models were evaluated under identical, deterministic CPU conditions using identical spatial preprocessing (256x256 letterboxed with aspect ratio preservation and ImageNet normalization).

### Comparative Metric Table (99 Holdout Test Images)

| Evaluation Metric | Baseline U-Net (Phase K) | Improved U-Net (Phase M1/M2) | Absolute Delta | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Macro Dice Score** | $42.86\%$ | **$46.60\%$** | **$+3.74\%$** | **$+8.73\%$** |
| **Macro IoU (Jaccard)** | $30.04\%$ | **$33.48\%$** | **$+3.44\%$** | **$+11.45\%$** |
| **Macro Precision** | $37.99\%$ | **$43.90\%$** | **$+5.91\%$** | **$+15.56\%$** |
| **Macro Recall** | $73.36\%$ | $69.85\%$ | $-3.51\%$ | Controlled tradeoff |
| **Pixel Accuracy** | $82.93\%$ | **$85.78\%$** | **$+2.85\%$** | **$+3.44\%$** |
| **Micro Dice Score** | $48.16\%$ | **$50.70\%$** | **$+2.54\%$** | **$+5.27\%$** |
| **Micro IoU** | $31.72\%$ | **$33.95\%$** | **$+2.23\%$** | **$+7.03\%$** |
| **Micro Precision** | $36.75\%$ | **$41.80\%$** | **$+5.05\%$** | **$+13.74\%$** |
| **Micro Recall** | $69.87\%$ | $64.40\%$ | $-5.47\%$ | Preserved sensitivity |
| **False Positive Pixels** | $885,886$ px | **$660,468$ px** | **$-225,418$ px** | **$-25.45\%$ over-segmentation** |
| **CPU Mean Latency** | $48.19$ ms | **$47.21$ ms** | $-0.98$ ms | Identical footprint (21.2 FPS) |

> [!IMPORTANT]
> The primary failure mode of the baseline model—runaway false positives into healthy and chlorotic leaf tissue—was systematically mitigated. Total false positive pixels on the untouched holdout test set dropped by **$225,418$ pixels ($-25.45\%$)**, lifting Macro Precision by **$+5.91\%$** and Macro Dice from $42.86\%$ to **$46.60\%$**.

---

## 2. Confusion Matrix & Global Pixel Analysis

Total evaluated pixels across all 99 test images: $99 \times 256 \times 256 = 6,488,064$ pixels.

| Pixel Category | Baseline (Phase K) | Improved (Phase M1/M2) | Absolute Change | Percentage Shift |
| :--- | :---: | :---: | :---: | :---: |
| **True Positives (TP)** | $514,616$ px | $474,383$ px | $-40,233$ px | $-7.82\%$ |
| **False Positives (FP)** | $885,886$ px | $660,468$ px | **$-225,418$ px** | **$-25.45\%$** |
| **False Negatives (FN)**| $221,961$ px | $262,194$ px | $+40,233$ px | $+18.13\%$ |
| **True Negatives (TN)** | $4,865,601$ px | $5,091,019$ px | **$+225,418$ px** | **$+4.63\%$** |
| **Total Test Pixels** | $6,488,064$ px | $6,488,064$ px | $0$ px | Exact Parity |

### Key Trade-off Analysis:
The baseline network traded 1 true positive pixel for **$1.72$ false positive pixels** ($\text{FP} / \text{TP} = 1.72$). The improved network, regularized by the asymmetric Combined BCE + Tversky loss ($\alpha=0.7, \beta=0.3$) and data augmentations, reduced the $\text{FP} / \text{TP}$ ratio to **$1.39$**, suppressing over $225,000$ erroneous pixels while maintaining over $474,000$ valid lesion detections.

---

## 3. Disease-by-Disease Performance Breakdown

Evaluation across all seven represented tomato disease classes in the holdout split:

| Disease Class | Test Images | Baseline Dice | Improved Dice | Dice Delta | Baseline Precision | Improved Precision | FP Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tomato Late Blight** | 12 | $43.11\%$ | **$52.48\%$** | **$+9.37\%$** | $37.52\%$ | **$50.41\%$** | **$-42.87\%$** |
| **Tomato Early Blight** | 19 | $36.32\%$ | **$43.68\%$** | **$+7.36\%$** | $31.96\%$ | **$40.38\%$** | **$-27.18\%$** |
| **Tomato Mosaic Virus** | 11 | $25.08\%$ | **$26.82\%$** | **$+1.74\%$** | $27.91\%$ | **$31.25\%$** | **$-14.91\%$** |
| **Tomato Leaf Mold** | 14 | $42.46\%$ | **$43.92\%$** | **$+1.45\%$** | $36.78\%$ | **$38.86\%$** | **$-13.09\%$** |
| **Tomato Bacterial Spot**| 18 | $51.11\%$ | **$52.37\%$** | **$+1.25\%$** | $45.19\%$ | **$52.14\%$** | **$-43.80\%$** |
| **Tomato Yellow Leaf Curl**| 13 | $44.85\%$ | **$45.32\%$** | **$+0.46\%$** | $43.43\%$ | **$45.62\%$** | **$-11.92\%$** |
| **Tomato Septoria Spot** | 12 | $51.78\%$ | $50.22\%$ | $-1.56\%$ | $42.12\%$ | **$46.20\%$** | **$-18.84\%$** |

### Insights by Pathology:
1. **Necrotic Leaf Blights (*Late Blight* & *Early Blight*)**: Show the largest performance leaps ($+9.37\%$ and $+7.36\%$ Dice). The baseline model severely over-segmented into yellow chlorotic margins surrounding concentric lesions; the Tversky loss prevented this leakage.
2. **Bacterial Spot**: Precision gained $+6.95\%$ ($45.19\% \to 52.14\%$) with a **$43.80\%$ reduction in false positives**, sharply delineating small dark water-soaked spots.
3. **Septoria Leaf Spot**: Small punctate spots ($<10\text{ px}$) were slightly conservative ($-1.56\%$ Dice), but precision still gained $+4.08\%$ ($42.12\% \to 46.20\%$).
4. **Tomato Mosaic Virus**: Modest gain ($+1.74\%$). Viral symptoms cause diffuse mottling rather than discrete necrosis, making clear ground-truth delineation inherently ambiguous.

---

## 4. Latency & Computational Parity

Both models share identical architecture: `UNet(in_channels=3, num_classes=1, init_features=16, bilinear=True)` with $2.44\text{M}$ parameters ($9.7\text{ MB}$).

Latency was benchmarked under identical CPU execution conditions (10 warmup runs, 99 inference passes on $256 \times 256$ inputs):

| Metric | Baseline (Phase K) | Improved (Phase M1/M2) | Delta |
| :--- | :---: | :---: | :---: |
| **Device** | CPU | CPU | — |
| **Mean Latency** | $48.19$ ms | **$47.21$ ms** | $-0.98$ ms |
| **Median Latency (p50)**| $47.09$ ms | **$46.12$ ms** | $-0.97$ ms |
| **95th Percentile (p95)**| $52.03$ ms | **$51.45$ ms** | $-0.58$ ms |
| **Throughput** | $20.8$ FPS | **$21.2$ FPS** | $+0.4$ FPS |
| **Model Size** | $9.7$ MB | $9.7$ MB | $0.0$ MB |

Both models achieve **$>20$ FPS on standard CPU**, making the improved U-Net fully suitable for real-time edge or local browser deployment.

---

## 5. Visual Overlays & Error Maps

7 representative 6-panel comparison figures were generated in [reports/plantseg_audit/test_prediction_overlays_comparison/](file:///Users/niyatijha/Desktop/crop-disease-detection/reports/plantseg_audit/test_prediction_overlays_comparison/):

Each figure contains:
1. **Input RGB Image (256x256)**: Denormalized ImageNet test sample.
2. **Ground-Truth Mask**: Binary lesion annotation ($Y \in \{0, 1\}$).
3. **Baseline Probability Heatmap**: Raw sigmoid outputs from `best_unet_real.pt`.
4. **Baseline Error Map**: Categorical breakdown (Green=TP, Red=FP, Blue=FN).
5. **Improved Probability Heatmap**: Raw sigmoid outputs from `best_unet_improved.pt`.
6. **Improved Error Map**: Side-by-side categorical error map showing significant reduction in red false-positive regions.

Representative examples:
- [comparison_tomato_late_blight_162.png](file:///Users/niyatijha/Desktop/crop-disease-detection/reports/plantseg_audit/test_prediction_overlays_comparison/comparison_tomato_late_blight_162.png): Drastic reduction in halo false positives; Dice increased substantially.
- [comparison_tomato_early_blight_120.png](file:///Users/niyatijha/Desktop/crop-disease-detection/reports/plantseg_audit/test_prediction_overlays_comparison/comparison_tomato_early_blight_120.png): Sharp boundary adherence around concentric rings.
- [comparison_tomato_bacterial_leaf_spot_102.png](file:///Users/niyatijha/Desktop/crop-disease-detection/reports/plantseg_audit/test_prediction_overlays_comparison/comparison_tomato_bacterial_leaf_spot_102.png): Clean separation of punctate bacterial spots without connecting bridges.

---

## 6. Data Integrity & Verification

1. **Test Set Discipline**:
   - `data/processed/tomato_lesions_plantseg/splits/test.csv` was strictly untouched.
   - Classification threshold fixed at $0.50$ (no post-hoc threshold searching).
2. **Production Checkpoint Integrity**:
   - `artifacts/checkpoints/best_resnet18_weighted.pt` (SHA-256: `b86d...a3fa`) is intact.
   - `artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt` (SHA-256: `01d8...1343`) is intact.
   - Baseline checkpoint `artifacts/checkpoints/segmentation_real/best_unet_real.pt` is intact.
3. **Automated Test Suite**:
   - All 6 tests in `tests/test_phase_m2_evaluation.py` pass.
   - Complete project test suite: **182 / 182 tests passing (100% green)** in 40.88s.

---

## 7. Remaining Limitations & Recommendations

1. **Small Punctate Lesions**: In *Septoria Leaf Spot*, pinpoint lesions (<5 px across) are occasionally missed or fragmented due to $256 \times 256$ downsampling from original $1000 \times 1000+$ resolutions. Feature pyramid networks or patch-based inference could recover higher-resolution details if needed.
2. **Viral Symptoms vs Fungal Necrosis**: Viral diseases (*Tomato Mosaic Virus*, *Yellow Leaf Curl*) exhibit systemic chlorosis rather than well-defined necrotic margins. Lesion segmentation is best coupled with the ResNet18 classifier which captures whole-leaf textural patterns.
3. **Integration Readiness**: `best_unet_improved.pt` is fully verified, non-destructive, and ready for deployment into the interactive Streamlit application.
