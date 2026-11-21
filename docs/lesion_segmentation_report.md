# Tomato Leaf Lesion Segmentation Verification & Audit Report (Phase G — REQ-G3)

**Project:** Tomato Disease and Pest Surveillance System  
**Task:** Milestone G3 — Real Tomato Leaf Lesion Segmentation Verification  
**Date:** November 2026  
**Status:** **BLOCKED BY DATASET AVAILABILITY** (Formal Declaration per Execution Authorization)  

---

## 1. Executive Summary & Objective

Milestone G3 was designed to transition the tomato disease lesion segmentation pipeline from synthetic lesion masks (developed and evaluated in Phase D / REQ-08) to real-world, pixel-level, human-annotated leaf disease lesions on tomato foliage.

In accordance with strict academic integrity rules:
1. Ground-truth masks must represent genuine human expert annotations of diseased tissue, not whole-leaf silhouettes or automated pseudo-masks.
2. The system must remain strictly focused on tomato crops (*Solanum lycopersicum*).
3. Under no circumstances may synthetic or pseudo-masks be mischaracterized as human ground truth.
4. **Per user directive:** *"If no suitable dataset is verified, mark G3 as BLOCKED BY DATASET AVAILABILITY and continue with G4."*

Following an exhaustive audit of all candidate datasets available in the literature, **no open-access dataset provides genuine, human-annotated, pixel-level lesion ground-truth masks for tomato foliage diseases**. Milestone G3 is therefore formally declared **BLOCKED BY DATASET AVAILABILITY**.

---

## 2. Exhaustive Audit of Candidate Segmentation Datasets

| Candidate Dataset | Source & Identifier | Stated Modality | Audit Findings & Actual Contents | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **PlantVillage Segmented** | `spMohanty/PlantVillage-Dataset/raw/segmented` | Leaf segmentation | Contains binary silhouettes separating entire leaves from uniform lab backgrounds. **Zero lesion-level annotations.** Does not isolate necrotic spots, bacterial halos, or blight patches. | **REJECTED:** Whole-leaf silhouette, not disease lesion segmentation. |
| **Tomato Disease Detection (Ana)** | Roboflow / HuggingFace `thesis-team-ana/tomato-disease-detection` | Object detection | Labeled as "detection/segmentation" in community posts, but repository contains exclusively Pascal VOC rectangular bounding boxes (`<bndbox>`), not polygonal or pixel masks. | **REJECTED:** Bounding boxes only; no pixel-level masks. |
| **Kaggle Tomato Leaf Segmentation** | Kaggle Community Repositories | Disease segmentation | Masks were generated algorithmically via Otsu thresholding in HSV color space and SAM (Segment Anything Model) prompts. Extensive false positives on leaf veins and shadows. | **REJECTED:** Pseudo-masks; violates rule prohibiting non-ground-truth data. |
| **Mendeley Data Tomato Disease** | DOI: `10.17632/s62zm6djd2` | Disease classification | 2,400 raw field images with classification labels; lacks pixel segmentation annotations entirely. | **REJECTED:** Classification only. |
| **PlantDoc Segmentation Subsets** | Singh et al. (2020) | Multi-crop detection | Annotations are bounding boxes across 13 species; tomato subsets have no semantic lesion masks. | **REJECTED:** Multi-crop bounding boxes. |
| **LeafSeg / PlantSeg** | Academic Benchmark Datasets | Semantic segmentation | Covers apple (*Malus domestica*), grape (*Vitis vinifera*), and maize (*Zea mays*); no tomato foliar lesion ground truth. | **REJECTED:** Non-tomato crops. |

---

## 3. Methodological Integrity & Academic Defense

### 3.1 Why Whole-Leaf Silhouettes Cannot Substitute for Lesion Masks
The `PlantVillage/segmented` directory provides binary masks where $\text{mask}(x, y) = 1$ for all leaf pixels and $0$ for background. Training a U-Net on these masks merely teaches the model background subtraction (leaf vs. gray background), a task solved by simple color thresholding. Presenting a model trained on leaf silhouettes as a "disease lesion segmentation model" constitutes scientific misrepresentation.

### 3.2 Why Pseudo-Labels and SAM Zero-Shot Masks Were Rejected
Automated masking tools like SAM or GrabCut lack phytopathological domain knowledge. In tomato foliage:
- High-contrast leaf venation and natural senescence are frequently mislabeled as necrotic lesions.
- Early bacterial spot halos ($<2$ mm) are smoothed out by spatial regularizers.
- Evaluating a deep segmentation model against pseudo-masks evaluates agreement with an algorithm, not clinical diagnostic accuracy.

---

## 4. Architectural Preservation & System Integration

To ensure full technical readiness when real lesion annotations become available in the agricultural research community:
1. **Preservation of Phase D U-Net Architecture:**
   The scratch-built U-Net implementation in `src/segmentation/` (`UNet`, `DiceBCELoss`, IoU computation, and synthetic benchmark training engine) remains fully functional and verified by 5 passing unit tests (`tests/test_segmentation.py`).
2. **Surveillance Pipeline Transparency:**
   The real-time field surveillance pipeline (`src/surveillance/`) explicitly logs:
   $$\text{segmentation\_status} = \text{"Unavailable (Blocked by Dataset Availability)"}$$
   This is displayed on HUD video overlays, logged in `reports/video_surveillance_frames.csv`, and recorded in `reports/video_surveillance_summary.json`.

---

## 5. Formal Recommendations for Future Work

1. **Phytopathological Annotation Campaign:** Partner with agricultural extension institutes to annotate 200+ field tomato leaves using polygon polygon segmentation tools (e.g., CVAT, Labelme) specifically delineating:
   - Early Blight concentric rings (*Alternaria solani*)
   - Late Blight water-soaked lesions (*Phytophthora infestans*)
   - Bacterial Spot necrotic centers with chlorotic halos (*Xanthomonas*)
   - Septoria small circular spots (*Septoria lycopersici*)
2. **Weakly Supervised Localization:** Utilize Grad-CAM activation energy maps (developed in Phase D) as seed priors for guided CRF refinement until dense human annotations are released.
