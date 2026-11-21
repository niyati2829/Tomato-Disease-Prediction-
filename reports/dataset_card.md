# Dataset Card: PlantVillage (Tomato Subset)

## 1. Dataset Summary
- **Dataset Name**: PlantVillage (Tomato leaf disease subset)
- **Primary Domain**: Agricultural Computer Vision / Plant Pathology
- **Task**: Multi-class Image Classification (10 classes: 9 disease conditions + 1 healthy control)
- **Source Repository**: [https://github.com/spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- **Paper Citation**: Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science, 7, 1419. [DOI: 10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419)

---

## 2. Dataset Origin & Licensing Attribution
- **Authors**: Sharada P. Mohanty, David P. Hughes, Marcel Salathé (EPFL & Penn State University).
- **License Status**: The official GitHub repository does not include an explicit standalone `LICENSE` file. The authors distribute the dataset as an **open-access research repository** intended for academic benchmarking and non-commercial scientific evaluation under citation of their published open-access paper.
- **Attribution Policy**: Requires formal academic citation of the 2016 Frontiers in Plant Science paper and link to the original GitHub repository.

---

## 3. Verified Dataset Composition (EDA Verified)
*Verified via `src/data/explore_dataset.py` across all 18,160 files in `data/raw/PlantVillage-Dataset/raw/color/`.*

| Class Directory Name | Pathogen / Condition | Verified Image Count | Percentage (%) |
|---|---|---|---|
| `Tomato___Bacterial_spot` | Bacterial (*Xanthomonas*) | 2,127 | 11.71% |
| `Tomato___Early_blight` | Fungal (*Alternaria solani*) | 1,000 | 5.51% |
| `Tomato___Late_blight` | Water Mold (*Phytophthora infestans*) | 1,909 | 10.51% |
| `Tomato___Leaf_Mold` | Fungal (*Passalora fulva*) | 952 | 5.24% |
| `Tomato___Septoria_leaf_spot` | Fungal (*Septoria lycopersici*) | 1,771 | 9.75% |
| `Tomato___Spider_mites Two-spotted_spider_mite` | Pest / Mite (*Tetranychus urticae*) | 1,676 | 9.23% |
| `Tomato___Target_Spot` | Fungal (*Corynespora cassiicola*) | 1,404 | 7.73% |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Viral (*Begomovirus*) | 5,357 | 29.50% |
| `Tomato___Tomato_mosaic_virus` | Viral (*Tobamovirus*) | 373 | 2.05% |
| `Tomato___healthy` | Healthy leaf control | 1,591 | 8.76% |
| **Total Tomato Subset** | — | **18,160** | **100.00%** |

---

## 4. Data Quality & Verified Properties
- **Total Images Scanned**: 18,160
- **Corrupt / Unreadable Files**: **0** (all 18,160 images successfully validated using `PIL.Image.verify()`).
- **File Format**: 100% JPEG (`.JPG`).
- **Dimensions (Width x Height)**: Uniform **256 x 256** pixels across all sampled images.
- **Color Mode**: 100% **RGB** (3 channels).
- **Target Training Resolution**: Will be resized to standard **224 x 224** for PyTorch vision models (e.g., MobileNetV2, ResNet, Custom CNN).
- **ImageNet Normalization Values**: Mean = `[0.485, 0.456, 0.406]`, Std = `[0.229, 0.224, 0.225]`.

---

## 5. Class Imbalance Analysis
- **Majority Class**: `Tomato___Tomato_Yellow_Leaf_Curl_Virus` (5,357 images, 29.50%)
- **Minority Class**: `Tomato___Tomato_mosaic_virus` (373 images, 2.05%)
- **Imbalance Ratio**: **14.36x** difference between largest and smallest class.
- **Implications for Phase 1 Modeling**:
  1. **Stratified Splitting is Mandatory**: Random unstratified splits could starve the test or validation sets of minority classes (e.g. Mosaic Virus).
  2. **Accuracy is Insufficient**: A model that simply memorizes the top classes could get high raw accuracy while failing completely on rare viral diseases. Macro-F1 score and confusion matrices are essential.

---

## 6. Verified Stratified Splitting & Manifests (Stage 2 Verified)
- **Train / Validation / Test Ratios**: 70% / 15% / 15% (Random Seed: `42`).
- **Splitting Method**: Two-stage stratified split via `sklearn.model_selection.train_test_split(stratify=labels)`.
- **Manifest Files**: Stored under `data/splits/` referencing relative image paths to prevent disk duplication.
  - `data/splits/train.csv` (12,712 images)
  - `data/splits/val.csv` (2,724 images)
  - `data/splits/test.csv` (2,724 images)
  - `data/splits/class_mapping.json` (Alphabetical class index mapping 0–9)
- **Data Leakage Check**: Mutually exclusive path sets mathematically verified ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- **Full Image Decoding Audit**: All 18,160 images were fully decoded into RAM via `PIL.Image.load()` with **0 unreadable or corrupted files**.

### Per-Class Split Distribution Table

| Index | Class Directory Name | Train Count (%) | Val Count (%) | Test Count (%) | Total Count |
|---|---|---|---|---|---|
| `0` | `Tomato___Bacterial_spot` | 1,489 (11.71%) | 319 (11.71%) | 319 (11.71%) | 2,127 |
| `1` | `Tomato___Early_blight` | 700 (5.51%) | 150 (5.51%) | 150 (5.51%) | 1,000 |
| `2` | `Tomato___Late_blight` | 1,336 (10.51%) | 287 (10.54%) | 286 (10.50%) | 1,909 |
| `3` | `Tomato___Leaf_Mold` | 666 (5.24%) | 143 (5.25%) | 143 (5.25%) | 952 |
| `4` | `Tomato___Septoria_leaf_spot` | 1,240 (9.75%) | 266 (9.77%) | 265 (9.73%) | 1,771 |
| `5` | `Tomato___Spider_mites Two-spotted_spider_mite` | 1,173 (9.23%) | 251 (9.21%) | 252 (9.25%) | 1,676 |
| `6` | `Tomato___Target_Spot` | 983 (7.73%) | 210 (7.71%) | 211 (7.75%) | 1,404 |
| `7` | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 3,750 (29.50%) | 803 (29.48%) | 804 (29.52%) | 5,357 |
| `8` | `Tomato___Tomato_mosaic_virus` | 261 (2.05%) | 56 (2.06%) | 56 (2.06%) | 373 |
| `9` | `Tomato___healthy` | 1,114 (8.76%) | 239 (8.77%) | 238 (8.74%) | 1,591 |
| **Total** | — | **12,712 (100.00%)** | **2,724 (100.00%)** | **2,724 (100.00%)** | **18,160** |

### Preprocessing & Transformation Pipeline
- **Input Resolution**: Resized from native $256 \times 256$ to $224 \times 224$ pixels.
- **Normalization**: ImageNet standard parameters (Channel Mean = `[0.485, 0.456, 0.406]`, Std = `[0.229, 0.224, 0.225]`).
- **Training Transformations**: `Resize(224, 224)`, `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=15)`, `ToTensor()`, `Normalize()`.
- **Validation & Test Transformations**: `Resize(224, 224)`, `ToTensor()`, `Normalize()` (strictly deterministic).
- **DataLoader Configurations**: Default `batch_size = 32`, `num_workers = 0` (configured for Apple Silicon M2 stability), `shuffle = True` for training and `shuffle = False` for evaluation.

---

## 7. Known Limitations & Scientific Biases
- **Controlled Laboratory Backdrops**: All leaves were photographed detached on uniform laboratory gray/black backgrounds. Real-world farm field conditions (soil, weeds, variable sunlight, overlapping leaf clusters) are not represented.
- **Single-Leaf Assumption**: Assumes an isolated leaf with a single dominant disease condition; cannot diagnose multi-pathogen co-infections.
- **Geographic Representation**: Collected from specific geographic regions; local variants or symptoms in different climates or tomato cultivars may look visually distinct.

---

## 8. Verified Visualizations & Artifacts
- **Class Distribution Plot**: Saved to [`artifacts/figures/tomato_class_distribution.png`](../artifacts/figures/tomato_class_distribution.png). Displays ordered class frequencies and percentages highlighting the 14.36x imbalance ratio.
- **Representative Sample Grid**: Saved to [`artifacts/figures/tomato_sample_grid.png`](../artifacts/figures/tomato_sample_grid.png). Displays a 2x5 visual sample grid across all 10 tomato categories (9 disease classes + 1 healthy control) showing symptomatic leaf lesions against lab backgrounds.
- **Integrity Audit**: 18,160 of 18,160 images validated with 0 corrupted files, 100% RGB color mode, and uniform 256x256 native resolution.
