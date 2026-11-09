# Data Directory Guide

This folder manages all local copies of the dataset.

> ⚠️ **CRITICAL: NEVER COMMIT DATASET IMAGES TO GIT**
>
> Image files, ZIP archives, and processed arrays are excluded via `.gitignore`.
> Only `.gitkeep` placeholder files and markdown documentation should be tracked by version control.

---

## 1. Dataset Source

- **Name**: PlantVillage Dataset
- **Repository**: [https://github.com/spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- **Paper**: Hughes, D., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv preprint arXiv:1511.08060.
- **Initial Target Crop**: Tomato (10 classes: 9 disease conditions + 1 healthy class).

---

## 2. License & Ethical Use Verification

- Verify the specific license conditions on the original repository before public release or commercial use.
- The dataset is intended for research, academic benchmarking, and non-commercial educational purposes.
- Maintain proper attribution to original authors and crowd contributors.

---

## 3. Expected Directory Structure

When raw data is downloaded, it will be structured as follows:

```text
data/
├── README.md
├── raw/
│   └── PlantVillage/
│       └── raw/
│           └── color/
│               ├── Tomato___Bacterial_spot/
│               ├── Tomato___Early_blight/
│               ├── Tomato___Late_blight/
│               ├── Tomato___Leaf_Mold/
│               ├── Tomato___Septoria_leaf_spot/
│               ├── Tomato___Spider_mites_Two_spotted_spider_mite/
│               ├── Tomato___Target_Spot/
│               ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│               ├── Tomato___Tomato_mosaic_virus/
│               └── Tomato___healthy/
├── processed/
│   └── [Resized, cleaned, or normalized cache if needed]
└── splits/
    ├── train.csv      # File paths and target class labels for training
    ├── val.csv        # Validation subset
    └── test.csv       # Test subset (kept isolated until final evaluation)
```

---

## 4. Why Use Split Index Files (`splits/*.csv`)?

Instead of copying or duplicating image files into separate `train/`, `val/`, and `test/` folders on disk, we generate CSV index files containing relative filepaths and corresponding integer class labels.

**Benefits:**
- **Zero Disk Waste**: Avoids duplicating gigabytes of image files.
- **Reproducibility**: The exact split is preserved in deterministic CSV files with fixed random seeds.
- **Stratification**: Guarantees equal class proportions across train, validation, and test subsets.
