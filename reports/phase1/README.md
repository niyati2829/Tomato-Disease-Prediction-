# Phase 1: Deliverables & Evidence Checklist

This checklist tracks the completion of all requirements for **Phase 1** of the college hackathon.

---

## 📋 Phase 1 Milestone Checklist

- [ ] **1. Project Foundation & Environment**
  - [x] Reproducible folder structure created
  - [x] Dependencies isolated in virtual environment (`.venv`)
  - [x] PyTorch MPS (Apple Silicon M2 acceleration) verified
  - [x] Configuration centralized in `configs/config.yaml`
  - [x] Git ignore configured for datasets and checkpoints

- [ ] **2. Data Pipeline & Exploratory Analysis**
  - [ ] PlantVillage tomato subset downloaded locally
  - [ ] Dataset exploration notebook (`notebooks/01_dataset_exploration.ipynb`) completed
  - [ ] Per-class image count and class imbalance visualized
  - [ ] Image dimensions, color distributions, and anomalies inspected
  - [ ] Deterministic, stratified split index files generated (`splits/train.csv`, `splits/val.csv`, `splits/test.csv`)
  - [ ] Custom PyTorch `Dataset` with augmentation and normalization implemented (`src/data/dataset.py`)
  - [ ] Data loader batch generation verified via unit test (`tests/test_dataset.py`)

- [ ] **3. Baseline Model Development**
  - [ ] Lightweight custom CNN architecture designed (`src/models/cnn.py`)
  - [ ] Forward pass tensor shape transformations documented
  - [ ] Model forward pass tested on dummy batch (`tests/test_models.py`)
  - [ ] Parameter count calculated and documented

- [ ] **4. Training Loop & Validation**
  - [ ] Modular training loop implemented (`src/training/train.py`)
  - [ ] Loss tracking per epoch (Cross-Entropy Loss)
  - [ ] Validation loss and accuracy calculated per epoch
  - [ ] Model checkpointing implemented (saves best weights based on validation loss)
  - [ ] Training history plotted (Loss curves and Accuracy curves saved in `artifacts/figures/`)

- [ ] **5. Evaluation & Documentation**
  - [ ] Baseline test set evaluation completed (`src/training/evaluate.py`)
  - [ ] Confusion matrix plotted and saved (`artifacts/figures/baseline_confusion_matrix.png`)
  - [ ] Precision, Recall, and Macro-F1 calculated (`src/evaluation/metrics.py`)
  - [ ] Baseline entry logged in `reports/experiment_log.csv`
  - [ ] `reports/dataset_card.md` populated with actual statistics
  - [ ] Viva questions reviewed in `docs/viva_questions.md`

---

## 📦 Phase 1 Submission Artifacts

When submitting Phase 1, ensure the following are available:
1. `notebooks/01_dataset_exploration.ipynb` (completed with executed visual cells)
2. `artifacts/figures/` (loss curves, class balance bar chart, confusion matrix)
3. `artifacts/checkpoints/baseline_cnn_best.pt` (local checkpoint)
4. `reports/experiment_log.csv` (contains Phase 1 baseline row)
5. `reports/dataset_card.md` (filled with class counts)
