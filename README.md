# Crop Disease Detection Using Deep Learning

An educational, reproducible deep learning project for plant pathology diagnosis, starting with **Tomato Leaf Disease Classification** using the PlantVillage dataset.

Developed for a two-phase college hackathon, this project emphasizes understanding the core mechanics of Computer Vision and PyTorch—from tensor operations and custom CNN architectures to transfer learning, evaluation metrics, and interactive deployment.

---

## 🎯 Project Objectives

1. **Understand Deep Learning Fundamentals**: Build an intuitive, mathematical, and practical understanding of Convolutional Neural Networks (CNNs), feature extraction, loss functions, and backpropagation.
2. **Robust Data Pipeline**: Implement reliable dataset download, validation, exploratory analysis (EDA), preprocessing, and leakage-free dataset splitting (train/val/test).
3. **Model Development**:
   - **Phase 1**: Custom lightweight CNN baseline trained from scratch to understand architectural decisions (kernel sizes, pooling, activation functions, regularization).
   - **Phase 2**: Transfer learning using established backbones (e.g., MobileNetV2, ResNet18/50) to evaluate performance gains and efficiency on edge devices.
4. **Comprehensive Evaluation**: Analyze confusion matrices, per-class precision/recall/F1-score, and conduct rigorous error analysis on misclassified leaf images.
5. **Interactive Demonstration**: Build a lightweight, local Streamlit web application allowing users to upload leaf photos and receive real-time disease predictions with confidence intervals.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning Framework** | [PyTorch](https://pytorch.org/) & [torchvision](https://pytorch.org/vision/) |
| **Hardware Acceleration** | Apple Silicon MPS (`torch.backends.mps`) with CPU fallback |
| **Image Processing** | Pillow (PIL), torchvision transforms |
| **Data Analysis & Visualization** | NumPy, pandas, Matplotlib, Seaborn |
| **Metrics & Evaluation** | scikit-learn |
| **Interactive UI** | Streamlit (planned for demo phase) |
| **Configuration** | PyYAML |
| **Testing** | pytest |

---

## 💻 Environment Setup (macOS / Apple Silicon)

Follow these steps to set up your isolated Python environment on your MacBook Air M2:

### 1. Prerequisites
Ensure you have Python 3.10 or newer installed:
```bash
python3 --version
```

### 2. Create and Activate Virtual Environment
From the project root (`crop-disease-detection`):
```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```
*(Your terminal prompt should now show `(.venv)`).*

### 3. Upgrade pip and Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Apple Silicon (MPS) Acceleration
Run a quick one-liner to verify that PyTorch detects your M2 GPU:
```bash
python -c "import torch; print('PyTorch Version:', torch.__version__); print('MPS Available (Apple Silicon GPU):', torch.backends.mps.is_available())"
```

---

## 🗺️ Planned Workflow

```mermaid
flowchart LR
    A[Raw PlantVillage Dataset] --> B[Data Inspection & Cleaning]
    B --> C[Stratified Split: Train / Val / Test]
    C --> D[Data Augmentation & Normalization]
    D --> E[Baseline Custom CNN]
    D --> F[Transfer Learning Models]
    E --> G[Evaluation & Error Analysis]
    F --> G
    G --> H[Model Checkpoint Export]
    H --> I[Streamlit Web App]
```

1. **Phase 1: Dataset Exploration & Baseline**
   - Download and curate the Tomato subset of PlantVillage.
   - Run exploratory data analysis (`notebooks/01_dataset_exploration.ipynb`).
   - Implement data loaders with reproducible stratified splits.
   - Train a custom baseline CNN from scratch and track experiments in `reports/experiment_log.csv`.
2. **Phase 2: Transfer Learning & Optimization**
   - Fine-tune pretrained vision models (MobileNet, ResNet).
   - Hyperparameter tuning, regularization (Dropout, weight decay), and learning rate scheduling.
   - Comprehensive error analysis (`reports/model_card.md`).
   - Deploy interactive inference app using Streamlit.

---

## 📂 Project Structure Overview

```text
crop-disease-detection/
├── configs/       # Experiment configuration (YAML)
├── data/          # Raw images, processed sets, and index splits (git-ignored)
├── notebooks/     # Exploratory analysis and scratchpads
├── src/           # Modular, production-ready Python packages
│   ├── data/      # Download, validation, preprocessing, and PyTorch Datasets
│   ├── models/    # CNN and Transfer Learning model architectures
│   ├── training/  # Training and validation loops
│   ├── evaluation/# Metrics calculation, confusion matrices, and plots
│   └── utils/     # Hardware acceleration (MPS), seeds, and logging
├── app/           # Streamlit web application
├── tests/         # Unit and integration tests
├── artifacts/     # Model weights (.pt) and generated figures (git-ignored)
├── reports/       # Dataset cards, model cards, experiment logs
└── docs/          # Architecture blueprints and oral exam (viva) prep
```
