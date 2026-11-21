# Model Card: Tomato Leaf Disease Classifier

## 1. Model Overview
- **Model Name**: Tomato Disease Vision Classifier (Baseline CNN & Transfer Learning Backbones)
- **Model Version**: v0.1.0-draft
- **Model Type**: Deep Convolutional Neural Network (PyTorch)
- **Developer**: College Hackathon Individual Project

---

## 2. Intended Use
- **Primary Use Case**: Diagnostic decision-support tool for classifying single tomato leaf disease symptoms from photographic imagery.
- **Target Audience**: Agronomists, smallholder farmers, agricultural extension workers, and students learning applied computer vision.
- **Out-of-Scope Use Cases**:
  - Direct autonomous chemical/pesticide dispersal without human agricultural expert review.
  - Diagnosis on plant species other than tomato.
  - Multi-leaf canopy or aerial drone imagery diagnosis.

---

## 3. Architecture & Training Details

### Baseline CNN Architecture (`src/models/cnn.py`)
- **Type**: 3-Stage Convolutional Neural Network with Adaptive Pooling and Fully Connected Classifier
- **Input Dimensions**: `(3, 224, 224)` RGB Tensor, normalized with ImageNet statistics
- **Block 1**: `Conv2d(3 -> 32, k=3, p=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)` (Spatial Output: $112 \times 112$)
- **Block 2**: `Conv2d(32 -> 64, k=3, p=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)` (Spatial Output: $56 \times 56$)
- **Block 3**: `Conv2d(64 -> 128, k=3, p=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)` (Spatial Output: $28 \times 28$)
- **Adaptive Pooling**: `AdaptiveAvgPool2d((4, 4))` $\rightarrow$ Feature vector of size $128 \times 4 \times 4 = 2,048$
- **Classifier Head**: `Flatten()` $\rightarrow$ `Linear(2048, 256)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(256, 10)`
- **Output**: 10 raw logits (unnormalized)
- **Total Trainable Parameters**: **620,362** (all parameters trainable)

### ResNet18 Transfer Learning Architecture (`src/models/transfer_learning.py`)
- **Backbone**: Pretrained ResNet18 (`ResNet18_Weights.DEFAULT`) with residual shortcut skip-connections
- **Classification Head**: `model.fc = nn.Linear(512, 10)` outputting raw unnormalized logits
- **Total Parameters**: 11,181,642
- **Phase A (Feature Extraction)**: Backbone frozen; only `fc` (5,130 parameters) trainable with $LR = 0.001$
- **Phase B (Fine-Tuning)**: `layer4` + `fc` (8,401,930 parameters) trainable with differential learning rates ($LR_{\text{layer4}} = 0.0001, LR_{\text{fc}} = 0.001$)
- **Training Strategy**: 5 Epochs Feature Extraction + 5 Epochs Fine-Tuning

### Baseline CNN Training Hyperparameters
- **Loss Function**: `nn.CrossEntropyLoss()`
- **Optimizer**: `Adam(lr=0.001, weight_decay=0.0001)`
- **Batch Size**: `32`
- **Initial Target Epochs**: `10`
- **Compute Device**: Apple Silicon Metal Performance Shaders (`mps`) with automatic CPU fallback
- **Data Augmentation (Training Only)**: `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=15)`

---

## 4. Evaluation & Performance Metrics

### 4.1 Validation Performance Across Candidates
| Metric | Target Baseline | Baseline CNN (Val) | ResNet18 Transfer (Val) | Exp A: Weighted Loss (Val) | Exp B: Weighted+Aug (Val) |
|---|---|---|---|---|---|
| Overall Accuracy | > 85% | **96.18%** (Ep 10) | **98.24%** (Ep 9) | **99.12%** (Ep 3) ★ | **98.53%** (Ep 4) |
| Validation Loss | < 0.40 | **0.1134** (Ep 10) | **0.0572** (Ep 9) | **0.0323** (Ep 3) ★ | **0.0494** (Ep 4) |
| Macro Precision | > 80% | 0.9538 | 0.9799 | **0.9889** | 0.9852 |
| Macro Recall | > 80% | 0.9495 | 0.9753 | **0.9906** | 0.9872 |
| Macro F1-Score | > 80% | 0.9511 | 0.9771 | **0.9897** ★ | 0.9861 |
| Early Blight Recall | — | 85.33% | 85.33% | **99.33%** ★ | 96.67% |
| Best Checkpoint | — | `best_baseline_cnn.pt` | `best_resnet18.pt` | `best_resnet18_weighted.pt` ★ | `best_resnet18_aug_weighted.pt` |
| Training Time | < 30 min | 14.19 min | 12.07 min | 7.56 min | 8.97 min |

### 4.2 Final Holdout Test Set Benchmarking (N = 2,724 images)
Model selection was frozen as **Experiment A (Class-Weighted Loss)** based purely on validation metrics prior to test set evaluation.

| Model Candidate | Test Accuracy | Test Macro F1 | Test Weighted F1 | Early Blight Recall | Parameters | Latency (MPS) | Checkpoint Size |
|---|---|---|---|---|---|---|---|
| **BaselineCNN** | 95.70% | 0.9464 | 0.9567 | 81.33% | **620,362** | **2.07 ms/img** | **7.11 MB** |
| **ResNet18 (Stage 4)** | 98.35% | 0.9749 | 0.9833 | 85.33% | 11,181,642 | 3.75 ms/img | 106.82 MB |
| **Exp A: Weighted Loss (Selected)** ★ | **99.23%** | **0.9885** | **0.9923** | **96.67%** | 11,181,642 | 3.75 ms/img | 106.82 MB |
| **Exp B: Weighted + Aug** | 98.53% | 0.9823 | 0.9852 | 90.00% | 11,181,642 | 3.77 ms/img | 106.83 MB |

**Final Model Highlights (Experiment A on Test Set)**:
- **Test Accuracy**: **99.23%** (2,703 / 2,724 correct, only 21 misclassifications across 10 classes)
- **Macro Precision / Recall / F1**: **0.9873 / 0.9899 / 0.9885**
- **Weighted F1**: **0.9923**
- **Inference Speed**: **3.75 ms/image** on Apple Silicon M2 GPU (Metal Performance Shaders)

---

## 5. Limitations & Bias
- **Controlled Setting Bias**: High training accuracy on laboratory photos with uniform backgrounds may drop significantly in natural outdoor farm environments.
- **Disease Co-infection**: Model assumes single-label classification; plants suffering from simultaneous multi-pathogen infections will only receive a single prediction.
- **Visual Similarity**: Early-stage symptoms between certain fungal diseases (e.g., Early Blight vs. Target Spot) can have subtle overlap.

---

## 6. Responsible Use & Ethical Considerations
- **Human-in-the-Loop**: Recommendations should always be accompanied by uncertainty scores and verified by qualified agricultural extension officers.
- **Pesticide Safety**: False positives or misdiagnoses could lead to unnecessary or inappropriate chemical applications, impacting local ecosystems and farmer economics.
