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
*(To be completed upon training runs)*

- **Model Architectures**:
  - Baseline: Custom Multi-layer CNN (`src/models/cnn.py`)
  - Transfer Learning: Pretrained MobileNetV2 / ResNet18 (`src/models/transfer_learning.py`)
- **Loss Function**: Multi-Class Cross-Entropy Loss (`nn.CrossEntropyLoss`)
- **Optimizer**: Adam / AdamW with Cosine Annealing learning rate schedule
- **Input Dimensions**: `(3, 224, 224)` RGB Tensor, normalized with ImageNet statistics

---

## 4. Evaluation & Performance Metrics
*(To be populated after testing)*

| Metric | Target Baseline | Baseline CNN Achieved | Transfer Learning Achieved |
|---|---|---|---|
| Overall Accuracy | > 85% | TBD | TBD |
| Macro Precision | > 80% | TBD | TBD |
| Macro Recall | > 80% | TBD | TBD |
| Macro F1-Score | > 80% | TBD | TBD |
| Inference Latency (M2 MPS) | < 20 ms | TBD | TBD |

---

## 5. Limitations & Bias
- **Controlled Setting Bias**: High training accuracy on laboratory photos with uniform backgrounds may drop significantly in natural outdoor farm environments.
- **Disease Co-infection**: Model assumes single-label classification; plants suffering from simultaneous multi-pathogen infections will only receive a single prediction.
- **Visual Similarity**: Early-stage symptoms between certain fungal diseases (e.g., Early Blight vs. Target Spot) can have subtle overlap.

---

## 6. Responsible Use & Ethical Considerations
- **Human-in-the-Loop**: Recommendations should always be accompanied by uncertainty scores and verified by qualified agricultural extension officers.
- **Pesticide Safety**: False positives or misdiagnoses could lead to unnecessary or inappropriate chemical applications, impacting local ecosystems and farmer economics.
