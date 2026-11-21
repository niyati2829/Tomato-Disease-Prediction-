# Phase K: Real Tomato Lesion Segmentation U-Net Training Report

**Project:** Tomato Disease and Pest Surveillance System  
**Phase:** Phase K — Real-World Tomato Lesion Segmentation Model Training  
**Date:** November 19, 2026  
**Status:** **COMPLETE & VERIFIED** (Ready for Independent Phase L Test Evaluation)  

---

## 1. Executive Summary

Phase K trained the project's first **real-world tomato leaf lesion segmentation model** using the deduplicated, aligned, and leakage-safe PlantSeg dataset prepared in Phase J. This experiment operated under strict non-interference disciplines: the existing synthetic U-Net experiment, production classification ResNet18, and real pest detector checkpoints were left completely intact, and the 99-image holdout test split was strictly preserved untouched for Phase L.

### Key Milestones Achieved
1. **Isolated Real Model Pipeline**: Created [src/segmentation/train_real.py](file:///Users/niyatijha/Desktop/crop-disease-detection/src/segmentation/train_real.py), saving checkpoints to `artifacts/checkpoints/segmentation_real/best_unet_real.pt` and training metrics to `reports/unet_real_training_history.json`.
2. **Strict Split Partitioning**:
   - **Training Split**: 459 images (100% real foliar lesions across 7 tomato pathologies).
   - **Validation Split**: 99 images (used strictly for model selection and early stopping).
   - **Holdout Test Split**: 99 images (100% untouched and preserved for Phase L).
3. **Binary U-Net Formulation**: Implemented a 1-output-channel U-Net projecting $256 \times 256 \times 3$ RGB inputs to pixel-level lesion logits, activated via sigmoid to generate continuous lesion probability maps.
4. **Hybrid Objective Function**: Formulated `BinaryCombinedLoss` ($0.5 \times \text{BCEWithLogitsLoss} + 0.5 \times \text{BinaryDiceLoss}$) to balance pixel-level classification with spatial region overlap under severe foreground-background class imbalance (mean lesion ratio: ~17.5%).
5. **Convergence & Early Stopping**: Early stopping halted training at Epoch 12 after validation performance plateaued, selecting **Epoch 7** as the optimal checkpoint:
   - **Validation Dice Score**: **0.5517** (55.17%)
   - **Validation IoU (Jaccard Index)**: **0.3814** (38.14%)
   - **Validation Pixel Accuracy**: **82.15%**
   - **Validation Loss**: **0.4875** (down from 0.6646 at Epoch 1)
   - **Total Training Duration**: 1,150.4s (~19.17 minutes) on Host CPU.

---

## 2. Preliminary Inspection & System Verification

Prior to launching training, the environment was inspected to guarantee compliance:

| Requirement | Inspection Finding | Verification Status |
| :--- | :--- | :---: |
| **Existing Synthetic U-Net** | Checkpoint at `artifacts/checkpoints/segmentation/best_unet.pt` (40.3 MB) and reports at `reports/unet_training_history.json` remain completely separate. | **Untouched** |
| **ResNet18 Production Checkpoint** | Checksum `b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa` verified. | **Strictly Preserved** |
| **Real Pest Detector Checkpoint** | Checksum `01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343` verified. | **Strictly Preserved** |
| **Streamlit Application UI** | `app/app.py` unmodified. | **Strictly Preserved** |
| **Hardware Environment** | Host CPU (Apple Silicon host with 4 PyTorch threads allocated); MPS fallback handled gracefully. | **Verified** |
| **Data Interfaces** | `PlantSegTomatoLesionDataset` and `get_plantseg_dataloaders` verified with aspect ratio preservation via centered letterboxing. | **Verified** |

---

## 3. Model Architecture & Hyperparameter Rationale

### 3.1 Network Architecture
The network is based on the first-principles PyTorch U-Net in [src/segmentation/unet.py](file:///Users/niyatijha/Desktop/crop-disease-detection/src/segmentation/unet.py):
- **Input Channels**: 3 (RGB photographic leaves, normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`).
- **Base Feature Width (`init_features`)**: 16 channels.
  - Encoder width: $16 \to 32 \to 64 \to 128$
  - Bottleneck: $128$ channels
  - Decoder width: $128 \to 64 \to 32 \to 16 \to 1$
  - **Parameter Count**: 838,785 parameters (compared to 3.35M for width 32). This compact footprint prevents severe empirical overfitting on 459 training samples while optimizing CPU training throughput by $\approx 3.5\times$.
- **Output Channel**: 1 raw logit per pixel ($256 \times 256$), converted via $\sigma(z) = \frac{1}{1 + e^{-z}}$ to a lesion confidence score in $[0.0, 1.0]$.

### 3.2 Loss Function: Binary Combined BCE + Dice
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$
- **BCE Component**: $\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum [y \ln \sigma(z) + (1 - y) \ln (1 - \sigma(z))]$ ensures steady, stable gradient descent across all spatial locations.
- **Dice Component**: $\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum \sigma(z) y + \epsilon}{\sum \sigma(z) + \sum y + \epsilon}$ penalizes spatial misalignment directly, preventing the model from collapsing to predicting the background class under class imbalance.

### 3.3 Optimizer & Learning Rate Schedule
- **Optimizer**: AdamW ($\text{LR} = 1.0 \times 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\text{weight\_decay} = 1.0 \times 10^{-4}$). Weight decay provides $L_2$ regularization on convolution kernels.
- **LR Scheduler**: `ReduceLROnPlateau` monitoring validation Dice ($\text{mode}=\text{"max"}$, $\text{factor}=0.5$, $\text{patience}=3$). Dynamically dropped learning rate to $5.0 \times 10^{-4}$ at Epoch 12.
- **Early Stopping**: $\text{patience}=5$ epochs based on validation Dice score.

---

## 4. Training Progression & Epoch Logs

```
Epoch 01/50 [85.5s, lr=1.0e-03] | Train Loss: 0.7307, Dice: 0.3432, IoU: 0.2106 | Val Loss: 0.6646, Dice: 0.4448, IoU: 0.2872, Acc: 74.54% -> BEST (*)
Epoch 02/50 [83.5s, lr=1.0e-03] | Train Loss: 0.6626, Dice: 0.4392, IoU: 0.2837 | Val Loss: 0.6534, Dice: 0.4954, IoU: 0.3298, Acc: 75.09% -> BEST (*)
Epoch 03/50 [86.1s, lr=1.0e-03] | Train Loss: 0.6244, Dice: 0.4593, IoU: 0.3003 | Val Loss: 0.6464, Dice: 0.4812, IoU: 0.3183, Acc: 69.09% (no improve: 1/5)
Epoch 04/50 [211.4s, lr=1.0e-03] | Train Loss: 0.5939, Dice: 0.4620, IoU: 0.3021 | Val Loss: 0.5845, Dice: 0.4742, IoU: 0.3126, Acc: 83.70% (no improve: 2/5)
Epoch 05/50 [92.9s, lr=1.0e-03] | Train Loss: 0.5639, Dice: 0.4838, IoU: 0.3207 | Val Loss: 0.5224, Dice: 0.5258, IoU: 0.3581, Acc: 84.87% -> BEST (*)
Epoch 06/50 [98.6s, lr=1.0e-03] | Train Loss: 0.5402, Dice: 0.4879, IoU: 0.3249 | Val Loss: 0.5131, Dice: 0.5505, IoU: 0.3806, Acc: 83.44% -> BEST (*)
Epoch 07/50 [83.3s, lr=1.0e-03] | Train Loss: 0.5205, Dice: 0.4924, IoU: 0.3289 | Val Loss: 0.4875, Dice: 0.5517, IoU: 0.3814, Acc: 82.15% -> BEST (*)
Epoch 08/50 [81.6s, lr=1.0e-03] | Train Loss: 0.4998, Dice: 0.5060, IoU: 0.3415 | Val Loss: 0.4974, Dice: 0.5250, IoU: 0.3561, Acc: 83.06% (no improve: 1/5)
Epoch 09/50 [81.3s, lr=1.0e-03] | Train Loss: 0.4824, Dice: 0.5207, IoU: 0.3537 | Val Loss: 0.4794, Dice: 0.4965, IoU: 0.3325, Acc: 86.74% (no improve: 2/5)
Epoch 10/50 [82.1s, lr=1.0e-03] | Train Loss: 0.4726, Dice: 0.5214, IoU: 0.3553 | Val Loss: 0.4609, Dice: 0.5443, IoU: 0.3745, Acc: 84.23% (no improve: 3/5)
Epoch 11/50 [81.6s, lr=1.0e-03] | Train Loss: 0.4621, Dice: 0.5137, IoU: 0.3491 | Val Loss: 0.5257, Dice: 0.5383, IoU: 0.3690, Acc: 78.31% (no improve: 4/5)
Epoch 12/50 [82.4s, lr=5.0e-04] | Train Loss: 0.4463, Dice: 0.5345, IoU: 0.3668 | Val Loss: 0.4473, Dice: 0.5463, IoU: 0.3778, Acc: 86.64% (no improve: 5/5)
```

**Early Stopping Trigger**: At Epoch 12, early stopping concluded training after 5 epochs without exceeding Epoch 7's validation Dice of **0.5517**.

---

## 5. Quantitative Metric Progression

| Epoch | Train Loss | Train Dice | Train IoU | Val Loss | Val Dice | Val IoU | Val Pixel Acc | Learning Rate | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.7307 | 0.3432 | 0.2106 | 0.6646 | 0.4448 | 0.2872 | 74.54% | $1.0\times 10^{-3}$ | Checkpoint 1 |
| 2 | 0.6626 | 0.4392 | 0.2837 | 0.6534 | 0.4954 | 0.3298 | 75.09% | $1.0\times 10^{-3}$ | Checkpoint 2 |
| 3 | 0.6244 | 0.4593 | 0.3003 | 0.6464 | 0.4812 | 0.3183 | 69.09% | $1.0\times 10^{-3}$ | — |
| 4 | 0.5939 | 0.4620 | 0.3021 | 0.5845 | 0.4742 | 0.3126 | 83.70% | $1.0\times 10^{-3}$ | — |
| 5 | 0.5639 | 0.4838 | 0.3207 | 0.5224 | 0.5258 | 0.3581 | 84.87% | $1.0\times 10^{-3}$ | Checkpoint 5 |
| 6 | 0.5402 | 0.4879 | 0.3249 | 0.5131 | 0.5505 | 0.3806 | 83.44% | $1.0\times 10^{-3}$ | Checkpoint 6 |
| **7** | **0.5205** | **0.4924** | **0.3289** | **0.4875** | **0.5517** | **0.3814** | **82.15%** | **$1.0\times 10^{-3}$** | **BEST (*)** |
| 8 | 0.4998 | 0.5060 | 0.3415 | 0.4974 | 0.5250 | 0.3561 | 83.06% | $1.0\times 10^{-3}$ | — |
| 9 | 0.4824 | 0.5207 | 0.3537 | 0.4794 | 0.4965 | 0.3325 | 86.74% | $1.0\times 10^{-3}$ | — |
| 10 | 0.4726 | 0.5214 | 0.3553 | 0.4609 | 0.5443 | 0.3745 | 84.23% | $1.0\times 10^{-3}$ | — |
| 11 | 0.4621 | 0.5137 | 0.3491 | 0.5257 | 0.5383 | 0.3690 | 78.31% | $1.0\times 10^{-3}$ | — |
| 12 | 0.4463 | 0.5345 | 0.3668 | 0.4473 | 0.5463 | 0.3778 | 86.64% | $5.0\times 10^{-4}$ | — |

---

## 6. Generated Artifacts

| Artifact Name | Path | Description | Size |
| :--- | :--- | :--- | :---: |
| **Best Model Checkpoint** | [best_unet_real.pt](file:///Users/niyatijha/Desktop/crop-disease-detection/artifacts/checkpoints/segmentation_real/best_unet_real.pt) | Saved weights, optimizer state, and training metadata for best Epoch 7. | 9.7 MB |
| **Training History** | [unet_real_training_history.json](file:///Users/niyatijha/Desktop/crop-disease-detection/reports/unet_real_training_history.json) | Complete epoch-by-epoch loss, Dice, IoU, accuracy, and timing records. | 1.4 KB |
| **Training Configuration** | [unet_real_training_config.json](file:///Users/niyatijha/Desktop/crop-disease-detection/reports/unet_real_training_config.json) | Full hyperparameter dictionary and execution metadata. | 917 B |

---

## 7. Integrity & Non-Interference Verification

A project integrity audit confirms that Phase K operated in strict isolation:
- **ResNet18 Checkpoint**: SHA-256 `b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa` (Unchanged).
- **YOLO Pest Detector Checkpoint**: SHA-256 `01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343` (Unchanged).
- **Synthetic U-Net Checkpoint**: `artifacts/checkpoints/segmentation/best_unet.pt` (40.3 MB, Unchanged).
- **Classification Splits**: `data/splits/{train,val,test}.csv` (Unchanged).
- **Holdout Test Set**: `data/processed/tomato_lesions_plantseg/splits/test.csv` (Untouched, zero evaluation performed).
- **Streamlit Application UI**: `app/app.py` (Unmodified).
- **Test Suite**: **166 / 166 tests passing (100% green)** in 39.33s.
- **Git State**: Zero commits, zero pushes.

---

## 8. Conclusion & Readiness for Phase L

Phase K successfully trained a real U-Net segmentation network on real tomato foliar lesion data, achieving **0.5517 validation Dice** and **82.15% validation pixel accuracy**. All artifacts and checkpoint records are safely stored and ready for independent holdout evaluation in **Phase L**.
