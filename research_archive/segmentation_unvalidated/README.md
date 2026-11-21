# Research Archive: Unvalidated Tomato Lesion Segmentation (Historical)

## 1. Overview & Context

During **Phase D (REQ-08)**, a custom U-Net architecture was implemented from scratch (`unet.py`) to explore semantic segmentation of damaged tomato leaf regions. The architecture incorporates:
- 4-level encoder-decoder hierarchy with double-convolution blocks.
- Max-pooling downsampling and transposed-convolution upsampling.
- Center-crop skip connections to preserve fine spatial details.
- Combined focal Dice and cross-entropy loss function (`losses_metrics.py`).

Because no verified, human-annotated pixel-level tomato lesion dataset was available at the time, the model was trained and evaluated on an educational synthetic lesion benchmark (`dataset.py`).

---

## 2. Synthetic Benchmark Results (Phase D)

- **Input Dimension**: $256 \times 256 \times 3$
- **Target Classes**: 3 (Background, Healthy Foliage, Diseased Lesion)
- **Synthetic Training Set**: 200 synthetic leaves with Perlin-noise lesion textures
- **Synthetic Test Set**: 50 synthetic leaves
- **Metrics Achieved on Synthetic Data**:
  - Mean Intersection-over-Union (mIoU): 0.8124
  - Mean Dice Coefficient: 0.8872
  - Pixel-Level Classification Accuracy: 94.61%

*Note: These metrics reflect performance on mathematically simulated foliar lesions, NOT real-world biological infections.*

---

## 3. Findings of Phase G Botanical Feasibility Audit

During the **Phase G1 Dataset Feasibility and Safety Audit** (`docs/lesion_segmentation_report.md`), all candidate open-access datasets claimed to support tomato leaf disease segmentation were systematically audited:

1. **PlantVillage Segmented (`raw/segmented/`)**:
   - Contains binary silhouettes separating entire leaf perimeters from studio backdrops.
   - Contains **zero lesion-level ground-truth annotations**.
   - Training on these masks teaches background subtraction (leaf perimeter vs background), not disease lesion localization.
2. **Community Bounding-Box Sets (e.g. `thesis-team-ana` / Roboflow)**:
   - Colloquially tagged as "segmentation", but contains exclusively rectangular bounding boxes (`<bndbox>`), lacking pixel contours.
3. **Kaggle / PlantDoc Pseudo-Mask Datasets**:
   - Generated algorithmically using Otsu color thresholding or Segment Anything Model (SAM) prompt grids.
   - Severe false positive rates: leaf venation, leaf serrations, and natural specular reflections are incorrectly marked as lesions.
   - Pseudo-labels fail scientific standards for human expert ground-truth validation.

---

## 4. Academic Integrity & De-Integration Decision

To uphold strict scientific and ethical standards:
1. **No Synthetic Masks in Production**: The synthetic-trained U-Net weights are not executed on real agricultural imagery.
2. **No Whole-Leaf Substitutions**: Binary leaf silhouettes are not misrepresented as lesion segmentations.
3. **De-Integration from Active Application**: The lesion segmentation page, sidebar navigation, capability cards, and inference endpoints have been completely removed from the user-facing Streamlit application.
4. **Code Preservation**: All U-Net models, loss functions, synthetic generators, evaluation scripts, and unit tests are preserved here in `research_archive/segmentation_unvalidated/` and in `src/segmentation/` for academic review. Unit tests (`tests/test_segmentation.py`) continue to verify architectural correctness.

---

## 5. File Inventory

- `unet.py`: Scratch-built PyTorch U-Net with skip connections.
- `losses_metrics.py`: Combined Dice + Cross-Entropy loss and IoU metrics.
- `dataset.py`: Synthetic lesion generator and PyTorch dataset wrapper.
- `train.py`: Synthetic training loop with checkpointing.
- `evaluate.py`: Multi-metric evaluation script.
- `visualization.py`: Saliency and mask overlay tools.
- `unet_training_history.json`: Epoch-wise synthetic loss and metric logs.
- `unet_evaluation_results.json`: Quantitative test results on synthetic data.
- `lesion_segmentation_report.md`: Detailed botanical audit of public datasets.
