"""Semantic Segmentation Package: U-Net for Leaf Lesion Delineation (REQ-08).

Provides:
- UNet: Full U-Net architecture implemented from scratch in PyTorch.
- Losses & Metrics: DiceLoss, CombinedLoss, mIoU, Dice coefficient, Pixel Accuracy.
- Dataset: Paired image-mask loaders and synthetic lesion benchmark generator.
- Training & Evaluation engines with validation checkpointing.
"""

from src.segmentation.evaluate_comparison import run_phase_m2_evaluation
from src.segmentation.losses_metrics import (
    BinaryCombinedLoss,
    BinaryDiceLoss,
    CombinedBCETverskyLoss,
    CombinedLoss,
    DiceLoss,
    TverskyLoss,
    calculate_binary_metrics,
    calculate_dice_score,
    calculate_iou,
    calculate_pixel_accuracy,
)
from src.segmentation.plantseg_dataset import (
    PlantSegTomatoLesionDataset,
    get_plantseg_dataloaders,
)
from src.segmentation.unet import UNet

__all__ = [
    "UNet",
    "DiceLoss",
    "CombinedLoss",
    "BinaryDiceLoss",
    "BinaryCombinedLoss",
    "TverskyLoss",
    "CombinedBCETverskyLoss",
    "calculate_iou",
    "calculate_dice_score",
    "calculate_pixel_accuracy",
    "calculate_binary_metrics",
    "PlantSegTomatoLesionDataset",
    "get_plantseg_dataloaders",
    "run_phase_m2_evaluation",
]

