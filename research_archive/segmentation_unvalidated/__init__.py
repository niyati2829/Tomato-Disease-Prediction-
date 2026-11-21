"""Semantic Segmentation Package: U-Net for Leaf Lesion Delineation (REQ-08).

Provides:
- UNet: Full U-Net architecture implemented from scratch in PyTorch.
- Losses & Metrics: DiceLoss, CombinedLoss, mIoU, Dice coefficient, Pixel Accuracy.
- Dataset: Paired image-mask loaders and synthetic lesion benchmark generator.
- Training & Evaluation engines with validation checkpointing.
"""

from src.segmentation.losses_metrics import (
    CombinedLoss,
    DiceLoss,
    calculate_dice_score,
    calculate_iou,
    calculate_pixel_accuracy,
)
from src.segmentation.unet import UNet

__all__ = [
    "UNet",
    "DiceLoss",
    "CombinedLoss",
    "calculate_iou",
    "calculate_dice_score",
    "calculate_pixel_accuracy",
]

