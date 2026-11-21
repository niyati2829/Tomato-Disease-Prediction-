"""Loss Functions & Quantitative Evaluation Metrics for Semantic Segmentation (REQ-08).

Formulations:
1. Dice Loss:
       L_Dice = 1 - (2 * sum(P * Y) + smooth) / (sum(P) + sum(Y) + smooth)
2. Intersection over Union (Jaccard Index):
       IoU_c = |P_c cap Y_c| / (|P_c cup Y_c| + eps)
3. Dice Score (Sørensen-Dice Coefficient):
       Dice_c = (2 * |P_c cap Y_c|) / (|P_c| + |Y_c| + eps)
4. Pixel Accuracy:
       Accuracy = (Number of correct pixels) / (Total pixels)

Author: Crop Disease Detection Project Team
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class Dice Loss with Softmax activation."""

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: Predicted raw logits of shape (B, C, H, W).
            targets: Categorical integer targets of shape (B, H, W).

        Returns:
            Scalar loss tensor.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets to (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - torch.mean(dice)
        return loss


class CombinedLoss(nn.Module):
    """Hybrid loss combining Cross-Entropy and Dice Loss."""

    def __init__(self, weight_ce: float = 0.5, smooth: float = 1e-5) -> None:
        super().__init__()
        self.weight_ce = weight_ce
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.weight_ce * loss_ce + (1.0 - self.weight_ce) * loss_dice


def calculate_pixel_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate overall pixel classification accuracy.

    Args:
        predictions: Predicted class indices (B, H, W) or logits (B, C, H, W).
        targets: Ground truth class indices (B, H, W).

    Returns:
        Accuracy as float in [0.0, 1.0].
    """
    if predictions.dim() == 4:
        preds = torch.argmax(predictions, dim=1)
    else:
        preds = predictions

    correct = torch.sum(preds == targets).item()
    total = targets.numel()
    return float(correct / max(total, 1))


def calculate_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3,
    smooth: float = 1e-5,
) -> Tuple[float, Dict[int, float]]:
    """Compute per-class IoU and mean IoU (mIoU).

    Args:
        predictions: Predicted class indices (B, H, W) or logits (B, C, H, W).
        targets: Ground truth class indices (B, H, W).
        num_classes: Total class count.
        smooth: Epsilon to avoid division by zero.

    Returns:
        Tuple of (mean_iou, dict_of_per_class_iou).
    """
    if predictions.dim() == 4:
        preds = torch.argmax(predictions, dim=1)
    else:
        preds = predictions

    per_class_iou = {}
    valid_ious = []

    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)

        intersection = torch.sum(pred_c & target_c).float().item()
        union = torch.sum(pred_c | target_c).float().item()

        if union == 0:
            # Class not present in ground truth or predictions
            iou_c = 1.0 if intersection == 0 else 0.0
        else:
            iou_c = (intersection + smooth) / (union + smooth)

        per_class_iou[c] = float(iou_c)
        valid_ious.append(iou_c)

    mean_iou = float(sum(valid_ious) / len(valid_ious)) if valid_ious else 0.0
    return mean_iou, per_class_iou


def calculate_dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3,
    smooth: float = 1e-5,
) -> Tuple[float, Dict[int, float]]:
    """Compute per-class Dice score and mean Dice score.

    Args:
        predictions: Predicted class indices (B, H, W) or logits (B, C, H, W).
        targets: Ground truth class indices (B, H, W).
        num_classes: Total class count.
        smooth: Epsilon to avoid zero division.

    Returns:
        Tuple of (mean_dice, dict_of_per_class_dice).
    """
    if predictions.dim() == 4:
        preds = torch.argmax(predictions, dim=1)
    else:
        preds = predictions

    per_class_dice = {}
    valid_dices = []

    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)

        intersection = torch.sum(pred_c & target_c).float().item()
        cardinality = (torch.sum(pred_c) + torch.sum(target_c)).float().item()

        if cardinality == 0:
            dice_c = 1.0 if intersection == 0 else 0.0
        else:
            dice_c = (2.0 * intersection + smooth) / (cardinality + smooth)

        per_class_dice[c] = float(dice_c)
        valid_dices.append(dice_c)

    mean_dice = float(sum(valid_dices) / len(valid_dices)) if valid_dices else 0.0
    return mean_dice, per_class_dice


class BinaryDiceLoss(nn.Module):
    """Binary Dice Loss with Sigmoid activation for single-channel lesion segmentation."""

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute binary Dice loss.

        Args:
            logits: Predicted raw logits of shape (B, 1, H, W) or (B, H, W).
            targets: Binary float targets of shape (B, 1, H, W) or (B, H, W) in [0.0, 1.0].
        """
        probs = torch.sigmoid(logits)
        if probs.dim() == 4 and targets.dim() == 3:
            targets = targets.unsqueeze(1)
        elif probs.dim() == 3 and targets.dim() == 4:
            probs = probs.unsqueeze(1)

        dims = tuple(range(2, probs.dim()))
        intersection = torch.sum(probs * targets, dim=dims)
        cardinality = torch.sum(probs + targets, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice)


class BinaryCombinedLoss(nn.Module):
    """Combined BCEWithLogitsLoss and Binary Dice Loss for single-channel lesion segmentation."""

    def __init__(self, weight_bce: float = 0.5, smooth: float = 1e-5, pos_weight: float = None) -> None:
        super().__init__()
        self.weight_bce = weight_bce
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.dice = BinaryDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.weight_bce * loss_bce + (1.0 - self.weight_bce) * loss_dice


def calculate_binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5,
) -> Dict[str, float]:
    """Compute binary Dice score, IoU, and Pixel Accuracy from logits and targets.

    Args:
        logits: (B, 1, H, W) or (B, H, W) raw logits.
        targets: (B, 1, H, W) or (B, H, W) binary targets in [0, 1].
        threshold: Sigmoid probability threshold for binary classification.
        smooth: Epsilon smoothing for Dice / IoU denominators.

    Returns:
        Dictionary containing 'dice', 'iou', and 'pixel_accuracy'.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets >= 0.5).float()

    intersection = torch.sum(preds * targets).item()
    cardinality = (torch.sum(preds) + torch.sum(targets)).item()
    union = (torch.sum(preds) + torch.sum(targets) - intersection).item()

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    iou = (intersection + smooth) / (union + smooth)

    correct_pixels = torch.sum(preds == targets).item()
    total_pixels = targets.numel()
    accuracy = correct_pixels / max(total_pixels, 1)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "pixel_accuracy": float(accuracy),
    }


class TverskyLoss(nn.Module):
    """Tversky Loss for asymmetric false positive / false negative weighting.

    Tversky index:
        TI = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    When alpha > beta (e.g. alpha=0.7, beta=0.3), penalizes False Positives more heavily,
    combating over-segmentation.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-5) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if probs.dim() == 4 and targets.dim() == 3:
            targets = targets.unsqueeze(1)
        elif probs.dim() == 3 and targets.dim() == 4:
            probs = probs.unsqueeze(1)

        dims = tuple(range(2, probs.dim()))
        tp = torch.sum(probs * targets, dim=dims)
        fp = torch.sum(probs * (1.0 - targets), dim=dims)
        fn = torch.sum((1.0 - probs) * targets, dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - torch.mean(tversky)


class CombinedBCETverskyLoss(nn.Module):
    """Combined BCE and Tversky Loss to optimize lesion precision and boundary tightness."""

    def __init__(
        self,
        weight_bce: float = 0.3,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.weight_bce = weight_bce
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_tversky = self.tversky(logits, targets)
        return self.weight_bce * loss_bce + (1.0 - self.weight_bce) * loss_tversky

