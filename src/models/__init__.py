"""Model zoo package for Crop Disease Detection."""

from src.models.cnn import BaselineCNN, count_parameters
from src.models.transfer_learning import (
    TomatoResNet18,
    count_trainable_parameters,
    freeze_backbone,
    get_fine_tuning_param_groups,
    get_resnet18,
    unfreeze_layer4_and_fc,
)

__all__ = [
    "BaselineCNN",
    "TomatoResNet18",
    "count_parameters",
    "get_resnet18",
    "freeze_backbone",
    "unfreeze_layer4_and_fc",
    "get_fine_tuning_param_groups",
    "count_trainable_parameters",
]


