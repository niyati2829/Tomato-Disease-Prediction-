"""Transfer Learning Architectures Module for Tomato Leaf Disease Classification.

Provides factory functions to load, adapt, freeze, and fine-tune torchvision's
pretrained ResNet18 model for 10-class tomato pathology classification.

BatchNorm Behavior & Phase Management:
- Phase A (Feature Extraction):
  * Entire backbone is frozen (requires_grad = False).
  * ALL BatchNorm layers in the backbone are frozen in eval mode (training = False).
  * Their ImageNet running_mean and running_var buffers are strictly preserved and never updated.
  * Only model.fc is trained.
- Phase B (Fine-Tuning):
  * Low- and mid-level feature extractors (conv1, bn1, layer1, layer2, layer3) remain frozen.
  * Their BatchNorm layers remain strictly in eval mode (running statistics untouched).
  * High-level residual block (layer4) and classification head (fc) are trainable.
  * layer4 BatchNorm layers are placed in train mode, adapting batch statistics to leaf pathology.
  * Differential learning rates: 1e-4 for layer4, 1e-3 for fc.

Author: Crop Disease Detection Project Team
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, ResNet


class TomatoResNet18(ResNet):
    """Custom ResNet18 subclass with strict BatchNorm phase isolation."""

    def __init__(
        self,
        num_classes: int = 10,
        weights: models.ResNet18_Weights | None = None,
    ) -> None:
        """Initialize ResNet18 and adapt final fc layer.

        Args:
            num_classes: Target classification categories (default 10).
            weights: Optional torchvision ResNet18 pretrained weights.
        """
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2])
        if weights is not None:
            self.load_state_dict(weights.get_state_dict(progress=True))

        # Replace ImageNet 1000-class head with 10-class Tomato head
        self.fc = nn.Linear(512, num_classes)
        self.phase = "A"  # "A" for feature extraction, "B" for fine-tuning

    def set_phase(self, phase: str) -> None:
        """Configure parameters and BatchNorm states for Phase A or Phase B.

        Args:
            phase: 'A' (feature extraction) or 'B' (fine-tuning).
        """
        if phase.upper() == "A":
            self.phase = "A"
            freeze_backbone(self)
        elif phase.upper() == "B":
            self.phase = "B"
            unfreeze_layer4_and_fc(self)
        else:
            raise ValueError(f"Unknown transfer learning phase: {phase}. Must be 'A' or 'B'.")

    def train(self, mode: bool = True) -> "TomatoResNet18":
        """Set model training mode while guaranteeing frozen BatchNorm layers remain in eval mode.

        Args:
            mode: Whether to set training mode (True) or evaluation mode (False).

        Returns:
            self
        """
        super().train(mode)
        if mode:
            if self.phase == "A":
                # In Phase A: entire backbone is frozen -> all BatchNorm layers must stay in eval mode
                for module in self.modules():
                    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        module.eval()
            elif self.phase == "B":
                # In Phase B: early layers are frozen -> keep conv1/layer1/layer2/layer3 BatchNorm in eval mode
                # layer4 is trainable -> layer4 BatchNorm runs in train mode
                for name, module in self.named_modules():
                    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        if "layer4" not in name:
                            module.eval()
                        else:
                            module.train()
        return self


def get_resnet18(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone_weights: bool = True,
) -> TomatoResNet18:
    """Build and adapt ResNet18 for tomato leaf disease classification.

    Args:
        num_classes: Number of target categories (default 10).
        pretrained: If True, loads ImageNet pretrained weights using torchvision API.
        freeze_backbone_weights: If True, freezes backbone and locks BatchNorm in eval mode (Phase A).

    Returns:
        TomatoResNet18 instance outputting raw logits with managed BatchNorm state.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = TomatoResNet18(num_classes=num_classes, weights=weights)

    if freeze_backbone_weights:
        model.set_phase("A")

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone parameters so only model.fc is trainable (Phase A).

    Also sets all backbone BatchNorm modules to eval mode so running statistics
    (running_mean, running_var) do not drift during forward passes.

    Args:
        model: ResNet model instance.
    """
    if hasattr(model, "phase"):
        model.phase = "A"

    for name, param in model.named_parameters():
        if "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Freeze all BatchNorm modules to protect running statistics
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()


def unfreeze_layer4_and_fc(model: nn.Module) -> None:
    """Unfreeze layer4 and fc for fine-tuning while keeping earlier layers frozen (Phase B).

    Also unfreezes layer4's BatchNorm modules into train mode, while keeping early
    BatchNorm modules (conv1, layer1, layer2, layer3) in eval mode.

    Args:
        model: ResNet model instance.
    """
    if hasattr(model, "phase"):
        model.phase = "B"

    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Keep early BatchNorm frozen in eval mode; let layer4 BatchNorm train
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if "layer4" in name:
                module.train()
            else:
                module.eval()


def get_fine_tuning_param_groups(
    model: nn.Module,
    lr_fc: float = 0.001,
    lr_layer4: float = 0.0001,
    weight_decay: float = 0.0001,
) -> List[Dict[str, any]]:
    """Construct differential learning rate parameter groups for Phase B fine-tuning.

    Args:
        model: ResNet model instance.
        lr_fc: Learning rate for newly added classification layer.
        lr_layer4: Learning rate for unfrozen layer4 backbone.
        weight_decay: L2 regularization penalty.

    Returns:
        List of parameter group dictionaries compatible with torch.optim.Adam.
    """
    layer4_params = [p for p in model.layer4.parameters() if p.requires_grad]
    fc_params = [p for p in model.fc.parameters() if p.requires_grad]

    return [
        {"params": layer4_params, "lr": lr_layer4, "weight_decay": weight_decay},
        {"params": fc_params, "lr": lr_fc, "weight_decay": weight_decay},
    ]


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and active trainable parameters in a model.

    Args:
        model: PyTorch nn.Module.

    Returns:
        Tuple of (total_parameters, trainable_parameters).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
