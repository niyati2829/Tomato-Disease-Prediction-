"""Custom Baseline Convolutional Neural Network (CNN) Module.

This module implements a modular, educational PyTorch CNN architecture
from scratch for multi-class tomato leaf disease classification.

Architecture Pipeline:
1. Input: (B, 3, 224, 224) RGB Leaf Tensor
2. Conv Block 1: Conv2d(3 -> 32) + ReLU + MaxPool2d(2) -> (B, 32, 112, 112)
3. Conv Block 2: Conv2d(32 -> 64) + ReLU + MaxPool2d(2) -> (B, 64, 56, 56)
4. Conv Block 3: Conv2d(64 -> 128) + ReLU + MaxPool2d(2) -> (B, 128, 28, 28)
5. Dimension-Safe Adaptive Pooling: AdaptiveAvgPool2d((4, 4)) -> (B, 128, 4, 4)
6. Flatten: (B, 128 * 4 * 4) = (B, 2048)
7. Fully Connected Dense: Linear(2048 -> 256) + ReLU
8. Dropout Regularization: Dropout(p=0.3)
9. Output Linear Layer: Linear(256 -> 10) -> Raw unnormalized logits

Author: Crop Disease Detection Project Team
"""

from typing import Tuple

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Custom 3-block Convolutional Neural Network baseline."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3) -> None:
        """Initialize CNN layers.

        Args:
            num_classes: Number of target classification classes (default 10).
            dropout_rate: Dropout probability for fully connected regularization (default 0.3).
        """
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Convolution Block 1: Low-level spatial features (edges, color gradients)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Convolution Block 2: Mid-level textural features (leaf veins, spots, patches)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Convolution Block 3: High-level disease lesion semantics (necrosis, curling patterns)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Dimension-safe adaptive pooling: guarantees fixed spatial dimensions (4x4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classifier Head
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (Batch_Size, 3, 224, 224).

        Returns:
            Raw unnormalized class logits of shape (Batch_Size, num_classes).
            NOTE: Does not apply Softmax here because torch.nn.CrossEntropyLoss
            internally applies LogSoftmax for numerical stability.
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Calculate total and trainable parameters in a PyTorch model.

    Args:
        model: PyTorch nn.Module.

    Returns:
        Tuple of (total_parameters, trainable_parameters).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # Quick standalone sanity inspection
    model = BaselineCNN(num_classes=10)
    print(model)
    total, trainable = count_parameters(model)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {out.shape}")
