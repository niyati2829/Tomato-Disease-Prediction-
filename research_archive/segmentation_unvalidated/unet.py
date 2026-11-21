"""U-Net Architecture Implemented from First Principles in PyTorch (REQ-08).

Original Paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
Ronneberger et al., MICCAI 2015.

Key Architectural Blocks:
1. DoubleConv: (Conv3x3 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm -> ReLU)
2. Down: MaxPool2x2 -> DoubleConv
3. Bottleneck: Highest channel capacity representation at lowest spatial resolution
4. Up: ConvTranspose2d / Bilinear Upsample -> Skip Connection Concatenation -> DoubleConv
5. OutConv: 1x1 Conv projecting features to C output segmentation classes

Author: Crop Disease Detection Project Team
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm2d -> ReLU) x 2 with padding=1 to preserve spatial dimensions."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """Downscaling block: MaxPool2d(2) -> DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpconv(x)


class Up(nn.Module):
    """Upscaling block: Upsample -> Concatenate Skip Connection -> DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.

        Args:
            x1: Decoder input from preceding lower layer.
            x2: Encoder skip connection feature map.
        """
        x1 = self.up(x1)

        # Handle potential padding discrepancies due to odd spatial dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffX > 0 or diffY > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Full U-Net Architecture for Leaf Semantic Segmentation.

    Attributes:
        in_channels: Input channels (3 for RGB, 1 for grayscale).
        num_classes: Number of segmentation categories (e.g. 3: Background, Healthy Lamina, Lesion).
        init_features: Base channel width (default: 32 for M2 efficiency, scalable to 64).
        bilinear: Whether to use bilinear upsampling or transposed convolutions.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        init_features: int = 32,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        f = init_features

        # Contracting Path (Encoder)
        self.inc = DoubleConv(in_channels, f)          # e.g. 3 -> 32
        self.down1 = Down(f, f * 2)                     # 32 -> 64
        self.down2 = Down(f * 2, f * 4)                 # 64 -> 128
        self.down3 = Down(f * 4, f * 8)                 # 128 -> 256

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(f * 8, (f * 16) // factor)    # 256 -> 256 (bilinear) or 512

        # Expansive Path (Decoder)
        self.up1 = Up(f * 16, (f * 8) // factor, bilinear)
        self.up2 = Up(f * 8, (f * 4) // factor, bilinear)
        self.up3 = Up(f * 4, (f * 2) // factor, bilinear)
        self.up4 = Up(f * 2, f, bilinear)

        # Final 1x1 Convolution Head
        self.outc = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing pixel-level logits.

        Args:
            x: Tensor of shape (B, in_channels, H, W).

        Returns:
            Logits tensor of shape (B, num_classes, H, W).
        """
        # Encoder with skip features saved
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connection fusion
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

