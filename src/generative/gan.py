"""Lightweight Deep Convolutional GAN (DCGAN) Architecture (REQ-09).

Generator G(z):   Maps Gaussian noise z in R^{latent_dim} -> RGB image in [-1, 1]^{3 x 64 x 64}.
Discriminator D(x): Maps image x in [-1, 1]^{3 x 64 x 64} -> scalar real/fake logit.

Academic Integrity Note:
GAN training on small subsets (N < 1,000) is inherently unstable and prone to mode collapse
or textural blurring. Generated outputs are purely synthetic demonstrations and must never
be interpreted as valid pathological data or used to alter production benchmarks.

Author: Crop Disease Detection Project Team
"""

import torch
import torch.nn as nn


def weights_init_dcgan(m: nn.Module) -> None:
    """Radford et al. (2015) DCGAN normal weight initialization (std=0.02)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGANGenerator(nn.Module):
    """DCGAN Generator upsampling z (B, latent_dim, 1, 1) -> (B, 3, 64, 64)."""

    def __init__(self, latent_dim: int = 64, ngf: int = 32, image_channels: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # Input: (B, latent_dim, 1, 1) -> (B, ngf*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (B, ngf*8, 4, 4) -> (B, ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (B, ngf*4, 8, 8) -> (B, ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (B, ngf*2, 16, 16) -> (B, ngf, 32, 32)
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (B, ngf, 32, 32) -> (B, 3, 64, 64)
            nn.ConvTranspose2d(ngf, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # Output in [-1.0, 1.0]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator downsampling (B, 3, 64, 64) -> real/fake logit (B, 1)."""

    def __init__(self, ndf: int = 32, image_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # (B, 3, 64, 64) -> (B, ndf, 32, 32)
            nn.Conv2d(image_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, ndf, 32, 32) -> (B, ndf*2, 16, 16)
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, ndf*2, 16, 16) -> (B, ndf*4, 8, 8)
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, ndf*4, 8, 8) -> (B, ndf*8, 4, 4)
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, ndf*8, 4, 4) -> (B, 1, 1, 1)
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(x.size(0), 1)

