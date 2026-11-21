"""Convolutional Variational Autoencoder (ConvVAE) Implemented from Scratch (REQ-09).

Mathematical Formulations (Kingma & Welling, 2013):
1. Encoder q_phi(z | x):
       Maps input RGB image x in R^{3 x 64 x 64} to latent Gaussian parameters:
       mean vector mu in R^{d_z} and log-variance vector logvar in R^{d_z}.
2. Reparameterization Trick (differentiable sampling):
       sigma = exp(0.5 * logvar)
       z = mu + sigma * eps,   where eps ~ N(0, I)
3. Decoder p_theta(x | z):
       Maps latent vector z back to reconstructed image x_hat in [0, 1]^{3 x 64 x 64}.
4. Evidence Lower Bound (ELBO) / beta-VAE Loss:
       L_VAE = L_recon(x_hat, x) + beta * L_KL(q(z|x) || N(0, I))
       L_KL  = -0.5 * sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j))

Author: Crop Disease Detection Project Team
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder for 64x64 RGB Tomato Leaf Images."""

    def __init__(self, image_channels: int = 3, latent_dim: int = 16) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.latent_dim = latent_dim

        # Encoder: (B, 3, 64, 64) -> (B, 256, 4, 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),               # 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),              # 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),             # 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten_dim = 256 * 4 * 4  # 4096
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder: latent_dim -> (B, 256, 4, 4) -> (B, 3, 64, 64)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),    # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),     # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),      # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.Sigmoid(),  # Outputs in [0.0, 1.0]
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input image into latent mean and log-variance."""
        h = self.encoder(x)
        h_flat = h.view(x.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = torch.clamp(self.fc_logvar(h_flat), min=-15.0, max=15.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps, where eps ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z to reconstructed RGB image."""
        h = self.fc_decode(z)
        h = h.view(z.size(0), 256, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward pass returning (recon_x, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def sample_prior(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Generate synthetic images by sampling z ~ N(0, I)."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            samples = self.decode(z)
        return samples


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute beta-VAE ELBO loss normalized per batch sample.

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss).
    """
    batch_size = x.size(0)
    # Sum over spatial pixels (C*H*W), average across batch
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size

    # Closed-form KL divergence between N(mu, sigma^2) and N(0, I)
    kl_loss = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - torch.exp(logvar)) / batch_size

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
