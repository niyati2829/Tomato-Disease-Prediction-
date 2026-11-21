"""Generative Deep Learning Package: VAE & DCGAN for Leaf Synthesis (REQ-09).

Provides:
- ConvVAE: Convolutional Variational Autoencoder with Reparameterization Trick & KL Divergence.
- Latent Analysis: 2D PCA manifold projections (Healthy vs Diseased) & smooth interpolations.
- DCGAN: Lightweight Deep Convolutional GAN architecture and stabilized training script.
"""

from src.generative.gan import DCGANDiscriminator, DCGANGenerator
from src.generative.vae import ConvVAE, vae_loss_function

__all__ = [
    "ConvVAE",
    "vae_loss_function",
    "DCGANGenerator",
    "DCGANDiscriminator",
]

