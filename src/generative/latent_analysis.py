"""Latent Space Analysis & Semantic Manifold Interpolation (REQ-09).

Provides:
- Extraction of latent mean vectors mu(x) across validation images.
- 2D Principal Component Analysis (PCA) projecting Healthy vs Diseased manifolds.
- Smooth linear interpolation across latent trajectories:
      z(alpha) = (1 - alpha) * z_start + alpha * z_end,   alpha in [0, 1]

Author: Crop Disease Detection Project Team
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.generative.vae import ConvVAE
from src.generative.vae_dataset import LeafGenerativeDataset


def extract_latent_embeddings(
    model: ConvVAE,
    dataset: LeafGenerativeDataset,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Encode dataset images into deterministic latent mean vectors mu(x)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    mus = []
    diseased_flags = []
    class_names = []

    with torch.no_grad():
        for imgs, flags, names in loader:
            imgs = imgs.to(device)
            mu, _ = model.encode(imgs)
            mus.append(mu.cpu().numpy())
            diseased_flags.extend(flags.numpy().tolist())
            class_names.extend(list(names))

    return np.vstack(mus), np.array(diseased_flags), class_names


def compute_pca_2d(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project high-dimensional latent vectors onto 2 principal components via SVD."""
    mean_vec = np.mean(embeddings, axis=0)
    centered = embeddings - mean_vec
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)

    proj_2d = centered @ Vt[:2].T
    total_var = np.sum(S ** 2)
    explained_ratio = (S[:2] ** 2) / max(total_var, 1e-12)
    return proj_2d, explained_ratio


def interpolate_latent(
    model: ConvVAE,
    img_start: torch.Tensor,
    img_end: torch.Tensor,
    device: torch.device,
    n_steps: int = 8,
) -> torch.Tensor:
    """Linearly interpolate between two images in latent space and decode frames.

    Args:
        model: Trained ConvVAE.
        img_start: Tensor of shape (3, H, W).
        img_end: Tensor of shape (3, H, W).
        device: PyTorch device.
        n_steps: Number of interpolation steps from 0.0 to 1.0 inclusive.

    Returns:
        Decoded image sequence tensor of shape (n_steps, 3, H, W) in [0, 1].
    """
    model.eval()
    with torch.no_grad():
        mu_start, _ = model.encode(img_start.unsqueeze(0).to(device))
        mu_end, _ = model.encode(img_end.unsqueeze(0).to(device))

        alphas = torch.linspace(0.0, 1.0, steps=n_steps, device=device).unsqueeze(1)  # (n_steps, 1)
        z_interp = (1.0 - alphas) * mu_start + alphas * mu_end  # (n_steps, latent_dim)
        decoded = model.decode(z_interp)

    return decoded.cpu()

