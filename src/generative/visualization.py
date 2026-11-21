"""Visualization Utilities for Generative Modeling & Latent Analysis (REQ-09).

Generates 4 publication-grade figures in artifacts/figures/generative/:
1. vae_loss_and_kl_curves.png
2. vae_reconstructions_and_samples.png
3. vae_latent_space_and_interpolation.png
4. gan_synthetic_samples_and_loss.png

All synthetic outputs are explicitly labeled as synthetic educational artifacts.

Author: Crop Disease Detection Project Team
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_vae_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot VAE Total ELBO, Reconstruction Loss, and KL Divergence."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)
    epochs = np.arange(1, len(history["train_total"]) + 1)

    axes[0].plot(epochs, history["train_total"], linewidth=2.0, label="Train ELBO")
    axes[0].plot(epochs, history["val_total"], linewidth=2.0, label="Val ELBO")
    axes[0].set_title("Total ELBO Objective (Recon + β·KL)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=8.5)

    axes[1].plot(epochs, history["train_recon"], linewidth=2.0, color="#2ca02c", label="Train Recon MSE")
    axes[1].plot(epochs, history["val_recon"], linewidth=2.0, color="#1f77b4", label="Val Recon MSE")
    axes[1].set_title("Pixel Reconstruction MSE", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Summed MSE / Image")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=8.5)

    axes[2].plot(epochs, history["train_kl"], linewidth=2.0, color="#d62728", label="Train KL Div")
    axes[2].plot(epochs, history["val_kl"], linewidth=2.0, color="#9467bd", label="Val KL Div")
    axes[2].set_title("Latent KL Divergence D_KL(q(z|x) || N(0,I))", fontsize=10.5, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL Penalty")
    axes[2].grid(True, linestyle=":", alpha=0.5)
    axes[2].legend(fontsize=8.5)

    fig.suptitle(
        "Convolutional VAE Training Convergence on Real PlantVillage Subset (Train vs Val)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")

    return fig


def plot_vae_reconstructions_and_prior(
    orig_imgs: torch.Tensor,
    recon_imgs: torch.Tensor,
    prior_samples: torch.Tensor,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot grid comparing Original vs Reconstructed validation leaves and synthetic prior samples."""
    n_show = min(6, orig_imgs.size(0))
    fig, axes = plt.subplots(3, n_show, figsize=(2.4 * n_show, 7.5), dpi=300)

    for i in range(n_show):
        # Row 0: Original Validation Leaf
        orig_np = np.clip(orig_imgs[i].permute(1, 2, 0).cpu().numpy(), 0, 1)
        axes[0, i].imshow(orig_np)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Real Val Leaf", fontsize=10, fontweight="bold", loc="left")

        # Row 1: Reconstructed Leaf
        rec_np = np.clip(recon_imgs[i].permute(1, 2, 0).cpu().numpy(), 0, 1)
        axes[1, i].imshow(rec_np)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("VAE Reconstructed", fontsize=10, fontweight="bold", loc="left")

        # Row 2: Synthetic Prior Sample z ~ N(0, I)
        syn_np = np.clip(prior_samples[i].permute(1, 2, 0).cpu().numpy(), 0, 1)
        axes[2, i].imshow(syn_np)
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title("Synthetic z~N(0,I)", fontsize=10, fontweight="bold", loc="left", color="#d62728")

    fig.suptitle(
        "ConvVAE Behavior: Validation Reconstructions & Synthetic Samples Generated from Latent Prior",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")

    return fig


def plot_latent_manifold_and_interpolation(
    proj_2d: np.ndarray,
    explained_var: np.ndarray,
    diseased_flags: np.ndarray,
    interp_frames: torch.Tensor,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot 2D PCA latent space separation and smooth Healthy -> Diseased interpolation."""
    fig = plt.figure(figsize=(14, 7.5), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0])

    # Top panel: 2D PCA scatter
    ax_pca = fig.add_subplot(gs[0])
    healthy_mask = (diseased_flags == 0)
    diseased_mask = (diseased_flags == 1)

    ax_pca.scatter(
        proj_2d[healthy_mask, 0], proj_2d[healthy_mask, 1],
        c="#2ca02c", alpha=0.8, s=45, edgecolor="k", linewidth=0.5, label="Healthy Foliage (Val)",
    )
    ax_pca.scatter(
        proj_2d[diseased_mask, 0], proj_2d[diseased_mask, 1],
        c="#d62728", alpha=0.7, s=45, marker="^", edgecolor="k", linewidth=0.5, label="Diseased Foliage (Val)",
    )

    ax_pca.set_title(
        f"2D PCA Projection of 16D VAE Latent Space (Var Explained: PC1={explained_var[0]*100:.1f}%, PC2={explained_var[1]*100:.1f}%)",
        fontsize=11,
        fontweight="bold",
    )
    ax_pca.set_xlabel("Latent Principal Component 1")
    ax_pca.set_ylabel("Latent Principal Component 2")
    ax_pca.grid(True, linestyle=":", alpha=0.5)
    ax_pca.legend(fontsize=9.5)

    # Bottom panel: Interpolation Strip
    n_steps = interp_frames.size(0)
    gs_sub = gs[1].subgridspec(1, n_steps, wspace=0.08)
    for idx in range(n_steps):
        ax_img = fig.add_subplot(gs_sub[idx])
        img_np = np.clip(interp_frames[idx].permute(1, 2, 0).cpu().numpy(), 0, 1)
        ax_img.imshow(img_np)
        alpha_val = idx / max(n_steps - 1, 1)
        ax_img.set_title(f"α={alpha_val:.2f}", fontsize=9)
        ax_img.axis("off")

    fig.suptitle(
        "VAE Latent Manifold Analysis: Healthy vs Diseased Embedding & Synthetic Interpolation",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")

    return fig


def plot_gan_results(
    history: Dict[str, List[float]],
    fake_samples: torch.Tensor,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot DCGAN adversarial loss trajectory and fixed-noise synthetic generated grid."""
    fig = plt.figure(figsize=(14, 5.5), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])

    # Left: Adversarial Loss Curves
    ax_loss = fig.add_subplot(gs[0])
    epochs = np.arange(1, len(history["loss_D"]) + 1)
    ax_loss.plot(epochs, history["loss_D"], linewidth=2.0, color="#1f77b4", label="Discriminator Loss D(x)")
    ax_loss.plot(epochs, history["loss_G"], linewidth=2.0, color="#d62728", label="Generator Loss G(z)")
    ax_loss.set_title("DCGAN Alternating Minimax Loss Dynamics", fontsize=11, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCEWithLogits Loss")
    ax_loss.grid(True, linestyle=":", alpha=0.5)
    ax_loss.legend(fontsize=9.5)

    # Right: 3x3 Grid of Synthetic Generated Leaves (rescaled from [-1, 1] to [0, 1])
    gs_grid = gs[1].subgridspec(3, 3, wspace=0.06, hspace=0.06)
    for idx in range(min(9, fake_samples.size(0))):
        r, c = divmod(idx, 3)
        ax_i = fig.add_subplot(gs_grid[r, c])
        img_np = (fake_samples[idx].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
        ax_i.imshow(np.clip(img_np, 0, 1))
        ax_i.axis("off")

    fig.suptitle(
        "Lightweight DCGAN Behavior: Adversarial Training Losses & Synthetic 64x64 Generated Outputs",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")

    return fig

