"""Automated Execution Driver for Generative Deep Learning Demonstrations (REQ-09).

Executes:
1. ConvVAE training on real training subset (data/splits/train.csv).
2. Validation evaluation on data/splits/val.csv (never test.csv).
3. 2D PCA latent space projection & Healthy -> Diseased interpolation.
4. Lightweight DCGAN stabilized training & generation monitoring.
5. Export of 4 figures in artifacts/figures/generative/.

Author: Crop Disease Detection Project Team
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.generative.evaluate_vae import evaluate_vae_checkpoint
from src.generative.gan import DCGANGenerator
from src.generative.latent_analysis import (
    compute_pca_2d,
    extract_latent_embeddings,
    interpolate_latent,
)
from src.generative.train_gan import train_dcgan
from src.generative.train_vae import train_conv_vae
from src.generative.vae import ConvVAE
from src.generative.vae_dataset import LeafGenerativeDataset
from src.generative.visualization import (
    plot_gan_results,
    plot_latent_manifold_and_interpolation,
    plot_vae_reconstructions_and_prior,
    plot_vae_training_curves,
)
from src.utils.device import get_device


def run_all_generative_demonstrations(
    out_dir: Path | str = "artifacts/figures/generative",
    seed: int = 42,
) -> None:
    """Execute end-to-end VAE and DCGAN training, evaluation, and visualizations."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    device = get_device(preferred="mps")

    # 1. Train & Evaluate ConvVAE
    vae_summary = train_conv_vae(
        epochs=25, batch_size=32, latent_dim=16, beta=1.0, lr=1e-3, seed=seed
    )
    _ = evaluate_vae_checkpoint(seed=seed)

    fig1 = plot_vae_training_curves(
        vae_summary["history"], save_path=out_p / "vae_loss_and_kl_curves.png"
    )
    plt.close(fig1)

    # 2. Load best VAE checkpoint for reconstructions, prior sampling & latent analysis
    vae_ckpt = torch.load("artifacts/checkpoints/generative/best_vae.pt", map_location=device)
    vae = ConvVAE(image_channels=3, latent_dim=vae_ckpt.get("latent_dim", 16)).to(device)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.eval()

    val_ds = LeafGenerativeDataset("data/splits/val.csv", image_size=64, max_per_class=40, seed=seed)

    # Reconstruct first 6 validation samples
    orig_imgs = torch.stack([val_ds[i][0] for i in range(6)])
    with torch.no_grad():
        mu, _ = vae.encode(orig_imgs.to(device))
        recon_imgs = vae.decode(mu).cpu()
    prior_samples = vae.sample_prior(n_samples=6, device=device).cpu()

    fig2 = plot_vae_reconstructions_and_prior(
        orig_imgs, recon_imgs, prior_samples, save_path=out_p / "vae_reconstructions_and_samples.png"
    )
    plt.close(fig2)

    # Latent PCA & Healthy -> Diseased Interpolation
    embeddings, diseased_flags, _ = extract_latent_embeddings(vae, val_ds, device)
    proj_2d, explained_var = compute_pca_2d(embeddings)

    # Pick one healthy (flag==0) and one diseased (flag==1) sample from validation set
    healthy_idx = int(np.where(diseased_flags == 0)[0][0])
    diseased_idx = int(np.where(diseased_flags == 1)[0][0])
    img_healthy = val_ds[healthy_idx][0]
    img_diseased = val_ds[diseased_idx][0]

    interp_frames = interpolate_latent(vae, img_healthy, img_diseased, device, n_steps=8)
    fig3 = plot_latent_manifold_and_interpolation(
        proj_2d, explained_var, diseased_flags, interp_frames,
        save_path=out_p / "vae_latent_space_and_interpolation.png",
    )
    plt.close(fig3)

    # 3. Train & Visualize Lightweight DCGAN
    gan_summary = train_dcgan(epochs=20, batch_size=32, latent_dim=64, lr=2e-4, seed=seed)
    gan_ckpt = torch.load("artifacts/checkpoints/generative/dcgan_checkpoint.pt", map_location=device)
    netG = DCGANGenerator(latent_dim=gan_ckpt.get("latent_dim", 64), ngf=32).to(device)
    netG.load_state_dict(gan_ckpt["netG_state_dict"])
    netG.eval()

    fixed_noise = gan_ckpt["fixed_noise"].to(device)
    with torch.no_grad():
        fake_samples = netG(fixed_noise).cpu()

    fig4 = plot_gan_results(
        gan_summary["history"], fake_samples, save_path=out_p / "gan_synthetic_samples_and_loss.png"
    )
    plt.close(fig4)

    print(f"\n[Phase E] All Generative Demonstrations complete! Figures saved in: {out_p}")


if __name__ == "__main__":
    run_all_generative_demonstrations()

