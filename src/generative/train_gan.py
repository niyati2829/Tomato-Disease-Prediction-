"""Training Engine for Lightweight DCGAN on Training Subset (REQ-09).

Uses stabilized DCGAN practices:
- Label smoothing (real targets = 0.9, fake targets = 0.0)
- Adam optimizer with beta1 = 0.5, beta2 = 0.999
- Fixed latent noise tracking for generation progression monitoring
- Strictly uses `data/splits/train.csv` ONLY (never `test.csv`).

Author: Crop Disease Detection Project Team
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.generative.gan import DCGANDiscriminator, DCGANGenerator, weights_init_dcgan
from src.generative.vae_dataset import LeafGenerativeDataset
from src.utils.device import get_device


def train_dcgan(
    epochs: int = 20,
    batch_size: int = 32,
    latent_dim: int = 64,
    lr: float = 2e-4,
    max_train_per_class: int = 150,
    seed: int = 42,
    ckpt_path: Path | str = "artifacts/checkpoints/generative/dcgan_checkpoint.pt",
) -> Dict[str, Any]:
    """Train DCGAN via alternating Generator/Discriminator updates."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device(preferred="mps")
    print(f"[Phase E] Training Lightweight DCGAN on device: {device}")

    train_ds = LeafGenerativeDataset(
        "data/splits/train.csv",
        image_size=64,
        max_per_class=max_train_per_class,
        range_neg_one=True,
        seed=seed,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    netG = DCGANGenerator(latent_dim=latent_dim, ngf=32).to(device)
    netD = DCGANDiscriminator(ndf=32).to(device)
    netG.apply(weights_init_dcgan)
    netD.apply(weights_init_dcgan)

    criterion = nn.BCEWithLogitsLoss()
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)

    history = {"loss_D": [], "loss_G": [], "D_x": [], "D_G_z": []}
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_loss_d, ep_loss_g, ep_dx, ep_dgz = [], [], [], []

        for real_imgs, _, _ in train_loader:
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            real_labels = torch.full((b_size, 1), 0.9, device=device)  # One-sided label smoothing
            fake_labels = torch.zeros((b_size, 1), device=device)

            # 1. Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            out_real = netD(real_imgs)
            errD_real = criterion(out_real, real_labels)
            errD_real.backward()
            d_x = torch.sigmoid(out_real).mean().item()

            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_imgs = netG(noise)
            out_fake = netD(fake_imgs.detach())
            errD_fake = criterion(out_fake, fake_labels)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optD.step()

            # 2. Update Generator: maximize log(D(G(z)))
            netG.zero_grad()
            target_real_for_G = torch.ones((b_size, 1), device=device)
            out_fake_for_G = netD(fake_imgs)
            errG = criterion(out_fake_for_G, target_real_for_G)
            errG.backward()
            d_g_z = torch.sigmoid(out_fake_for_G).mean().item()
            optG.step()

            ep_loss_d.append(errD.item())
            ep_loss_g.append(errG.item())
            ep_dx.append(d_x)
            ep_dgz.append(d_g_z)

        history["loss_D"].append(float(np.mean(ep_loss_d)))
        history["loss_G"].append(float(np.mean(ep_loss_g)))
        history["D_x"].append(float(np.mean(ep_dx)))
        history["D_G_z"].append(float(np.mean(ep_dgz)))

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Loss_D: {history['loss_D'][-1]:.4f}, Loss_G: {history['loss_G'][-1]:.4f} | "
                f"D(x): {history['D_x'][-1]:.3f}, D(G(z)): {history['D_G_z'][-1]:.3f}"
            )

    duration = time.perf_counter() - start_time
    g_params = sum(p.numel() for p in netG.parameters())
    d_params = sum(p.numel() for p in netD.parameters())

    ckpt_p = Path(ckpt_path)
    ckpt_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "netG_state_dict": netG.state_dict(),
        "netD_state_dict": netD.state_dict(),
        "latent_dim": latent_dim,
        "fixed_noise": fixed_noise.cpu(),
        "epochs": epochs,
    }, ckpt_p)

    summary = {
        "history": history,
        "training_time_sec": duration,
        "generator_params": g_params,
        "discriminator_params": d_params,
        "config": {"epochs": epochs, "batch_size": batch_size, "latent_dim": latent_dim, "lr": lr, "seed": seed},
    }

    hist_p = Path("reports/gan_training_history.json")
    hist_p.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Phase E] DCGAN Training Complete in {duration:.2f}s | G Params: {g_params:,}, D Params: {d_params:,}")
    return summary


if __name__ == "__main__":
    train_dcgan()

