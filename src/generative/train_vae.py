"""Training Engine for Convolutional VAE on PlantVillage Training Subset (REQ-09).

Safety Rules:
- Uses ONLY `data/splits/train.csv` for fitting and `data/splits/val.csv` for checkpoint selection.
- NEVER accesses `data/splits/test.csv`.
- Saves best checkpoint to artifacts/checkpoints/generative/best_vae.pt.

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
from torch.utils.data import DataLoader

from src.generative.vae import ConvVAE, vae_loss_function
from src.generative.vae_dataset import LeafGenerativeDataset
from src.utils.device import get_device


def train_conv_vae(
    epochs: int = 25,
    batch_size: int = 32,
    latent_dim: int = 16,
    beta: float = 1.0,
    lr: float = 1e-3,
    max_train_per_class: int = 150,
    max_val_per_class: int = 40,
    seed: int = 42,
    ckpt_path: Path | str = "artifacts/checkpoints/generative/best_vae.pt",
) -> Dict[str, Any]:
    """Train ConvVAE and record ELBO curves."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device(preferred="mps")
    print(f"[Phase E] Training ConvVAE on device: {device}")

    train_ds = LeafGenerativeDataset(
        "data/splits/train.csv", image_size=64, max_per_class=max_train_per_class, seed=seed
    )
    val_ds = LeafGenerativeDataset(
        "data/splits/val.csv", image_size=64, max_per_class=max_val_per_class, seed=seed
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ConvVAE(image_channels=3, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_total": [],
        "train_recon": [],
        "train_kl": [],
        "val_total": [],
        "val_recon": [],
        "val_kl": [],
    }

    best_val_loss = float("inf")
    ckpt_p = Path(ckpt_path)
    ckpt_p.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        t_tot, t_rec, t_kl = [], [], []
        for imgs, _, _ in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(imgs)
            loss, rec_l, kl_l = vae_loss_function(recon, imgs, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            t_tot.append(loss.item())
            t_rec.append(rec_l.item())
            t_kl.append(kl_l.item())

        # Validation
        model.eval()
        v_tot, v_rec, v_kl = [], [], []
        with torch.no_grad():
            for imgs, _, _ in val_loader:
                imgs = imgs.to(device)
                mu, logvar = model.encode(imgs)
                recon = model.decode(mu)  # Deterministic evaluation using mean embedding
                loss, rec_l, kl_l = vae_loss_function(recon, imgs, mu, logvar, beta=beta)

                v_tot.append(loss.item())
                v_rec.append(rec_l.item())
                v_kl.append(kl_l.item())

        avg_t_tot, avg_t_rec, avg_t_kl = float(np.mean(t_tot)), float(np.mean(t_rec)), float(np.mean(t_kl))
        avg_v_tot, avg_v_rec, avg_v_kl = float(np.mean(v_tot)), float(np.mean(v_rec)), float(np.mean(v_kl))

        history["train_total"].append(avg_t_tot)
        history["train_recon"].append(avg_t_rec)
        history["train_kl"].append(avg_t_kl)
        history["val_total"].append(avg_v_tot)
        history["val_recon"].append(avg_v_rec)
        history["val_kl"].append(avg_v_kl)

        if avg_v_tot < best_val_loss:
            best_val_loss = avg_v_tot
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_total_loss": best_val_loss,
                "val_recon_loss": avg_v_rec,
                "val_kl_loss": avg_v_kl,
                "latent_dim": latent_dim,
                "beta": beta,
            }, ckpt_p)
            tag = " (*)"
        else:
            tag = ""

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train ELBO: {avg_t_tot:.2f} (Rec: {avg_t_rec:.2f}, KL: {avg_t_kl:.2f}) | "
                f"Val ELBO: {avg_v_tot:.2f}{tag}"
            )

    duration = time.perf_counter() - start_time
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Phase E] ConvVAE Training Complete in {duration:.2f}s | Params: {param_count:,} | Best Val ELBO: {best_val_loss:.2f}")

    summary = {
        "history": history,
        "training_time_sec": duration,
        "parameter_count": param_count,
        "best_val_total_loss": best_val_loss,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "beta": beta,
            "lr": lr,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "seed": seed,
        },
    }

    hist_p = Path("reports/vae_training_history.json")
    hist_p.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    train_conv_vae()

