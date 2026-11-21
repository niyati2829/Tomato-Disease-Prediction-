"""Evaluation Engine for Convolutional VAE (REQ-09).

Evaluates reconstruction fidelity (Per-Pixel MSE, PSNR) and KL divergence
on validation images from `data/splits/val.csv` (strictly preserving production test.csv).

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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.generative.vae import ConvVAE, vae_loss_function
from src.generative.vae_dataset import LeafGenerativeDataset
from src.utils.device import get_device


def evaluate_vae_checkpoint(
    ckpt_path: Path | str = "artifacts/checkpoints/generative/best_vae.pt",
    out_report_path: Path | str = "reports/vae_evaluation_results.json",
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate VAE reconstruction and KL metrics on validation split."""
    ckpt_p = Path(ckpt_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

    device = get_device(preferred="mps")
    ckpt = torch.load(ckpt_p, map_location=device)
    latent_dim = ckpt.get("latent_dim", 16)
    beta = ckpt.get("beta", 1.0)

    model = ConvVAE(image_channels=3, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ds = LeafGenerativeDataset("data/splits/val.csv", image_size=64, max_per_class=40, seed=seed)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    pixel_mses = []
    psnrs = []
    kl_losses = []
    latencies = []

    with torch.no_grad():
        for imgs, _, _ in val_loader:
            imgs = imgs.to(device)
            t0 = time.perf_counter()
            mu, logvar = model.encode(imgs)
            recon = model.decode(mu)
            t1 = time.perf_counter()
            latencies.append(((t1 - t0) * 1000.0) / imgs.size(0))

            _, _, kl = vae_loss_function(recon, imgs, mu, logvar, beta=beta)
            kl_losses.append(kl.item())

            for i in range(imgs.size(0)):
                mse_i = F.mse_loss(recon[i], imgs[i], reduction="mean").item()
                pixel_mses.append(mse_i)
                psnr_i = 10.0 * np.log10(1.0 / max(mse_i, 1e-10))
                psnrs.append(psnr_i)

    results = {
        "checkpoint": str(ckpt_p),
        "val_samples": len(val_ds),
        "latent_dim": latent_dim,
        "mean_pixel_mse": float(np.mean(pixel_mses)),
        "mean_psnr_db": float(np.mean(psnrs)),
        "mean_kl_divergence": float(np.mean(kl_losses)),
        "per_image_inference_latency_ms": float(np.mean(latencies)),
    }

    report_p = Path(out_report_path)
    report_p.parent.mkdir(parents=True, exist_ok=True)
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n================ ConvVAE Validation Evaluation ================")
    print(f"Samples Evaluated:          {results['val_samples']}")
    print(f"Mean Pixel MSE:             {results['mean_pixel_mse']:.5f}")
    print(f"Mean PSNR:                  {results['mean_psnr_db']:.2f} dB")
    print(f"Mean KL Divergence:         {results['mean_kl_divergence']:.2f}")
    print(f"Inference Latency / Image:  {results['per_image_inference_latency_ms']:.3f} ms")
    print(f"Saved to: {report_p}")

    return results


if __name__ == "__main__":
    evaluate_vae_checkpoint()

