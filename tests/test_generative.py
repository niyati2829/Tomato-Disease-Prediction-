"""Unit Tests for Generative Models: ConvVAE & DCGAN (REQ-09).

Verifies:
1. ConvVAE encoder (mu, logvar) and decoder reconstruction shapes.
2. Reparameterization trick dimensions & gradient flow.
3. KL divergence analytical exactness and numerical stability.
4. DCGAN Generator & Discriminator output dimensions and value bounds.
5. Latent space linear interpolation shapes and endpoints.
6. Dataset security guard preventing accidental access to production test.csv.
7. Checkpoint save/load consistency.

Author: Crop Disease Detection Project Team
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.generative.gan import DCGANDiscriminator, DCGANGenerator
from src.generative.latent_analysis import compute_pca_2d, interpolate_latent
from src.generative.vae import ConvVAE, vae_loss_function
from src.generative.vae_dataset import LeafGenerativeDataset


def test_vae_encoder_reparameterize_decoder_dimensions():
    """Verify forward shapes across ConvVAE encode, reparameterize, and decode."""
    B, C, H, W = 4, 3, 64, 64
    latent_dim = 16
    vae = ConvVAE(image_channels=C, latent_dim=latent_dim)
    x = torch.rand(B, C, H, W)

    mu, logvar = vae.encode(x)
    assert mu.shape == (B, latent_dim)
    assert logvar.shape == (B, latent_dim)

    z = vae.reparameterize(mu, logvar)
    assert z.shape == (B, latent_dim)

    recon, mu_f, logvar_f = vae(x)
    assert recon.shape == (B, C, H, W)
    assert torch.all((recon >= 0.0) & (recon <= 1.0))


def test_vae_kl_divergence_exactness_and_gradients():
    """Verify KL(N(0,I) || N(0,I)) == 0, non-negativity, and finite backward gradients."""
    B = 2
    x = torch.rand(B, 3, 64, 64)
    recon = torch.rand(B, 3, 64, 64, requires_grad=True)

    # 1. Exact zero KL at standard normal parameters
    mu_zero = torch.zeros(B, 16)
    logvar_zero = torch.zeros(B, 16)
    _, _, kl_zero = vae_loss_function(recon, x, mu_zero, logvar_zero, beta=1.0)
    assert kl_zero.item() == pytest.approx(0.0, abs=1e-7)

    # 2. Positive KL for shifted parameters & valid gradients
    mu_shift = torch.ones(B, 16, requires_grad=True)
    logvar_shift = torch.full((B, 16), -2.0, requires_grad=True)
    tot, rec_l, kl_l = vae_loss_function(recon, x, mu_shift, logvar_shift, beta=0.5)

    assert kl_l.item() > 0.0
    assert torch.isfinite(tot)
    tot.backward()
    assert recon.grad is not None and torch.all(torch.isfinite(recon.grad))
    assert mu_shift.grad is not None and torch.all(torch.isfinite(mu_shift.grad))


def test_dcgan_generator_and_discriminator_shapes():
    """Verify DCGAN Generator produces [-1, 1] images and Discriminator outputs logits."""
    B = 4
    latent_dim = 64
    netG = DCGANGenerator(latent_dim=latent_dim, ngf=16, image_channels=3)
    netD = DCGANDiscriminator(ndf=16, image_channels=3)

    z = torch.randn(B, latent_dim, 1, 1)
    fake_imgs = netG(z)
    assert fake_imgs.shape == (B, 3, 64, 64)
    assert torch.all((fake_imgs >= -1.0) & (fake_imgs <= 1.0))

    logits = netD(fake_imgs)
    assert logits.shape == (B, 1)
    assert torch.all(torch.isfinite(logits))


def test_latent_interpolation_and_pca():
    """Verify linear latent interpolation trajectory and 2D PCA dimensionality reduction."""
    vae = ConvVAE(image_channels=3, latent_dim=8)
    vae.eval()

    img_a = torch.rand(3, 64, 64)
    img_b = torch.rand(3, 64, 64)

    interp = interpolate_latent(vae, img_a, img_b, torch.device("cpu"), n_steps=5)
    assert interp.shape == (5, 3, 64, 64)
    assert torch.all((interp >= 0.0) & (interp <= 1.0))

    # Endpoint alpha=0 should match decode(encode(img_a))
    with torch.no_grad():
        mu_a, _ = vae.encode(img_a.unsqueeze(0))
        dec_a = vae.decode(mu_a).squeeze(0)
    torch.testing.assert_close(interp[0], dec_a, atol=1e-5, rtol=1e-5)

    # PCA 2D projection
    emb = np.random.RandomState(42).randn(20, 8)
    proj, exp_var = compute_pca_2d(emb)
    assert proj.shape == (20, 2)
    assert exp_var.shape == (2,)
    assert np.all((exp_var >= 0.0) & (exp_var <= 1.0))


def test_generative_dataset_forbids_test_split():
    """Verify security guard refuses to load holdout test.csv."""
    with pytest.raises(ValueError, match="SECURITY VIOLATION"):
        _ = LeafGenerativeDataset("data/splits/test.csv", max_per_class=10)


def test_vae_checkpoint_save_and_load():
    """Verify ConvVAE weights serialize and deserialize identically."""
    vae1 = ConvVAE(image_channels=3, latent_dim=8)
    vae1.eval()
    x = torch.rand(1, 3, 64, 64)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_p = Path(tmpdir) / "test_vae.pt"
        torch.save({"model_state_dict": vae1.state_dict()}, ckpt_p)

        vae2 = ConvVAE(image_channels=3, latent_dim=8)
        ckpt = torch.load(ckpt_p, map_location="cpu")
        vae2.load_state_dict(ckpt["model_state_dict"])
        vae2.eval()

        with torch.no_grad():
            mu1, _ = vae1.encode(x)
            mu2, _ = vae2.encode(x)
        torch.testing.assert_close(mu1, mu2)

