"""Tests for VQVAE model."""

import torch

from src.maskgit3d.models.vqvae import VQVAE


def test_vqvae_forward_pass():
    vqvae = VQVAE(
        in_channels=1,
        out_channels=1,
        latent_channels=256,
        num_embeddings=8192,
        embedding_dim=256,
    )

    x = torch.randn(2, 1, 32, 32, 32)
    x_recon, vq_loss = vqvae(x)

    assert x_recon.shape[0] == 2
    assert x_recon.shape[1] == 1
    assert x_recon.shape[2] == 32
    assert vq_loss.dim() == 0


def test_vqvae_encode_decode():
    vqvae = VQVAE(
        in_channels=1,
        out_channels=1,
        latent_channels=256,
        num_embeddings=100,
        embedding_dim=16,
    )

    x = torch.randn(1, 1, 16, 16, 16)
    z_q, vq_loss, indices = vqvae.encode(x)
    x_recon = vqvae.decode(z_q)

    assert z_q.shape[1] == 16
    assert indices.shape[0] == 1
    assert x_recon.shape == x.shape
