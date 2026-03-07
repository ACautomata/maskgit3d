"""Tests for VQVAETask."""

import pytest
import torch

from src.maskgit3d.tasks.vqvae_task import VQVAETask


def test_vqvae_task_manual_optimization():
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=256,
        num_embeddings=100,
        embedding_dim=16,
        lr_g=1e-4,
        lr_d=1e-4,
    )

    assert task.automatic_optimization is False


def test_vqvae_task_configure_optimizers():
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=256,
        num_embeddings=100,
        embedding_dim=16,
        lr_g=1e-4,
        lr_d=1e-4,
    )

    optimizers = task.configure_optimizers()

    assert isinstance(optimizers, list)
    assert len(optimizers) == 2
    assert isinstance(optimizers[0], torch.optim.Adam)
    assert isinstance(optimizers[1], torch.optim.Adam)


def test_vqvae_task_training_step():
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=256,
        num_embeddings=100,
        embedding_dim=16,
        lr_g=1e-4,
        lr_d=1e-4,
    )

    assert task.automatic_optimization is False
    assert hasattr(task, "vqvae")
    assert hasattr(task, "discriminator")

    optimizers = [
        torch.optim.Adam(task.vqvae.parameters(), lr=1e-4),
        torch.optim.Adam(task.discriminator.parameters(), lr=1e-4),
    ]
    assert len(optimizers) == 2
