"""Tests for VQVAETask."""

import pytest
import torch

from src.maskgit3d.tasks.vqvae_task import (
    VQVAETask,
    compute_downsampling_factor,
    compute_padded_size,
    validate_crop_size,
)


def test_compute_downsampling_factor():
    assert compute_downsampling_factor((1, 1, 2, 2, 4)) == 16
    assert compute_downsampling_factor((1, 2, 4)) == 4
    assert compute_downsampling_factor((1, 2, 4, 8)) == 8


def test_validate_crop_size():
    # Valid crop size
    result = validate_crop_size((64, 64, 64), 16)
    assert result == (64, 64, 64)

    # Invalid crop size
    with pytest.raises(ValueError):
        validate_crop_size((50, 64, 64), 16)


def test_compute_padded_size():
    # Size already divisible
    result = compute_padded_size((64, 64, 64), 16)
    assert result == (64, 64, 64)

    # Size needs padding
    result = compute_padded_size((50, 50, 50), 16)
    assert result == (64, 64, 64)

    result = compute_padded_size((17, 31, 47), 16)
    assert result == (32, 32, 48)


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


def test_vqvae_task_has_vqvae_and_discriminator():
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=256,
        num_embeddings=100,
        embedding_dim=16,
        lr_g=1e-4,
        lr_d=1e-4,
    )

    assert hasattr(task, "vqvae")
    assert hasattr(task, "loss_fn")
    assert hasattr(task.loss_fn, "discriminator")


def test_vqvae_task_sliding_window_config():
    sliding_window_cfg = {
        "enabled": True,
        "roi_size": [64, 64, 64],
        "overlap": 0.25,
        "mode": "gaussian",
        "sw_batch_size": 1,
    }
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=100,
        embedding_dim=64,
        sliding_window=sliding_window_cfg,
    )

    assert task.sliding_window_cfg == sliding_window_cfg
    assert task.sliding_window_cfg["enabled"] is True
    assert task.sliding_window_cfg["roi_size"] == [64, 64, 64]
    assert task.sliding_window_cfg["overlap"] == 0.25


def test_vqvae_task_sliding_window_inferer_creation():
    sliding_window_cfg = {
        "enabled": True,
        "roi_size": [64, 64, 64],
        "overlap": 0.25,
        "mode": "gaussian",
        "sw_batch_size": 2,
    }
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=100,
        embedding_dim=64,
        sliding_window=sliding_window_cfg,
    )

    inferer = task._get_sliding_window_inferer()
    assert inferer is not None
    assert inferer.roi_size == (64, 64, 64)
    assert inferer.overlap == 0.25
    assert inferer.sw_batch_size == 2


def test_vqvae_task_sliding_window_disabled():
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=100,
        embedding_dim=64,
        sliding_window={"enabled": False},
    )

    inferer = task._get_sliding_window_inferer()
    assert inferer is None


def test_vqvae_task_divisible_pad():
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=100,
        embedding_dim=64,
    )

    pad = task._get_divisible_pad()
    assert pad is not None
    assert pad.k == 16


@pytest.mark.integration
def test_vqvae_task_forward_pass():
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=100,
        embedding_dim=64,
        use_perceptual=False,
    )
    task.eval()

    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        recon, vq_loss = task(x)
    assert recon.shape == x.shape


@pytest.mark.integration
def test_vqvae_task_predict_step_direct():
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=100,
        embedding_dim=64,
        use_perceptual=False,
        sliding_window={"enabled": False},
    )
    task.eval()

    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        output = task.predict_step(x, 0)
    assert output.shape == x.shape
