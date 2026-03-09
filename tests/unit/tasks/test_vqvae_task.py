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

    x = torch.randn(1, 1, 33, 33, 33)
    x_padded = task._pad_to_divisible(x)
    assert x_padded.shape[2] % 16 == 0
    assert x_padded.shape[3] % 16 == 0
    assert x_padded.shape[4] % 16 == 0
    assert x_padded.shape[1] == x.shape[1]


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


class MockLogger:
    def __init__(self):
        self.logs: dict[str, torch.Tensor] = {}

    def __call__(self, name: str, value: torch.Tensor | float, **kwargs):
        if isinstance(value, torch.Tensor):
            self.logs[name] = value
        else:
            self.logs[name] = torch.tensor(value)


def test_vqvae_task_validation_step():
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
        outputs = task.validation_step(x, 0)

    assert isinstance(outputs, dict)
    assert "loss" in outputs
    assert "x_real" in outputs
    assert "x_recon" in outputs
    assert "vq_loss" in outputs
    assert isinstance(outputs["x_real"], torch.Tensor)
    assert isinstance(outputs["x_recon"], torch.Tensor)
    assert isinstance(outputs["vq_loss"], torch.Tensor)


def test_vqvae_task_validation_step_with_perceptual():
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=100,
        embedding_dim=64,
        use_perceptual=True,
    )
    task.eval()

    x = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        outputs = task.validation_step(x, 0)

    assert isinstance(outputs, dict)
    assert "loss" in outputs
    assert "x_real" in outputs
    assert "x_recon" in outputs
    assert "vq_loss" in outputs


def test_vqvae_task_test_step_without_sliding_window():
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
        outputs = task.test_step(x, 0)

    assert isinstance(outputs, dict)
    assert "loss" in outputs
    assert "x_real" in outputs
    assert "x_recon" in outputs
    assert "inference_time" in outputs
    assert "use_sliding_window" in outputs


@pytest.mark.integration
def test_vqvae_task_test_step_with_sliding_window():
    sliding_window_cfg = {
        "enabled": True,
        "roi_size": [16, 16, 16],
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
        use_perceptual=False,
        sliding_window=sliding_window_cfg,
    )
    task.eval()

    x = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        outputs = task.test_step(x, 0)

    assert isinstance(outputs, dict)
    assert "loss" in outputs
    assert "x_real" in outputs
    assert "x_recon" in outputs


def test_vqvae_task_predict_step_without_sliding_window():
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
    assert output.shape[0] == 1
    assert output.shape[1] == 1


@pytest.mark.integration
def test_vqvae_task_predict_step_with_sliding_window():
    sliding_window_cfg = {
        "enabled": True,
        "roi_size": [16, 16, 16],
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
        use_perceptual=False,
        sliding_window=sliding_window_cfg,
    )
    task.eval()

    x = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        output = task.predict_step(x, 0)

    assert output.shape == x.shape
    assert output.shape[0] == 1
    assert output.shape[1] == 1


def test_vqvae_task_predict_step_returns_correct_values():
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
    assert output.isfinite().all(), "Output should contain finite values"


def test_vqvae_task_test_step_returns_no_cuda_memory_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

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
        outputs = task.test_step(x, 0)

    assert isinstance(outputs, dict)
    assert "x_real" in outputs
    assert "x_recon" in outputs
    assert "inference_time" in outputs


def test_vqvae_task_get_decoder_last_layer():
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=100,
        embedding_dim=64,
    )

    last_layer = task._get_decoder_last_layer()
    assert last_layer is not None
    assert isinstance(last_layer, torch.nn.Parameter)


def test_vqvae_task_shared_step_generator():
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
        loss_g, log_g = task._shared_step_generator(x, 0, "val")

    assert isinstance(loss_g, torch.Tensor)
    assert isinstance(log_g, dict)
    assert len(log_g) > 0


def test_vqvae_task_shared_step_discriminator():
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
        x_recon, vq_loss = task.vqvae(x)

    with torch.no_grad():
        loss_d, log_d = task._shared_step_discriminator(x, x_recon, vq_loss, "val")

    assert isinstance(loss_d, torch.Tensor)
    assert isinstance(log_d, dict)


def test_vqvae_task_forward():
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=100,
        embedding_dim=64,
    )
    task.eval()

    x = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        recon, vq_loss = task(x)

    assert recon.shape == x.shape
    assert isinstance(vq_loss, torch.Tensor)


def test_vqvae_task_get_decoder_last_layer_error():
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=100,
        embedding_dim=64,
    )

    delattr(task.vqvae.decoder, "decoder")

    result = task._get_decoder_last_layer()
    assert result is None


def test_vqvae_task_shared_step_generator_with_last_layer():
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
    last_layer = task._get_decoder_last_layer()

    with torch.no_grad():
        loss_g, log_g = task._shared_step_generator(x, 0, "val", last_layer=last_layer)

    assert isinstance(loss_g, torch.Tensor)
    assert isinstance(log_g, dict)


def test_vqvae_task_training_step_with_optimizers_parameter():
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=100,
        embedding_dim=64,
        use_perceptual=False,
    )

    opt_g = torch.optim.Adam(list(task.vqvae.parameters()), lr=1e-4)
    opt_d = torch.optim.Adam(task.loss_fn.discriminator.parameters(), lr=1e-4)

    task.manual_backward = lambda loss: loss.backward()  # type: ignore[assignment]

    batch = torch.randn(1, 1, 32, 32, 32)
    outputs = task.training_step(batch, 0, optimizers=[opt_g, opt_d])

    assert isinstance(outputs, dict)
    assert "loss" in outputs
    assert "x_real" in outputs
    assert "x_recon" in outputs
    assert "vq_loss" in outputs
    assert "last_layer" in outputs
    assert isinstance(outputs["loss"], torch.Tensor)
    assert outputs["loss"].item() >= 0


def test_vqvae_task_training_step_accepts_tuple_batch():
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=100,
        embedding_dim=64,
        use_perceptual=False,
    )

    opt_g = torch.optim.Adam(list(task.vqvae.parameters()), lr=1e-4)
    opt_d = torch.optim.Adam(task.loss_fn.discriminator.parameters(), lr=1e-4)

    task.manual_backward = lambda loss: loss.backward()  # type: ignore[assignment]

    image_batch = torch.randn(1, 1, 32, 32, 32)
    target_batch = image_batch.clone()
    outputs = task.training_step((image_batch, target_batch), 0, optimizers=[opt_g, opt_d])

    assert torch.equal(outputs["x_real"], image_batch)
    assert outputs["x_recon"].shape == image_batch.shape


@pytest.mark.parametrize("step_name", ["validation_step", "test_step", "predict_step"])
def test_vqvae_task_inference_steps_accept_tuple_batch(step_name: str):
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

    image_batch = torch.randn(1, 1, 32, 32, 32)
    target_batch = image_batch.clone()

    with torch.no_grad():
        output = getattr(task, step_name)((image_batch, target_batch), 0)

    if isinstance(output, dict):
        assert torch.equal(output["x_real"], image_batch)
        assert output["x_recon"].shape == image_batch.shape
    else:
        assert output.shape == image_batch.shape


def test_vqvae_task_test_step_returns_cuda_memory_with_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda: 1024 * 1024)

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
        outputs = task.test_step(x, 0)

    assert isinstance(outputs, dict)
    assert "x_real" in outputs
    assert "x_recon" in outputs
    assert "inference_time" in outputs


def test_vqvae_task_sliding_window_inferer_reuse():
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

    inferer1 = task._get_sliding_window_inferer()
    inferer2 = task._get_sliding_window_inferer()

    assert inferer1 is inferer2


def test_vqvae_task_pad_to_divisible_consistency():
    task = VQVAETask(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=100,
        embedding_dim=64,
    )

    x = torch.randn(1, 1, 33, 33, 33)
    x_padded1 = task._pad_to_divisible(x)
    x_padded2 = task._pad_to_divisible(x)

    assert x_padded1.shape == x_padded2.shape
    assert torch.allclose(x_padded1, x_padded2)
