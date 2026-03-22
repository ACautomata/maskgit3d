"""Tests for VQVAETask."""

from typing import Any, cast
import warnings

import pytest
import torch
from omegaconf import DictConfig

from src.maskgit3d.inference import VQVAEReconstructor
from src.maskgit3d.losses.vq_perceptual_loss import VQPerceptualLoss
from src.maskgit3d.models.vqvae import VQVAE
from src.maskgit3d.models.vqvae.splitting import compute_downsampling_factor
from src.maskgit3d.runtime.optimizer_factory import GANOptimizerFactory
from src.maskgit3d.tasks.gan_training_strategy import GANTrainingStrategy
from src.maskgit3d.tasks.vqvae_task import VQVAETask
from src.maskgit3d.training import VQVAETrainingSteps
from src.maskgit3d.utils import compute_padded_size, validate_crop_size


def _build_injected_vqvae_components() -> tuple[
    VQVAE,
    VQPerceptualLoss,
    VQVAETrainingSteps,
    GANOptimizerFactory,
]:
    model = VQVAE(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=100,
        embedding_dim=64,
    )
    loss_fn = VQPerceptualLoss(
        disc_in_channels=1,
        disc_num_layers=3,
        disc_ndf=64,
        disc_norm="instance",
        disc_loss="hinge",
        lambda_l1=1.0,
        lambda_vq=1.0,
        lambda_perceptual=0.1,
        discriminator_weight=0.1,
        disc_start=0,
        disc_factor=1.0,
        use_adaptive_weight=True,
        adaptive_weight_max=100.0,
        perceptual_network="alex",
        use_perceptual=False,
    )
    training_steps = VQVAETrainingSteps(
        vqvae=model,
        loss_fn=loss_fn,
        gan_strategy=GANTrainingStrategy(gradient_clip_val=1.0, gradient_clip_enabled=True),
        reconstructor=VQVAEReconstructor(sliding_window={}, downsampling_factor=4),
    )
    optimizer_factory = GANOptimizerFactory(lr_g=1e-4, lr_d=2e-4)
    return model, loss_fn, training_steps, optimizer_factory


def test_vqvae_task_accepts_injected_components():
    model, loss_fn, training_steps, optimizer_factory = _build_injected_vqvae_components()

    task = VQVAETask(
        model=model,
        loss_fn=loss_fn,
        training_steps=training_steps,
        optimizer_factory=optimizer_factory,
    )

    assert task.vqvae is model
    assert task.loss_fn is loss_fn
    assert task.training_steps is training_steps
    assert task.optimizer_factory is optimizer_factory
    assert task.training_steps.vqvae is task.vqvae
    assert task.training_steps.loss_fn is task.loss_fn


def test_vqvae_task_legacy_path_emits_deprecation():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        task = VQVAETask(
            in_channels=1,
            out_channels=1,
            latent_channels=64,
            num_embeddings=100,
            embedding_dim=64,
            use_perceptual=False,
        )

    assert isinstance(task, VQVAETask)
    assert any(item.category is DeprecationWarning for item in caught)


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

    optimizers, schedulers = task.configure_optimizers()

    assert isinstance(optimizers, list)
    assert len(optimizers) == 2
    assert isinstance(optimizers[0], torch.optim.Adam)
    assert isinstance(optimizers[1], torch.optim.Adam)
    assert isinstance(schedulers, list)
    assert len(schedulers) == 2


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
    assert x_padded.shape[2] % task._downsampling_factor == 0
    assert x_padded.shape[3] % task._downsampling_factor == 0
    assert x_padded.shape[4] % task._downsampling_factor == 0
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
        output = cast(dict[str, Any], task.predict_step(x, 0))
    assert isinstance(output, dict)
    assert set(output) == {"x_real", "x_recon"}
    assert output["x_real"].shape == x.shape
    assert output["x_recon"].shape == x.shape


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
    assert set(outputs) == {"x_real", "x_recon"}
    assert isinstance(outputs["x_real"], torch.Tensor)
    assert isinstance(outputs["x_recon"], torch.Tensor)


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
    assert set(outputs) == {"x_real", "x_recon"}


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
    assert set(outputs) == {"x_real", "x_recon"}
    assert isinstance(outputs["x_real"], torch.Tensor)
    assert isinstance(outputs["x_recon"], torch.Tensor)


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
    assert set(outputs) == {"x_real", "x_recon"}


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
        output = cast(dict[str, Any], task.predict_step(x, 0))

    assert set(output) == {"x_real", "x_recon"}
    assert output["x_real"].shape == x.shape
    assert output["x_recon"].shape == x.shape


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
        output = cast(dict[str, Any], task.predict_step(x, 0))

    assert set(output) == {"x_real", "x_recon"}
    assert output["x_real"].shape == x.shape
    assert output["x_recon"].shape == x.shape


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
        output = cast(dict[str, Any], task.predict_step(x, 0))

    assert output["x_recon"].shape == x.shape
    assert output["x_recon"].isfinite().all(), "Output should contain finite values"


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
    assert set(outputs) == {"x_real", "x_recon"}


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
    output = task.training_step(batch, 0, optimizers=[opt_g, opt_d])

    assert isinstance(output, dict)
    assert "loss" in output
    assert isinstance(output["loss"], torch.Tensor)
    assert {
        "loss_g",
        "loss_d",
        "nll_loss",
        "rec_loss",
        "p_loss",
        "g_loss",
        "vq_loss",
        "disc_loss",
    } <= set(output)


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
    output = task.training_step((image_batch, target_batch), 0, optimizers=[opt_g, opt_d])

    assert isinstance(output, dict)
    assert isinstance(output["loss"], torch.Tensor)


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

    assert isinstance(output, dict)
    assert set(output) == {"x_real", "x_recon"}
    assert output["x_real"].shape == image_batch.shape
    assert output["x_recon"].shape == image_batch.shape


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
    assert set(outputs) == {"x_real", "x_recon"}


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


class TestVQVAETaskModelConfig:
    """Tests for VQVAETask accepting model via DictConfig."""

    def test_accepts_model_config(self):
        """VQVAETask can be constructed with model_config DictConfig."""
        from omegaconf import DictConfig

        from src.maskgit3d.models.vqvae import VQVAE
        from src.maskgit3d.tasks.vqvae_task import VQVAETask

        model_cfg = DictConfig(
            {
                "_target_": "maskgit3d.models.vqvae.VQVAE",
                "in_channels": 1,
                "out_channels": 1,
                "latent_channels": 4,
                "num_embeddings": 32,
                "embedding_dim": 4,
                "num_channels": [32, 64],
                "num_res_blocks": [1, 1],
                "attention_levels": [False, False],
                "commitment_cost": 0.25,
            }
        )
        task = VQVAETask(model_config=model_cfg, lr_g=1e-4, lr_d=1e-4)
        assert isinstance(task.vqvae, VQVAE)
        assert task.vqvae.in_channels == 1

    def test_model_config_overrides_scalar_params(self):
        """When model_config is provided, scalar model params are ignored."""
        from omegaconf import DictConfig

        from src.maskgit3d.tasks.vqvae_task import VQVAETask

        model_cfg = DictConfig(
            {
                "_target_": "maskgit3d.models.vqvae.VQVAE",
                "in_channels": 2,  # different from scalar
                "out_channels": 2,
                "latent_channels": 4,
                "num_embeddings": 32,
                "embedding_dim": 4,
                "num_channels": [32, 64],
                "num_res_blocks": [1, 1],
                "attention_levels": [False, False],
            }
        )
        task = VQVAETask(
            model_config=model_cfg,
            in_channels=1,  # should be ignored
            out_channels=1,  # should be ignored
            lr_g=1e-4,
            lr_d=1e-4,
        )
        assert task.vqvae.in_channels == 2

    def test_scalar_params_still_work(self):
        """Backward compat: scalar params still construct VQVAE when no model_config."""
        from src.maskgit3d.models.vqvae import VQVAE
        from src.maskgit3d.tasks.vqvae_task import VQVAETask

        task = VQVAETask(
            in_channels=1,
            out_channels=1,
            latent_channels=4,
            num_embeddings=32,
            embedding_dim=4,
            num_channels=[32, 64],
            num_res_blocks=[1, 1],
            attention_levels=[False, False],
            lr_g=1e-4,
            lr_d=1e-4,
        )
        assert isinstance(task.vqvae, VQVAE)

    def test_model_config_saved_in_hparams(self):
        """model_config should be serializable via save_hyperparameters."""
        from omegaconf import DictConfig

        from src.maskgit3d.tasks.vqvae_task import VQVAETask

        model_cfg = DictConfig(
            {
                "_target_": "maskgit3d.models.vqvae.VQVAE",
                "in_channels": 1,
                "out_channels": 1,
                "latent_channels": 4,
                "num_embeddings": 32,
                "embedding_dim": 4,
                "num_channels": [32, 64],
                "num_res_blocks": [1, 1],
                "attention_levels": [False, False],
            }
        )
        task = VQVAETask(model_config=model_cfg, lr_g=1e-4, lr_d=1e-4)
        assert "model_config" in task.hparams

    def test_fsq_model_config(self):
        """VQVAETask works with FSQ model variant via model_config."""
        from omegaconf import DictConfig

        from src.maskgit3d.models.vqvae import VQVAE
        from src.maskgit3d.tasks.vqvae_task import VQVAETask

        model_cfg = DictConfig(
            {
                "_target_": "maskgit3d.models.vqvae.VQVAE",
                "in_channels": 1,
                "out_channels": 1,
                "latent_channels": 4,
                "num_channels": [32, 64],
                "num_res_blocks": [1, 1],
                "attention_levels": [False, False],
                "quantizer_type": "fsq",
                "fsq_levels": [8, 8, 8, 5, 5, 5],
            }
        )
        task = VQVAETask(model_config=model_cfg, lr_g=1e-4, lr_d=1e-4)
        assert isinstance(task.vqvae, VQVAE)


class TestVQVAETaskOptimizerConfig:
    """Tests for VQVAETask accepting optimizer via DictConfig."""

    def test_accepts_optimizer_config(self):
        """VQVAETask uses optimizer_config for generator optimizer."""
        opt_cfg = DictConfig(
            {
                "_target_": "torch.optim.AdamW",
                "lr": 2e-4,
                "weight_decay": 0.01,
            }
        )
        task = VQVAETask(
            model_config=DictConfig(
                {
                    "_target_": "maskgit3d.models.vqvae.VQVAE",
                    "in_channels": 1,
                    "out_channels": 1,
                    "latent_channels": 4,
                    "num_embeddings": 32,
                    "embedding_dim": 4,
                    "num_channels": [32, 64],
                    "num_res_blocks": [1, 1],
                    "attention_levels": [False, False],
                }
            ),
            optimizer_config=opt_cfg,
            lr_g=1e-4,
            lr_d=1e-4,
        )
        optimizers, schedulers = task.configure_optimizers()
        # Generator optimizer should be AdamW (from config)
        assert isinstance(optimizers[0], torch.optim.AdamW)
        # Discriminator optimizer should still be Adam (default)
        assert isinstance(optimizers[1], torch.optim.Adam)
        assert len(schedulers) == 2

    def test_accepts_disc_optimizer_config(self):
        """VQVAETask uses disc_optimizer_config for discriminator optimizer."""
        disc_opt_cfg = DictConfig(
            {
                "_target_": "torch.optim.SGD",
                "lr": 1e-3,
                "momentum": 0.9,
            }
        )
        task = VQVAETask(
            model_config=DictConfig(
                {
                    "_target_": "maskgit3d.models.vqvae.VQVAE",
                    "in_channels": 1,
                    "out_channels": 1,
                    "latent_channels": 4,
                    "num_embeddings": 32,
                    "embedding_dim": 4,
                    "num_channels": [32, 64],
                    "num_res_blocks": [1, 1],
                    "attention_levels": [False, False],
                }
            ),
            disc_optimizer_config=disc_opt_cfg,
            lr_g=1e-4,
            lr_d=1e-4,
        )
        optimizers, schedulers = task.configure_optimizers()
        assert isinstance(optimizers[1], torch.optim.SGD)
        assert len(schedulers) == 2

    def test_optimizer_config_fallback_to_scalar(self):
        """Without optimizer_config, uses hardcoded Adam with lr_g/lr_d."""
        task = VQVAETask(
            model_config=DictConfig(
                {
                    "_target_": "maskgit3d.models.vqvae.VQVAE",
                    "in_channels": 1,
                    "out_channels": 1,
                    "latent_channels": 4,
                    "num_embeddings": 32,
                    "embedding_dim": 4,
                    "num_channels": [32, 64],
                    "num_res_blocks": [1, 1],
                    "attention_levels": [False, False],
                }
            ),
            lr_g=1e-4,
            lr_d=1e-4,
        )
        optimizers, schedulers = task.configure_optimizers()
        assert isinstance(optimizers[0], torch.optim.Adam)
        assert isinstance(optimizers[1], torch.optim.Adam)
        assert len(schedulers) == 2

    def test_both_optimizer_configs(self):
        """Both generator and discriminator optimizers can be configured."""
        gen_opt_cfg = DictConfig(
            {
                "_target_": "torch.optim.AdamW",
                "lr": 2e-4,
                "weight_decay": 0.01,
            }
        )
        disc_opt_cfg = DictConfig(
            {
                "_target_": "torch.optim.AdamW",
                "lr": 1e-4,
                "weight_decay": 0.0,
            }
        )
        task = VQVAETask(
            model_config=DictConfig(
                {
                    "_target_": "maskgit3d.models.vqvae.VQVAE",
                    "in_channels": 1,
                    "out_channels": 1,
                    "latent_channels": 4,
                    "num_embeddings": 32,
                    "embedding_dim": 4,
                    "num_channels": [32, 64],
                    "num_res_blocks": [1, 1],
                    "attention_levels": [False, False],
                }
            ),
            optimizer_config=gen_opt_cfg,
            disc_optimizer_config=disc_opt_cfg,
        )
        optimizers, schedulers = task.configure_optimizers()
        assert isinstance(optimizers[0], torch.optim.AdamW)
        assert isinstance(optimizers[1], torch.optim.AdamW)
        assert len(schedulers) == 2
