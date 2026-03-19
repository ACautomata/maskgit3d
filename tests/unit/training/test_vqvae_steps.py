"""Tests for extracted VQVAE training and reconstruction steps."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from src.maskgit3d.training.vqvae_steps import VQVAETrainingSteps


class RecordingLoss:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.use_perceptual = True
        self.perceptual_loss = lambda recon, real: torch.tensor(0.75)

    def __call__(self, **kwargs: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.calls.append(kwargs)
        optimizer_idx = kwargs["optimizer_idx"]
        loss = kwargs["reconstructions"].mean() + kwargs["vq_loss"] + optimizer_idx
        metric_prefix = kwargs["split"]
        metric_name = "disc_loss" if optimizer_idx == 1 else "total_loss"
        return loss, {f"{metric_prefix}/{metric_name}": torch.tensor(float(optimizer_idx + 1))}


class RecordingStrategy:
    def __init__(self) -> None:
        self.generator_calls: list[tuple[Any, torch.Tensor, Any]] = []
        self.discriminator_calls: list[tuple[Any, torch.Tensor]] = []

    def step_generator(self, optimizer: Any, loss: torch.Tensor, vqvae: Any) -> None:
        self.generator_calls.append((optimizer, loss, vqvae))
        optimizer.step()

    def step_discriminator(self, optimizer: Any, loss: torch.Tensor) -> None:
        self.discriminator_calls.append((optimizer, loss))
        optimizer.step()


class RecordingOptimizer:
    def __init__(self) -> None:
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


class RecordingReconstructor:
    def __init__(self, recon_offset: float = 0.5, inferer: object | None = None) -> None:
        self.recon_offset = recon_offset
        self.inferer = inferer
        self.extract_calls: list[Any] = []
        self.reconstruct_calls: list[tuple[Any, torch.Tensor]] = []

    def extract_input_tensor(self, batch: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        self.extract_calls.append(batch)
        if isinstance(batch, tuple):
            return batch[0]
        return batch

    def get_sliding_window_inferer(self) -> object | None:
        return self.inferer

    def reconstruct(
        self, vqvae: Any, batch: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        x_real = self.extract_input_tensor(batch)
        self.reconstruct_calls.append((vqvae, x_real))
        return x_real + self.recon_offset


class RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []

    def __call__(self, name: str, value: Any, **kwargs: Any) -> None:
        self.calls.append((name, value, kwargs))


class DummyVQVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.decoder = SimpleNamespace(decoder=torch.nn.Sequential(torch.nn.Conv3d(1, 1, 1)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recon = x * self.weight
        vq_loss = self.weight.square()
        return recon, vq_loss


def test_training_step_runs_manual_optimization_and_logs_train_metrics() -> None:
    logger = RecordingLogger()
    reconstructor = RecordingReconstructor()
    loss_fn = RecordingLoss()
    strategy = RecordingStrategy()
    backward_calls: list[torch.Tensor] = []
    service = VQVAETrainingSteps(
        vqvae=DummyVQVAE(),
        loss_fn=loss_fn,
        gan_strategy=strategy,
        reconstructor=reconstructor,
        log_fn=logger,
        manual_backward_fn=backward_calls.append,
    )
    opt_g = RecordingOptimizer()
    opt_d = RecordingOptimizer()
    batch = (torch.ones(2, 1, 4, 4, 4), torch.zeros(2, 1, 4, 4, 4))

    result = service.training_step(batch, batch_idx=0, optimizers=[opt_g, opt_d], global_step=3)

    assert result is None
    assert len(backward_calls) == 2
    assert opt_g.zero_grad_calls == 1
    assert opt_d.zero_grad_calls == 1
    assert opt_g.step_calls == 1
    assert opt_d.step_calls == 1
    assert len(strategy.generator_calls) == 1
    assert len(strategy.discriminator_calls) == 1
    assert loss_fn.calls[0]["optimizer_idx"] == 0
    assert loss_fn.calls[1]["optimizer_idx"] == 1
    assert loss_fn.calls[1]["reconstructions"].requires_grad is False
    logged_names = {name for name, _, _ in logger.calls}
    assert "train/total_loss" in logged_names
    assert "train/disc_loss" in logged_names
    assert "train/x_recon_min" in logged_names
    assert "train/x_real_max" in logged_names


def test_get_decoder_last_layer_returns_final_decoder_weight() -> None:
    service = VQVAETrainingSteps(
        vqvae=DummyVQVAE(),
        loss_fn=RecordingLoss(),
        gan_strategy=RecordingStrategy(),
        reconstructor=RecordingReconstructor(),
        log_fn=RecordingLogger(),
        manual_backward_fn=lambda loss: None,
    )

    last_layer = service.get_decoder_last_layer()

    assert isinstance(last_layer, torch.nn.Parameter)
    assert last_layer.shape == torch.Size([1, 1, 1, 1, 1])


def test_reconstruction_step_reuses_reconstructor_and_logs_validation_metrics() -> None:
    logger = RecordingLogger()
    reconstructor = RecordingReconstructor(recon_offset=0.25)
    service = VQVAETrainingSteps(
        vqvae=DummyVQVAE(),
        loss_fn=RecordingLoss(),
        gan_strategy=RecordingStrategy(),
        reconstructor=reconstructor,
        log_fn=logger,
        manual_backward_fn=lambda loss: None,
    )
    batch = torch.zeros(1, 1, 4, 4, 4)

    outputs = service.reconstruction_step(batch, split="val")

    assert reconstructor.reconstruct_calls
    assert outputs["x_real"].shape == batch.shape
    assert outputs["x_recon"].shape == batch.shape
    logged_names = {name for name, _, _ in logger.calls}
    assert "val_perceptual_loss" in logged_names
    assert "val_rec_loss" in logged_names


def test_test_reconstruction_step_reports_sliding_window_usage_and_resets_cuda_stats(
    monkeypatch,
) -> None:
    logger = RecordingLogger()
    reconstructor = RecordingReconstructor(recon_offset=0.25, inferer=object())
    service = VQVAETrainingSteps(
        vqvae=DummyVQVAE(),
        loss_fn=RecordingLoss(),
        gan_strategy=RecordingStrategy(),
        reconstructor=reconstructor,
        log_fn=logger,
        manual_backward_fn=lambda loss: None,
    )
    reset_calls: list[str] = []
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda: reset_calls.append("reset"))

    outputs = service.reconstruction_step(torch.zeros(1, 1, 4, 4, 4), split="test")

    assert outputs["use_sliding_window"] is True
    assert outputs["inference_time"] == 0.0
    assert isinstance(outputs["loss"], float)
    assert reset_calls == ["reset"]
