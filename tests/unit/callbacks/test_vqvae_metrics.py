from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer

from maskgit3d.callbacks.vqvae_metrics import VQVAEMetricsCallback


class SimpleModel(LightningModule):
    def __init__(self, use_perceptual: bool = False):
        super().__init__()
        self.use_perceptual_flag = use_perceptual
        self.logged_values: list[tuple[str, torch.Tensor]] = []
        self._callback_payloads: dict[str, dict[str, Any]] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def training_step(self, batch: object, batch_idx: int) -> dict[str, torch.Tensor]:
        return {}

    def save_callback_payload(self, stage: str, payload: dict[str, Any]) -> None:
        self._callback_payloads[stage] = payload

    def pop_callback_payload(self, stage: str) -> dict[str, Any] | None:
        return self._callback_payloads.pop(stage, None)

    def log(
        self,
        name: str,
        value: object,
        prog_bar: bool = False,
        logger: bool | None = None,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        reduce_fx: str | object = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: object | None = None,
        add_dataloader_idx: bool = True,
        batch_size: int | None = None,
        metric_attribute: str | None = None,
        rank_zero_only: bool = False,
    ) -> None:
        del (
            prog_bar,
            logger,
            on_step,
            on_epoch,
            reduce_fx,
            enable_graph,
            sync_dist,
            sync_dist_group,
            add_dataloader_idx,
            batch_size,
            metric_attribute,
            rank_zero_only,
        )
        if isinstance(value, torch.Tensor):
            logged_value = value.detach().cpu()
        elif isinstance(value, bool):
            logged_value = torch.tensor(float(value))
        elif isinstance(value, int | float):
            logged_value = torch.tensor(float(value))
        else:
            return
        self.logged_values.append((name, logged_value))


class MockLossFn(nn.Module):
    def __init__(self, use_perceptual: bool = False):
        self.use_perceptual = use_perceptual
        self.perceptual_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        super().__init__()
        self.perceptual_loss = self._perceptual_loss if use_perceptual else None

    def _perceptual_loss(self, x_recon: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        del x_recon, x_real
        return torch.tensor(0.1)

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        vq_loss: torch.Tensor,
        optimizer_idx: int = 0,
        global_step: int = 0,
        last_layer: torch.Tensor | None = None,
        split: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del inputs, reconstructions, vq_loss, global_step, last_layer
        if optimizer_idx == 0:
            loss = torch.tensor(1.0)
            return loss, {f"{split}/total_loss": loss, f"{split}/rec_loss": torch.tensor(0.5)}
        loss = torch.tensor(0.5)
        return loss, {f"{split}/disc_loss": loss}


class TestVQVAEMetricsCallback:
    def test_callback_initialization_default(self) -> None:
        callback = VQVAEMetricsCallback()
        assert callback.log_every_n_steps == 1
        assert callback.log_val_every_n_batches == 1
        assert callback._train_step_count == 0

    def test_callback_initialization_custom(self) -> None:
        callback = VQVAEMetricsCallback(log_every_n_steps=10, log_val_every_n_batches=5)
        assert callback.log_every_n_steps == 10
        assert callback.log_val_every_n_batches == 5

    def test_on_train_batch_end_with_none_outputs(self) -> None:
        callback = VQVAEMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()
        model.loss_fn = MockLossFn()

        callback.on_train_batch_end(trainer, model, None, None, 0)
        assert callback._train_step_count == 1
        assert model.logged_values == []

    def test_on_train_batch_end_uses_cached_payload_with_tensor_loss(self) -> None:
        callback = VQVAEMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()
        model.loss_fn = MockLossFn()
        model.save_callback_payload(
            "train",
            {
                "x_real": torch.randn(1, 1, 8, 8, 8),
                "x_recon": torch.randn(1, 1, 8, 8, 8),
                "vq_loss": torch.tensor(0.1),
                "last_layer": torch.tensor(1.0),
            },
        )

        callback.on_train_batch_end(trainer, model, torch.tensor(1.0), None, 0)

        logged_names = [name for name, _ in model.logged_values]
        assert "train/total_loss" in logged_names
        assert "train/rec_loss" in logged_names
        assert "train/disc_loss" in logged_names

    def test_on_train_batch_end_with_missing_payload(self) -> None:
        callback = VQVAEMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()
        model.loss_fn = MockLossFn()

        callback.on_train_batch_end(trainer, model, torch.tensor(1.0), None, 0)
        assert model.logged_values == []

    def test_on_train_batch_end_log_every_n_steps(self) -> None:
        callback = VQVAEMetricsCallback(log_every_n_steps=2)
        trainer = MagicMock()
        model = SimpleModel()
        model.loss_fn = MockLossFn()

        payload = {
            "x_real": torch.randn(1, 1, 8, 8, 8),
            "x_recon": torch.randn(1, 1, 8, 8, 8),
            "vq_loss": torch.tensor(0.1),
            "last_layer": torch.tensor(1.0),
        }
        model.save_callback_payload("train", payload)
        callback.on_train_batch_end(trainer, model, torch.tensor(1.0), None, 0)
        assert callback._train_step_count == 1
        assert model.logged_values == []

        model.save_callback_payload("train", payload)
        callback.on_train_batch_end(trainer, model, torch.tensor(1.0), None, 1)
        assert callback._train_step_count == 2
        assert len(model.logged_values) == 3

    def test_on_validation_batch_end_logs_rec_and_fid_metrics(self) -> None:
        callback = VQVAEMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()
        model.loss_fn = MockLossFn()
        x_real = torch.zeros(1, 1, 8, 8, 8)
        x_recon = torch.ones(1, 1, 8, 8, 8) * 0.2
        outputs = {
            "x_real": x_real,
            "x_recon": x_recon,
            "vq_loss": torch.tensor(0.1),
        }

        callback.on_validation_batch_end(trainer, model, outputs, None, 0)

        logged = dict(model.logged_values)
        assert "val_rec_loss" in logged
        assert "val_fid" in logged

    def test_on_validation_batch_end_with_fid(self) -> None:
        callback = VQVAEMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel(use_perceptual=True)
        model.loss_fn = MockLossFn(use_perceptual=True)
        outputs = {
            "x_real": torch.randn(1, 1, 8, 8, 8),
            "x_recon": torch.randn(1, 1, 8, 8, 8),
            "vq_loss": torch.tensor(0.1),
        }

        callback.on_validation_batch_end(trainer, model, outputs, None, 0)

        logged = dict(model.logged_values)
        assert "val_fid" in logged

    def test_on_test_batch_end_logs_metrics_from_outputs(self) -> None:
        callback = VQVAEMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()
        outputs = {
            "x_real": torch.zeros(1, 1, 8, 8, 8),
            "x_recon": torch.ones(1, 1, 8, 8, 8) * 0.2,
            "vq_loss": torch.tensor(0.1),
            "inference_time": torch.tensor(0.3),
            "use_sliding_window": True,
        }

        callback.on_test_batch_end(trainer, model, outputs, None, 0)

        logged_names = [name for name, _ in model.logged_values]
        assert "loss_l1:test" in logged_names
        assert "loss_vq:test" in logged_names
        assert "fid:test" in logged_names
        assert "inference_time:test" in logged_names
        assert "sliding_window_enabled:test" in logged_names

    def test_on_train_epoch_start_resets_counter(self) -> None:
        callback = VQVAEMetricsCallback()
        callback._train_step_count = 50

        model = SimpleModel()
        trainer = MagicMock(spec=Trainer)
        callback.on_train_epoch_start(trainer, model)

        assert callback._train_step_count == 0
