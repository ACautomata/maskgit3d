from typing import Any
from unittest.mock import MagicMock

import torch
from lightning.pytorch import LightningModule

from maskgit3d.callbacks.maskgit_metrics import MaskGITMetricsCallback


class _MaskGITStub:
    def __init__(self, sliding_window_enabled: bool = False) -> None:
        self.sliding_window_cfg = {"enabled": sliding_window_enabled}


class SimpleModel(LightningModule):
    def __init__(self, sliding_window_enabled: bool = False) -> None:
        super().__init__()
        self.logged_values: list[tuple[str, torch.Tensor]] = []
        self._callback_payloads: dict[str, dict[str, Any]] = {}
        self.maskgit = _MaskGITStub(sliding_window_enabled=sliding_window_enabled)

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


class TestMaskGITMetricsCallback:
    def test_callback_initialization_default(self) -> None:
        callback = MaskGITMetricsCallback()
        assert callback.log_every_n_steps == 1
        assert callback.log_val_every_n_batches == 1
        assert callback._train_step_count == 0

    def test_callback_initialization_custom(self) -> None:
        callback = MaskGITMetricsCallback(log_every_n_steps=10, log_val_every_n_batches=5)
        assert callback.log_every_n_steps == 10
        assert callback.log_val_every_n_batches == 5

    def test_on_train_batch_end_with_none_outputs(self) -> None:
        callback = MaskGITMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()

        callback.on_train_batch_end(trainer, model, None, None, 0)
        assert callback._train_step_count == 1
        assert model.logged_values == []

    def test_on_train_batch_end_uses_cached_payload(self) -> None:
        callback = MaskGITMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()
        model.save_callback_payload(
            "train",
            {
                "masked_logits": torch.tensor([[0.1, 0.9], [0.7, 0.3]]),
                "masked_targets": torch.tensor([1, 0]),
                "mask_ratio": 0.5,
            },
        )

        loss = torch.tensor(1.5)
        callback.on_train_batch_end(trainer, model, loss, None, 0)

        logged = dict(model.logged_values)
        assert torch.equal(logged["loss:train"], loss)
        assert torch.isclose(logged["mask_acc:train"], torch.tensor(1.0))
        assert torch.isclose(logged["mask_ratio:train"], torch.tensor(0.5))

    def test_on_train_batch_end_without_cached_payload_logs_only_loss(self) -> None:
        callback = MaskGITMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()

        callback.on_train_batch_end(trainer, model, torch.tensor(2.0), None, 0)

        logged_names = [name for name, _ in model.logged_values]
        assert logged_names == ["loss:train"]

    def test_on_train_batch_end_log_every_n_steps(self) -> None:
        callback = MaskGITMetricsCallback(log_every_n_steps=2)
        trainer = MagicMock()
        model = SimpleModel()

        model.save_callback_payload(
            "train",
            {
                "masked_logits": torch.tensor([[0.1, 0.9]]),
                "masked_targets": torch.tensor([1]),
                "mask_ratio": 0.25,
            },
        )
        callback.on_train_batch_end(trainer, model, torch.tensor(1.0), None, 0)
        assert callback._train_step_count == 1
        assert model.logged_values == []

        model.save_callback_payload(
            "train",
            {
                "masked_logits": torch.tensor([[0.1, 0.9]]),
                "masked_targets": torch.tensor([1]),
                "mask_ratio": 0.25,
            },
        )
        callback.on_train_batch_end(trainer, model, torch.tensor(1.0), None, 1)
        assert callback._train_step_count == 2
        assert len(model.logged_values) == 3

    def test_on_validation_batch_end_logs_from_eval_outputs(self) -> None:
        callback = MaskGITMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()
        outputs = {
            "x_real": torch.randn(1, 1, 8, 8, 8),
            "generated_images": torch.randn(1, 1, 8, 8, 8),
            "masked_logits": torch.tensor([[0.1, 0.9], [0.7, 0.3]]),
            "masked_targets": torch.tensor([1, 0]),
            "mask_ratio": 0.5,
        }

        callback.on_validation_batch_end(trainer, model, outputs, None, 0)

        logged = dict(model.logged_values)
        assert "val_loss" in logged
        assert torch.isclose(logged["val_mask_acc"], torch.tensor(1.0))
        assert torch.isclose(logged["val_mask_ratio"], torch.tensor(0.5))
        assert torch.isclose(logged["sample_shape:val"], torch.tensor(1.0))

    def test_on_test_batch_end_logs_from_eval_outputs(self) -> None:
        callback = MaskGITMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel(sliding_window_enabled=True)
        outputs = {
            "x_real": torch.randn(1, 1, 8, 8, 8),
            "generated_images": torch.randn(1, 1, 8, 8, 8),
            "masked_logits": torch.tensor([[0.1, 0.9], [0.7, 0.3]]),
            "masked_targets": torch.tensor([1, 0]),
            "mask_ratio": 0.5,
        }

        callback.on_test_batch_end(trainer, model, outputs, None, 0)

        logged = dict(model.logged_values)
        assert "loss:test" in logged
        assert torch.isclose(logged["mask_acc:test"], torch.tensor(1.0))
        assert torch.isclose(logged["sliding_window_enabled:test"], torch.tensor(1.0))

    def test_on_train_epoch_start_resets_counter(self) -> None:
        callback = MaskGITMetricsCallback()
        callback._train_step_count = 50

        model = SimpleModel()
        callback.on_train_epoch_start(MagicMock(), model)

        assert callback._train_step_count == 0
