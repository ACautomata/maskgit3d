from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from maskgit3d.infrastructure.training.callbacks import (
    EarlyStopping,
    FabricEarlyStopping,
    FabricMetricsLogger,
    FabricModelCheckpoint,
    FabricNaNMonitor,
    LightningEarlyStopping,
    LightningModelCheckpoint,
    MetricsLogger,
    ModelCheckpoint,
    NaNMonitor,
)


def _make_trainer(metrics: dict, optimizer: torch.optim.Optimizer | None = None) -> MagicMock:
    trainer = MagicMock()
    trainer.callback_metrics = metrics
    trainer.optimizers = [optimizer] if optimizer is not None else []
    return trainer


def _make_fabric(metrics: dict | None = None, is_global_zero: bool = True) -> MagicMock:
    fabric = MagicMock()
    fabric._current_metrics = metrics or {}
    fabric.is_global_zero = is_global_zero
    fabric.save = MagicMock()
    return fabric


class TestModelCheckpointProperties:
    def test_current_epoch_property_returns_internal_value(self, tmp_path: Path) -> None:
        callback = ModelCheckpoint(dirpath=str(tmp_path))
        callback._current_epoch = 7
        assert callback.current_epoch == 7

    def test_best_scores_property_returns_only_score_values(self, tmp_path: Path) -> None:
        callback = ModelCheckpoint(dirpath=str(tmp_path))
        callback._best_scores = [(0.2, tmp_path / "a.ckpt"), (0.1, tmp_path / "b.ckpt")]
        assert callback.best_scores == [0.2, 0.1]

    def test_best_model_path_property_returns_path_or_empty(self, tmp_path: Path) -> None:
        callback = ModelCheckpoint(dirpath=str(tmp_path))
        assert callback.best_model_path == ""
        best_path = tmp_path / "best.ckpt"
        callback._best_scores = [(0.1, best_path)]
        assert callback.best_model_path == str(best_path)

    def test_best_model_score_property_returns_tensor_or_none(self, tmp_path: Path) -> None:
        callback = ModelCheckpoint(dirpath=str(tmp_path))
        assert callback.best_model_score is None
        callback._best_scores = [(0.123, tmp_path / "best.ckpt")]
        score = callback.best_model_score
        assert isinstance(score, torch.Tensor)
        assert score.item() == pytest.approx(0.123)


class TestLightningWrappers:
    def test_lightning_model_checkpoint_builds_default_filename(self, tmp_path: Path) -> None:
        callback = LightningModelCheckpoint(monitor="val_acc", filename=None, dirpath=tmp_path)
        assert callback.filename == "{epoch:02d}-{val_acc:.4f}"

    def test_lightning_model_checkpoint_uses_custom_filename(self, tmp_path: Path) -> None:
        callback = LightningModelCheckpoint(
            monitor="val_acc",
            filename="custom-name",
            dirpath=tmp_path,
        )
        assert callback.filename == "custom-name"

    def test_lightning_early_stopping_init(self) -> None:
        callback = LightningEarlyStopping(
            monitor="val_loss", mode="min", patience=3, min_delta=0.05
        )
        assert callback.monitor == "val_loss"
        assert callback.mode == "min"
        assert callback.patience == 3
        assert callback.min_delta == -0.05


class TestEarlyStoppingFlow:
    def test_on_fit_start_initializes_state(self) -> None:
        callback = EarlyStopping(mode="min")
        trainer = _make_trainer({})
        model = torch.nn.Linear(2, 2)
        callback.on_fit_start(trainer, model)
        assert callback._early_stopping.wait_count == 0
        assert callback._early_stopping.stopped_epoch == 0
        assert callback.should_stop is False
        assert callback.best_score is None

    def test_validation_epoch_end_sets_should_stop_on_non_finite(self) -> None:
        callback = EarlyStopping(check_finite=True)
        trainer = _make_trainer({"val_loss": torch.tensor(float("nan"))})
        callback.on_validation_epoch_end(trainer, torch.nn.Linear(2, 2))
        assert callback.should_stop is True

    def test_validation_epoch_end_initializes_best_when_none(self) -> None:
        callback = EarlyStopping(monitor="val_loss", mode="min")
        callback._early_stopping.best_score = None
        trainer = _make_trainer({"val_loss": torch.tensor(0.8)})
        callback.on_validation_epoch_end(trainer, torch.nn.Linear(2, 2))
        best_score = callback._early_stopping.best_score
        assert isinstance(best_score, torch.Tensor)
        assert best_score.item() == pytest.approx(0.8)
        assert callback.counter == 0

    def test_validation_epoch_end_improvement_resets_wait_count(self) -> None:
        callback = EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=0.0)
        callback._early_stopping.best_score = torch.tensor(0.8)
        callback._early_stopping.wait_count = 2
        trainer = _make_trainer({"val_loss": torch.tensor(0.6)})
        callback.on_validation_epoch_end(trainer, torch.nn.Linear(2, 2))
        assert callback._early_stopping.best_score.item() == pytest.approx(0.6)
        assert callback.counter == 0

    def test_validation_epoch_end_stops_after_patience_exceeded(self) -> None:
        callback = EarlyStopping(monitor="val_loss", mode="min", patience=2, min_delta=0.0)
        callback._early_stopping.best_score = torch.tensor(0.5)
        trainer = _make_trainer({"val_loss": torch.tensor(0.7)})
        callback.on_validation_epoch_end(trainer, torch.nn.Linear(2, 2))
        assert callback.should_stop is False
        assert callback.counter == 1
        callback.on_validation_epoch_end(trainer, torch.nn.Linear(2, 2))
        assert callback.should_stop is True
        assert callback.counter == 2

    def test_best_score_property_handles_inf_and_valid_values(self) -> None:
        callback = EarlyStopping(mode="min")
        callback._early_stopping.best_score = torch.tensor(float("inf"))
        assert callback.best_score is None
        callback._early_stopping.best_score = torch.tensor(0.42)
        assert callback.best_score == pytest.approx(0.42)


class TestNaNMonitorFlow:
    def test_on_fit_start_resets_batch_count(self) -> None:
        callback = NaNMonitor(check_interval=1)
        callback._batch_count = 9
        callback.on_fit_start(_make_trainer({}), torch.nn.Linear(2, 2))
        assert callback.batch_count == 0

    def test_on_train_batch_end_skips_when_not_at_interval(self) -> None:
        callback = NaNMonitor(check_interval=3, raise_on_nan=False)
        callback.on_train_batch_end(
            _make_trainer({}), torch.nn.Linear(2, 2), {"loss": 1.0}, None, 0
        )
        assert callback.batch_count == 1

    def test_on_train_batch_end_handles_nan_loss_in_dict(self) -> None:
        callback = NaNMonitor(check_interval=1, raise_on_nan=False, on_nan_action="log")
        with patch.object(callback, "_handle_nan") as mocked_handle:
            callback.on_train_batch_end(
                _make_trainer({}),
                torch.nn.Linear(2, 2),
                {"loss": torch.tensor(float("nan"))},
                None,
                0,
            )
        mocked_handle.assert_called_once()
        args = mocked_handle.call_args[0]
        assert args[0] == "loss"
        assert math.isnan(args[1])

    def test_on_train_batch_end_handles_tensor_outputs(self) -> None:
        callback = NaNMonitor(check_interval=1, raise_on_nan=False, on_nan_action="log")
        with patch.object(callback, "_handle_nan") as mocked_handle:
            callback.on_train_batch_end(
                _make_trainer({}),
                torch.nn.Linear(2, 2),
                torch.tensor(float("nan")),
                None,
                0,
            )
        mocked_handle.assert_called_once()
        args = mocked_handle.call_args[0]
        assert args[0] == "loss"

    def test_on_train_batch_end_handles_nan_gradient(self) -> None:
        callback = NaNMonitor(check_interval=1, raise_on_nan=False, on_nan_action="log")
        model = torch.nn.Linear(2, 2)
        model.weight.grad = torch.full_like(model.weight, float("nan"))
        with patch.object(callback, "_handle_nan") as mocked_handle:
            callback.on_train_batch_end(
                _make_trainer({}), model, {"loss": torch.tensor(1.0)}, None, 0
            )
        mocked_handle.assert_called_once_with("gradient", None)

    def test_handle_nan_error_action_raises(self) -> None:
        callback = NaNMonitor(check_interval=1, raise_on_nan=False, on_nan_action="error")
        with pytest.raises(RuntimeError, match="Training stopped due to NaN/Inf in loss"):
            callback._handle_nan("loss", 1.0)

    def test_handle_nan_log_action_warns_only(self, caplog: pytest.LogCaptureFixture) -> None:
        callback = NaNMonitor(check_interval=1, raise_on_nan=False, on_nan_action="log")
        callback._handle_nan("loss", 1.0)
        assert "NaN/Inf detected in loss: 1.0" in caplog.text

    def test_handle_nan_skip_action_continues(self, caplog: pytest.LogCaptureFixture) -> None:
        callback = NaNMonitor(check_interval=1, raise_on_nan=False, on_nan_action="skip")
        callback._handle_nan("gradient", None)
        assert "NaN/Inf detected" not in caplog.text


class TestMetricsLoggerFlow:
    def test_on_train_epoch_end_records_metrics_and_saves(self, tmp_path: Path) -> None:
        callback = MetricsLogger(log_dir=str(tmp_path), log_interval=1)
        trainer = _make_trainer({"train_loss": torch.tensor(0.2), "step": 1, "note": "ignore"})
        callback.on_train_epoch_end(trainer, torch.nn.Linear(2, 2))
        assert callback.history["train_loss"] == [0.20000000298023224]
        assert callback.history["step"] == [1]
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "metrics.csv").exists()

    def test_on_train_epoch_end_skips_when_not_at_interval(self, tmp_path: Path) -> None:
        callback = MetricsLogger(log_dir=str(tmp_path), log_interval=2)
        callback.on_train_epoch_end(
            _make_trainer({"train_loss": torch.tensor(0.3)}), torch.nn.Linear(2, 2)
        )
        assert callback.history == {}
        assert not (tmp_path / "metrics.json").exists()

    def test_on_fit_end_saves_metrics(self, tmp_path: Path) -> None:
        callback = MetricsLogger(log_dir=str(tmp_path), log_interval=10)
        callback._history = {"train_loss": [0.4]}
        callback.on_fit_end(_make_trainer({}), torch.nn.Linear(2, 2))
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "metrics.csv").exists()

    def test_save_metrics_writes_json_and_csv(self, tmp_path: Path) -> None:
        callback = MetricsLogger(log_dir=str(tmp_path), log_interval=1)
        callback._history = {"a": [1.0, 2.0], "b": [3.0]}
        callback._save_metrics()

        with (tmp_path / "metrics.json").open() as f:
            data = json.load(f)
        assert data == {"a": [1.0, 2.0], "b": [3.0]}

        csv_lines = (tmp_path / "metrics.csv").read_text().strip().splitlines()
        assert csv_lines[0] == "epoch,a,b"
        assert csv_lines[1] == "0,1.0,3.0"
        assert csv_lines[2] == "1,2.0,"

    def test_history_property_and_get_history_copy(self, tmp_path: Path) -> None:
        callback = MetricsLogger(log_dir=str(tmp_path), log_interval=1)
        callback._history = {"metric": [1.0]}
        assert callback.history == {"metric": [1.0]}
        copied = callback.get_history()
        assert copied == {"metric": [1.0]}
        assert copied is not callback.history


class TestFabricModelCheckpoint:
    def test_on_validation_epoch_end_uses_fabric_metrics_and_updates(self, tmp_path: Path) -> None:
        callback = FabricModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            dirpath=str(tmp_path),
        )
        fabric = _make_fabric({"val_loss": 0.4})

        with (
            patch.object(callback, "_save_checkpoint_fabric") as save_mock,
            patch.object(callback, "_update_best_checkpoints_fabric") as update_mock,
        ):
            callback.on_validation_epoch_end(fabric, MagicMock(), MagicMock())

        save_mock.assert_called_once()
        update_mock.assert_called_once()
        assert callback.current_epoch == 1

    def test_on_validation_epoch_end_returns_early_with_no_score(self, tmp_path: Path) -> None:
        callback = FabricModelCheckpoint(monitor="val_loss", dirpath=str(tmp_path))
        fabric = _make_fabric({})
        with patch.object(callback, "_save_checkpoint_fabric") as save_mock:
            callback.on_validation_epoch_end(fabric, MagicMock(), MagicMock())
        save_mock.assert_not_called()
        assert callback.current_epoch == 0

    def test_update_best_checkpoints_saves_and_removes_excess(self, tmp_path: Path) -> None:
        callback = FabricModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            dirpath=str(tmp_path),
        )
        old_file = tmp_path / "old.ckpt"
        old_file.write_text("old")
        callback._best_scores = [(0.6, old_file)]

        with patch.object(callback, "_save_checkpoint_fabric") as save_mock:
            callback._update_best_checkpoints_fabric(_make_fabric(), MagicMock(), MagicMock(), 0.5)

        save_mock.assert_called_once()
        assert len(callback._best_scores) == 1
        assert callback._best_scores[0][0] == 0.5
        assert old_file.exists() is False

    def test_save_checkpoint_fabric_calls_fabric_save(self, tmp_path: Path) -> None:
        callback = FabricModelCheckpoint(dirpath=str(tmp_path))
        callback._current_epoch = 3
        fabric = _make_fabric()
        model = MagicMock()
        optimizer = MagicMock()

        callback._save_checkpoint_fabric(fabric, model, optimizer, "checkpoint")

        expected_path = tmp_path / "checkpoint.ckpt"
        fabric.save.assert_called_once()
        call_path, payload = fabric.save.call_args[0]
        assert call_path == expected_path
        assert payload["epoch"] == 3
        assert payload["model"] is model
        assert payload["optimizer"] is optimizer

    def test_on_fit_start_resets_state(self, tmp_path: Path) -> None:
        callback = FabricModelCheckpoint(dirpath=str(tmp_path))
        callback._current_epoch = 99
        callback._best_scores = [(0.1, tmp_path / "x.ckpt")]
        callback.on_fit_start(_make_fabric())
        assert callback.current_epoch == 0
        assert callback.best_scores == []


class TestFabricEarlyStopping:
    def test_on_validation_epoch_end_returns_when_score_missing(self) -> None:
        callback = FabricEarlyStopping(monitor="val_loss")
        callback.on_validation_epoch_end(_make_fabric({}), MagicMock(), MagicMock())
        assert callback.should_stop is False

    def test_on_validation_epoch_end_sets_should_stop_on_non_finite(self) -> None:
        callback = FabricEarlyStopping(monitor="val_loss", check_finite=True)
        callback.on_validation_epoch_end(
            _make_fabric({"val_loss": float("nan")}),
            MagicMock(),
            MagicMock(),
        )
        assert callback.should_stop is True

    def test_on_validation_epoch_end_initializes_best_when_none(self) -> None:
        callback = FabricEarlyStopping(monitor="val_loss", mode="min")
        callback._early_stopping.best_score = None
        callback.on_validation_epoch_end(_make_fabric({"val_loss": 0.7}), MagicMock(), MagicMock())
        best_score = callback._early_stopping.best_score
        assert isinstance(best_score, torch.Tensor)
        assert best_score.item() == pytest.approx(0.7)
        assert callback.counter == 0

    def test_on_validation_epoch_end_improvement_resets_wait_count(self) -> None:
        callback = FabricEarlyStopping(monitor="val_loss", mode="min", patience=3)
        callback._early_stopping.best_score = torch.tensor(0.8)
        callback._early_stopping.wait_count = 2
        callback.on_validation_epoch_end(_make_fabric({"val_loss": 0.5}), MagicMock(), MagicMock())
        assert callback._early_stopping.best_score.item() == pytest.approx(0.5)
        assert callback.counter == 0

    def test_on_validation_epoch_end_stops_after_patience(self) -> None:
        callback = FabricEarlyStopping(monitor="val_loss", mode="min", patience=2)
        callback._early_stopping.best_score = torch.tensor(0.4)
        callback.on_validation_epoch_end(_make_fabric({"val_loss": 0.9}), MagicMock(), MagicMock())
        assert callback.should_stop is False
        callback.on_validation_epoch_end(_make_fabric({"val_loss": 0.9}), MagicMock(), MagicMock())
        assert callback.should_stop is True

    def test_on_fit_start_initializes_state(self) -> None:
        callback = FabricEarlyStopping(mode="max")
        callback._early_stopping.wait_count = 7
        callback._early_stopping.stopped_epoch = 3
        callback.should_stop = True
        callback.on_fit_start(_make_fabric())
        assert callback._early_stopping.wait_count == 0
        assert callback._early_stopping.stopped_epoch == 0
        assert callback.should_stop is False
        assert callback.best_score is None


class TestFabricNaNMonitor:
    def test_on_fit_start_resets_batch_count(self) -> None:
        callback = FabricNaNMonitor(check_interval=1)
        callback._batch_count = 5
        callback.on_fit_start(_make_fabric())
        assert callback.batch_count == 0

    def test_on_train_batch_end_nan_loss_calls_handle_nan(self) -> None:
        callback = FabricNaNMonitor(check_interval=1, raise_on_nan=False, on_nan_action="log")
        with patch.object(callback, "_handle_nan") as mocked_handle:
            callback.on_train_batch_end(
                _make_fabric(), torch.nn.Linear(2, 2), None, None, 0, float("nan")
            )
        mocked_handle.assert_called_once()
        args = mocked_handle.call_args[0]
        assert args[0] == "loss"
        assert math.isnan(args[1])

    def test_on_train_batch_end_skips_on_interval(self) -> None:
        callback = FabricNaNMonitor(check_interval=2, raise_on_nan=False, on_nan_action="log")
        with patch.object(callback, "_handle_nan") as mocked_handle:
            callback.on_train_batch_end(
                _make_fabric(), torch.nn.Linear(2, 2), None, None, 0, float("nan")
            )
        mocked_handle.assert_not_called()
        assert callback.batch_count == 1

    def test_on_train_batch_end_nan_gradient_calls_handle_nan(self) -> None:
        callback = FabricNaNMonitor(check_interval=1, raise_on_nan=False, on_nan_action="log")
        model = torch.nn.Linear(2, 2)
        model.weight.grad = torch.full_like(model.weight, float("nan"))
        with patch.object(callback, "_handle_nan") as mocked_handle:
            callback.on_train_batch_end(
                _make_fabric(is_global_zero=True), model, None, None, 0, 0.1
            )
        mocked_handle.assert_called_once_with("gradient", None)


class TestFabricMetricsLogger:
    def test_on_train_epoch_end_records_metrics(self, tmp_path: Path) -> None:
        callback = FabricMetricsLogger(log_dir=str(tmp_path), log_interval=1)
        fabric = _make_fabric({"train_loss": 0.25, "val_acc": 0.9})
        callback.on_train_epoch_end(fabric, MagicMock(), MagicMock())
        assert callback.history["train_loss"] == [0.25]
        assert callback.history["val_acc"] == [0.9]
        assert (tmp_path / "metrics.json").exists()

    def test_on_train_epoch_end_skips_when_not_at_interval(self, tmp_path: Path) -> None:
        callback = FabricMetricsLogger(log_dir=str(tmp_path), log_interval=2)
        callback.on_train_epoch_end(_make_fabric({"train_loss": 0.1}), MagicMock(), MagicMock())
        assert callback.history == {}

    def test_on_fit_end_saves_metrics(self, tmp_path: Path) -> None:
        callback = FabricMetricsLogger(log_dir=str(tmp_path), log_interval=100)
        callback._history = {"loss": [1.0]}
        callback.on_fit_end(_make_fabric())
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "metrics.csv").exists()
