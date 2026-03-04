"""Tests for training callbacks."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from maskgit3d.infrastructure.training.callbacks import (
    EarlyStopping,
    MetricsLogger,
    ModelCheckpoint,
    NaNMonitor,
)


class TestModelCheckpoint:
    """Tests for ModelCheckpoint callback."""

    def test_checkpoint_creation(self, tmp_path):
        """Test ModelCheckpoint initialization."""
        callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            dirpath=str(tmp_path),
        )
        assert callback.monitor == "val_loss"
        assert callback.mode == "min"
        assert callback.save_top_k == 3
        assert callback.dirpath.exists()

    def test_checkpoint_saves_last(self, tmp_path):
        """Test that last checkpoint is saved."""
        callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_last=True,
            dirpath=str(tmp_path),
        )

        mock_fabric = MagicMock()
        mock_fabric.is_global_zero = True
        mock_fabric._current_metrics = {"val_loss": 0.5}

        mock_model = MagicMock()
        mock_optimizer = MagicMock()

        callback.on_validation_epoch_end(mock_fabric, mock_model, mock_optimizer)

        assert callback.current_epoch == 1

    def test_checkpoint_saves_best(self, tmp_path):
        """Test that best checkpoints are tracked."""
        callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_best=True,
            dirpath=str(tmp_path),
        )

        mock_fabric = MagicMock()
        mock_fabric.is_global_zero = True
        mock_model = MagicMock()
        mock_optimizer = MagicMock()

        # First validation epoch
        mock_fabric._current_metrics = {"val_loss": 0.5}
        callback.on_validation_epoch_end(mock_fabric, mock_model, mock_optimizer)

        # Second validation epoch (better)
        mock_fabric._current_metrics = {"val_loss": 0.3}
        callback.on_validation_epoch_end(mock_fabric, mock_model, mock_optimizer)

        assert len(callback.best_scores) > 0


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_early_stopping_creation(self):
        """Test EarlyStopping initialization."""
        callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            min_delta=0.01,
        )
        assert callback.monitor == "val_loss"
        assert callback.mode == "min"
        assert callback.patience == 5
        assert callback.min_delta == 0.01
        assert not callback.should_stop

    def test_early_stopping_triggers(self):
        """Test that early stopping triggers after patience."""
        callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=2,
            min_delta=0.01,
        )

        mock_fabric = MagicMock()
        mock_fabric.is_global_zero = True
        mock_model = MagicMock()
        mock_optimizer = MagicMock()

        # Initial validation
        mock_fabric._current_metrics = {"val_loss": 1.0}
        callback.on_validation_epoch_end(mock_fabric, mock_model, mock_optimizer)
        assert not callback.should_stop

        # Worse validation (no improvement)
        mock_fabric._current_metrics = {"val_loss": 1.1}
        callback.on_validation_epoch_end(mock_fabric, mock_model, mock_optimizer)
        assert not callback.should_stop

        # Another worse validation
        callback.on_validation_epoch_end(mock_fabric, mock_model, mock_optimizer)
        assert callback.should_stop

    def test_early_stopping_reset(self):
        """Test that early stopping resets properly."""
        callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)

        mock_fabric = MagicMock()
        callback.on_fit_start(mock_fabric)

        assert callback.best_score is None
        assert callback.counter == 0
        assert not callback.should_stop


class TestNaNMonitor:
    """Tests for NaNMonitor callback."""

    def test_nan_monitor_creation(self):
        """Test NaNMonitor initialization."""
        callback = NaNMonitor(check_interval=10, raise_on_nan=True)
        assert callback.check_interval == 10
        assert callback.raise_on_nan
        assert callback.batch_count == 0

    def test_nan_monitor_detects_nan_loss(self):
        """Test that NaN monitor detects NaN in loss."""
        callback = NaNMonitor(check_interval=1, raise_on_nan=True)

        mock_fabric = MagicMock()
        mock_fabric.is_global_zero = True
        mock_model = MagicMock()
        mock_optimizer = MagicMock()

        with pytest.raises(RuntimeError, match="NaN"):
            callback.on_train_batch_end(
                mock_fabric, mock_model, mock_optimizer, None, 0, float("nan")
            )

    def test_nan_monitor_detects_inf_loss(self):
        """Test that NaN monitor detects Inf in loss."""
        callback = NaNMonitor(check_interval=1, raise_on_nan=True)

        mock_fabric = MagicMock()
        mock_fabric.is_global_zero = True
        mock_model = MagicMock()
        mock_optimizer = MagicMock()

        with pytest.raises(RuntimeError, match="NaN"):
            callback.on_train_batch_end(
                mock_fabric, mock_model, mock_optimizer, None, 0, float("inf")
            )

    def test_nan_monitor_skips_normal_loss(self):
        """Test that NaN monitor skips normal loss values."""
        callback = NaNMonitor(check_interval=1, raise_on_nan=True)

        mock_fabric = MagicMock()
        mock_fabric.is_global_zero = True
        mock_model = MagicMock()
        mock_optimizer = MagicMock()

        # Should not raise
        callback.on_train_batch_end(mock_fabric, mock_model, mock_optimizer, None, 0, 0.5)
        assert callback.batch_count == 1


class TestMetricsLogger:
    """Tests for MetricsLogger callback."""

    def test_metrics_logger_creation(self, tmp_path):
        """Test MetricsLogger initialization."""
        callback = MetricsLogger(log_dir=str(tmp_path), log_interval=1)
        assert callback.log_dir.exists()
        assert callback.log_interval == 1
        assert callback.history == {}

    def test_metrics_logger_logs_metrics(self, tmp_path):
        """Test that metrics are logged."""
        callback = MetricsLogger(log_dir=str(tmp_path), log_interval=1)

        mock_fabric = MagicMock()
        mock_fabric._current_metrics = {"train_loss": 0.5, "val_loss": 0.3}
        mock_model = MagicMock()
        mock_optimizer = MagicMock()

        callback.on_train_epoch_end(mock_fabric, mock_model, mock_optimizer)

        assert "train_loss" in callback.history
        assert "val_loss" in callback.history
        assert len(callback.history["train_loss"]) == 1

    def test_metrics_logger_exports_files(self, tmp_path):
        """Test that metrics are exported to files."""
        callback = MetricsLogger(log_dir=str(tmp_path), log_interval=1)

        mock_fabric = MagicMock()
        mock_fabric._current_metrics = {"train_loss": 0.5}
        mock_model = MagicMock()
        mock_optimizer = MagicMock()

        callback.on_train_epoch_end(mock_fabric, mock_model, mock_optimizer)
        callback.on_fit_end(mock_fabric)

        json_path = tmp_path / "metrics.json"
        csv_path = tmp_path / "metrics.csv"

        assert json_path.exists()
        assert csv_path.exists()

    def test_metrics_logger_respects_interval(self, tmp_path):
        """Test that logging respects interval."""
        callback = MetricsLogger(log_dir=str(tmp_path), log_interval=2)

        mock_fabric = MagicMock()
        mock_fabric._current_metrics = {"train_loss": 0.5}
        mock_model = MagicMock()
        mock_optimizer = MagicMock()

        # First epoch (should not log)
        callback.on_train_epoch_end(mock_fabric, mock_model, mock_optimizer)
        assert len(callback.history) == 0

        # Second epoch (should log)
        callback.on_train_epoch_end(mock_fabric, mock_model, mock_optimizer)
        assert len(callback.history) > 0
