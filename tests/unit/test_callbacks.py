"""Tests for training callbacks."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from maskgit3d.infrastructure.training.callbacks import (
    AxialSliceVisualizationCallback, EarlyStopping, MetricsLogger,
    ModelCheckpoint, NaNMonitor)


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

        mock_trainer = MagicMock()
        mock_trainer.callback_metrics = {"val_loss": torch.tensor(0.5)}
        mock_model = torch.nn.Linear(2, 2)
        mock_trainer.optimizers = [torch.optim.SGD(mock_model.parameters(), lr=0.01)]

        callback.on_validation_epoch_end(mock_trainer, mock_model)

        assert callback._current_epoch == 1
        # Check that last checkpoint file was created
        last_file = tmp_path / "last.ckpt"
        assert last_file.exists()

    def test_checkpoint_saves_best(self, tmp_path):
        """Test that best checkpoints are tracked."""
        callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last=False,
            dirpath=str(tmp_path),
        )

        mock_trainer = MagicMock()
        mock_model = torch.nn.Linear(2, 2)
        mock_trainer.optimizers = [torch.optim.SGD(mock_model.parameters(), lr=0.01)]

        # First validation epoch
        mock_trainer.callback_metrics = {"val_loss": torch.tensor(0.5)}
        callback.on_validation_epoch_end(mock_trainer, mock_model)

        # Second validation epoch (better)
        mock_trainer.callback_metrics = {"val_loss": torch.tensor(0.3)}
        callback.on_validation_epoch_end(mock_trainer, mock_model)

        assert len(callback._best_scores) > 0


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

        mock_trainer = MagicMock()
        mock_model = MagicMock()

        # Initial validation
        mock_trainer.callback_metrics = {"val_loss": torch.tensor(1.0)}
        callback.on_validation_epoch_end(mock_trainer, mock_model)
        assert not callback.should_stop

        # Worse validation (no improvement)
        mock_trainer.callback_metrics = {"val_loss": torch.tensor(1.1)}
        callback.on_validation_epoch_end(mock_trainer, mock_model)
        assert not callback.should_stop

        # Another worse validation
        callback.on_validation_epoch_end(mock_trainer, mock_model)
        assert callback.should_stop

    def test_early_stopping_reset(self):
        """Test that early stopping resets properly."""
        callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)

        mock_trainer = MagicMock()
        mock_model = torch.nn.Linear(2, 2)
        callback.on_fit_start(mock_trainer, mock_model)

        # After on_fit_start, best_score is initialized to inf for min mode
        assert callback._early_stopping.best_score == float("inf")
        assert callback._early_stopping.wait_count == 0
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

        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_batch = MagicMock()

        # Pass NaN in a dict as outputs
        nan_output = {"loss": torch.tensor(float("nan"))}

        with pytest.raises(RuntimeError, match="NaN"):
            callback.on_train_batch_end(mock_trainer, mock_model, nan_output, mock_batch, 0)

    def test_nan_monitor_detects_inf_loss(self):
        """Test that NaN monitor detects Inf in loss."""
        callback = NaNMonitor(check_interval=1, raise_on_nan=True)

        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_batch = MagicMock()

        # Pass Inf in a dict as outputs
        inf_output = {"loss": torch.tensor(float("inf"))}

        with pytest.raises(RuntimeError, match="NaN"):
            callback.on_train_batch_end(mock_trainer, mock_model, inf_output, mock_batch, 0)

    def test_nan_monitor_skips_normal_loss(self):
        """Test that NaN monitor skips normal loss values."""
        callback = NaNMonitor(check_interval=1, raise_on_nan=True)

        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_batch = MagicMock()

        # Pass normal loss in a dict as outputs
        normal_output = {"loss": torch.tensor(0.5)}

        # Should not raise
        callback.on_train_batch_end(mock_trainer, mock_model, normal_output, mock_batch, 0)
        assert callback._batch_count == 1


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

        mock_trainer = MagicMock()
        mock_trainer.callback_metrics = {"train_loss": torch.tensor(0.5)}
        mock_model = MagicMock()

        callback.on_train_epoch_end(mock_trainer, mock_model)

        assert "train_loss" in callback.history
        assert len(callback.history["train_loss"]) == 1

    def test_metrics_logger_exports_files(self, tmp_path):
        """Test that metrics are exported to files."""
        callback = MetricsLogger(log_dir=str(tmp_path), log_interval=1)

        mock_trainer = MagicMock()
        mock_trainer.callback_metrics = {"train_loss": torch.tensor(0.5)}
        mock_model = MagicMock()

        callback.on_train_epoch_end(mock_trainer, mock_model)
        callback.on_fit_end(mock_trainer, mock_model)

        json_path = tmp_path / "metrics.json"
        csv_path = tmp_path / "metrics.csv"

        assert json_path.exists()
        assert csv_path.exists()

    def test_metrics_logger_respects_interval(self, tmp_path):
        """Test that logging respects interval."""
        callback = MetricsLogger(log_dir=str(tmp_path), log_interval=2)

        mock_trainer = MagicMock()
        mock_trainer.callback_metrics = {"train_loss": torch.tensor(0.5)}
        mock_model = MagicMock()

        # First epoch (should not log)
        callback.on_train_epoch_end(mock_trainer, mock_model)
        assert len(callback.history) == 0

        # Second epoch (should log)
        callback.on_train_epoch_end(mock_trainer, mock_model)
        assert len(callback.history) > 0


class TestAxialSliceVisualizationCallback:
    """Tests for AxialSliceVisualizationCallback."""

    def test_callback_creation(self, tmp_path):
        """Test AxialSliceVisualizationCallback initialization."""
        callback = AxialSliceVisualizationCallback(
            num_samples=4,
            slice_range=3,
            output_dir=str(tmp_path),
            enable_tensorboard=True,
        )
        assert callback.num_samples == 4
        assert callback.slice_range == 3
        assert callback.output_dir.exists()
        assert callback.enable_tensorboard is True

    def test_extract_random_slice_normalizes_to_01(self):
        """Test that slice extraction normalizes values to [0,1]."""
        callback = AxialSliceVisualizationCallback(
            num_samples=4,
            slice_range=3,
            output_dir="/tmp/test",
            enable_tensorboard=False,
        )
        # Create a 3D volume with values in [-1, 1] range
        volume = torch.randn(1, 1, 28, 28, 28)  # [B, C, D, H, W]
        volume = torch.clamp(volume, -1, 1)  # Ensure range is [-1, 1]

        slice_img = callback._extract_random_slice(volume)

        # Check that values are normalized to [0, 1]
        assert slice_img.min() >= 0.0
        assert slice_img.max() <= 1.0

    def test_extract_random_slice_shape(self):
        """Test that extracted slice has correct shape."""
        callback = AxialSliceVisualizationCallback(
            num_samples=4,
            slice_range=3,
            output_dir="/tmp/test",
            enable_tensorboard=False,
        )
        # Create a 3D volume [B, C, D, H, W]
        volume = torch.randn(2, 3, 28, 32, 32)

        slice_img = callback._extract_random_slice(volume)

        # Should return H x W image
        assert slice_img.shape == (32, 32)

    def test_save_to_disk_creates_file(self, tmp_path):
        """Test that saving to disk creates a PNG file."""
        callback = AxialSliceVisualizationCallback(
            num_samples=4,
            slice_range=3,
            output_dir=str(tmp_path),
            enable_tensorboard=False,
        )
        # Create a test image [0, 1] range
        image = torch.rand(28, 28)
        filepath = tmp_path / "test_slice.png"

        callback._save_to_disk(image, str(filepath))

        assert filepath.exists()
        assert filepath.suffix == ".png"

    @patch("maskgit3d.infrastructure.training.callbacks.SummaryWriter")
    def test_log_to_tensorboard_calls_writer(self, mock_summary_writer):
        """Test that logging to tensorboard calls SummaryWriter."""
        callback = AxialSliceVisualizationCallback(
            num_samples=4,
            slice_range=3,
            output_dir="/tmp/test",
            enable_tensorboard=True,
        )
        callback._tensorboard_writer = MagicMock()
        mock_writer = callback._tensorboard_writer

        # Create a test image
        image = torch.rand(28, 28)

        callback._log_to_tensorboard(image, "test/tag", step=1)

        mock_writer.add_image.assert_called_once()

    def test_process_batch_handles_volume_input(self, tmp_path):
        """Test that _process_batch handles 3D volume input."""
        callback = AxialSliceVisualizationCallback(
            num_samples=2,
            slice_range=3,
            output_dir=str(tmp_path),
            enable_tensorboard=False,
        )

        # Create mock batch: dict with 'volumes' key containing 3D tensor [B, C, D, H, W]
        batch = {"volumes": torch.randn(4, 1, 28, 28, 28)}
        batch_idx = 0

        # Should not raise any errors
        callback._process_batch(batch, batch_idx, "val")

    def test_on_validation_batch_end_calls_process(self, tmp_path):
        """Test that on_validation_batch_end triggers visualization."""
        callback = AxialSliceVisualizationCallback(
            num_samples=2,
            slice_range=3,
            output_dir=str(tmp_path),
            enable_tensorboard=False,
        )

        mock_trainer = MagicMock()
        mock_model = MagicMock()
        batch = {"volumes": torch.randn(4, 1, 28, 28, 28)}
        batch_idx = 0
        outputs = None

        # Should not raise
        callback.on_validation_batch_end(mock_trainer, mock_model, outputs, batch, batch_idx)

    def test_on_test_batch_end_calls_process(self, tmp_path):
        """Test that on_test_batch_end triggers visualization."""
        callback = AxialSliceVisualizationCallback(
            num_samples=2,
            slice_range=3,
            output_dir=str(tmp_path),
            enable_tensorboard=False,
        )

        mock_trainer = MagicMock()
        mock_model = MagicMock()
        batch = {"volumes": torch.randn(4, 1, 28, 28, 28)}
        batch_idx = 0
        outputs = None

        # Should not raise
        callback.on_test_batch_end(mock_trainer, mock_model, outputs, batch, batch_idx)
