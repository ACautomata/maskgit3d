"""Tests for MaskGIT metrics callback."""

from unittest.mock import MagicMock, Mock

import torch
from lightning.pytorch import LightningModule

from maskgit3d.callbacks.maskgit_metrics import MaskGITMetricsCallback


class SimpleModel(LightningModule):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        return {}


class TestMaskGITMetricsCallback:
    """Test suite for MaskGITMetricsCallback."""

    def test_callback_initialization_default(self):
        """Test callback can be initialized with default parameters."""
        callback = MaskGITMetricsCallback()
        assert callback.log_every_n_steps == 1
        assert callback.log_val_every_n_batches == 1
        assert callback._train_step_count == 0

    def test_callback_initialization_custom(self):
        """Test callback can be initialized with custom parameters."""
        callback = MaskGITMetricsCallback(log_every_n_steps=10, log_val_every_n_batches=5)
        assert callback.log_every_n_steps == 10
        assert callback.log_val_every_n_batches == 5

    def test_on_train_batch_end_with_none_outputs(self):
        """Test on_train_batch_end with None outputs."""
        callback = MaskGITMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()

        # Should not raise
        callback.on_train_batch_end(trainer, model, None, None, 0)
        assert callback._train_step_count == 1

    def test_on_train_batch_end_with_tensor_outputs(self):
        """Test on_train_batch_end with tensor outputs (just loss)."""
        callback = MaskGITMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()

        # Create a mock log method that records calls
        logged_values = []

        def mock_log(name, value, **kwargs):
            logged_values.append((name, value))

        model.log = mock_log

        loss = torch.tensor(1.5)
        callback.on_train_batch_end(trainer, model, loss, None, 0)

        # Should log loss:train
        assert ("loss:train", loss) in logged_values or any(
            "loss:train" in str(v[0]) for v in logged_values
        )

    def test_on_train_batch_end_with_dict_loss_only(self):
        """Test on_train_batch_end with dict containing loss only."""
        callback = MaskGITMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()

        logged_values = []

        def mock_log(name, value, **kwargs):
            logged_values.append((name, value))

        model.log = mock_log

        outputs = {"loss": torch.tensor(2.0)}
        callback.on_train_batch_end(trainer, model, outputs, None, 0)

        # Should log loss:train
        assert any("loss:train" in str(v[0]) for v in logged_values)

    def test_on_train_batch_end_with_log_data(self):
        """Test on_train_batch_end with log_data containing scalars."""
        callback = MaskGITMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()

        logged_values = []

        def mock_log(name, value, **kwargs):
            logged_values.append((name, value))

        model.log = mock_log

        # Use new scalar format
        log_data = {
            "correct": 7,
            "total": 10,
            "mask_ratio": 0.5,
        }
        outputs = {
            "loss": torch.tensor(1.0),
            "log_data": log_data,
        }

        callback.on_train_batch_end(trainer, model, outputs, None, 0)

        # Should log loss:train and mask_acc:train and mask_ratio
        logged_names = [str(v[0]) for v in logged_values]
        assert any("loss:train" in name for name in logged_names)
        assert any("mask_acc:train" in name for name in logged_names)
        assert any("mask_ratio:train" in name for name in logged_names)

    def test_on_train_batch_end_with_log_data_float_mask_ratio(self):
        """Test on_train_batch_end with float mask_ratio."""
        callback = MaskGITMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()

        logged_values = []

        def mock_log(name, value, **kwargs):
            logged_values.append((name, value))

        model.log = mock_log

        # Use new scalar format
        log_data = {
            "correct": 5,
            "total": 10,
            "mask_ratio": 0.75,
        }
        outputs = {
            "loss": torch.tensor(1.0),
            "log_data": log_data,
        }

        callback.on_train_batch_end(trainer, model, outputs, None, 0)

        # Check mask_ratio was logged (as tensor)
        assert any("mask_ratio:train" in str(v[0]) for v in logged_values)

    def test_on_train_batch_end_log_every_n_steps(self):
        """Test on_train_batch_end respects log_every_n_steps."""
        callback = MaskGITMetricsCallback(log_every_n_steps=2)
        trainer = MagicMock()
        model = SimpleModel()

        logged_values = []

        def mock_log(name, value, **kwargs):
            logged_values.append((name, value))

        model.log = mock_log

        outputs = {"loss": torch.tensor(1.0)}

        # First call - should not log (step 1, not divisible by 2)
        callback.on_train_batch_end(trainer, model, outputs, None, 0)
        assert callback._train_step_count == 1

        # Reset logged values
        logged_values.clear()

        # Second call - should log (step 2, divisible by 2)
        callback.on_train_batch_end(trainer, model, outputs, None, 1)
        assert callback._train_step_count == 2
        assert len(logged_values) > 0

    def test_on_validation_batch_end_with_none_outputs(self):
        """Test on_validation_batch_end with None outputs."""
        callback = MaskGITMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()

        # Should not raise
        callback.on_validation_batch_end(trainer, model, None, None, 0)

    def test_on_validation_batch_end_with_tensor_outputs(self):
        """Test on_validation_batch_end with tensor outputs."""
        callback = MaskGITMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()

        logged_values = []

        def mock_log(name, value, **kwargs):
            logged_values.append((name, value))

        model.log = mock_log

        loss = torch.tensor(1.5)
        callback.on_validation_batch_end(trainer, model, loss, None, 0)

        # Should log loss:val
        assert any("loss:val" in str(v[0]) for v in logged_values)

    def test_on_validation_batch_end_with_dict_loss_only(self):
        """Test on_validation_batch_end with dict containing loss only."""
        callback = MaskGITMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()

        # Add maskgit mock to avoid AttributeError
        model.maskgit = Mock()
        model.maskgit.generate = Mock(return_value=torch.randn(1, 4, 4, 4, 4))

        logged_values = []

        def mock_log(name, value, **kwargs):
            logged_values.append((name, value))

        model.log = mock_log

        outputs = {"loss": torch.tensor(2.0)}
        callback.on_validation_batch_end(trainer, model, outputs, None, 0)

        # Should log loss:val
        assert any("loss:val" in str(v[0]) for v in logged_values)

    def test_on_validation_batch_end_with_log_data(self):
        """Test on_validation_batch_end with log_data containing scalars."""
        callback = MaskGITMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()
        model.maskgit = Mock()
        model.maskgit.generate = Mock(return_value=torch.randn(1, 4, 4, 4, 4))

        logged_values = []

        def mock_log(name, value, **kwargs):
            logged_values.append((name, value))

        model.log = mock_log

        # Use new scalar format
        log_data = {
            "correct": 7,
            "total": 10,
        }
        outputs = {
            "loss": torch.tensor(1.0),
            "log_data": log_data,
        }

        callback.on_validation_batch_end(trainer, model, outputs, None, 0)

        # Should log loss:val and mask_acc:val
        logged_names = [str(v[0]) for v in logged_values]
        assert any("loss:val" in name for name in logged_names)
        assert any("mask_acc:val" in name for name in logged_names)

    def test_on_train_epoch_start_resets_counter(self):
        """Test that on_train_epoch_start resets step counter."""
        callback = MaskGITMetricsCallback()
        callback._train_step_count = 50

        model = SimpleModel()
        callback.on_train_epoch_start(None, model)

        assert callback._train_step_count == 0
