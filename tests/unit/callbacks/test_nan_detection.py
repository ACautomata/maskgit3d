"""Tests for NaN detection callback."""

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer

from maskgit3d.callbacks.nan_detection import NaNDetectionCallback


class SimpleModel(LightningModule):
    """Simple model for testing."""

    def __init__(self, return_nan=False):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)
        self.return_nan = return_nan

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        if self.return_nan:
            loss = torch.tensor(float("nan"))
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class TestNaNDetectionCallback:
    """Test suite for NaNDetectionCallback."""

    def test_callback_initialization(self):
        """Test callback can be initialized with different parameters."""
        callback = NaNDetectionCallback()
        assert callback.action == "abort"
        assert callback.check_loss is True
        assert callback.check_gradients is True

        callback = NaNDetectionCallback(check_gradients=False)
        assert callback.check_gradients is False

    def test_nan_detection_abort_action(self):
        """Test that abort action raises RuntimeError when NaN is detected."""
        callback = NaNDetectionCallback()
        callback._trainer = None

        with pytest.raises(RuntimeError, match="Training aborted"):
            callback._handle_nan("loss", 0, float("nan"))

    def test_nan_count_tracking(self):
        """Test that NaN count is tracked correctly before abort."""
        callback = NaNDetectionCallback()
        assert callback._nan_count == 0

        # Simulate NaN detection - should raise RuntimeError after counting
        with pytest.raises(RuntimeError, match="Training aborted"):
            callback._handle_nan("test", 0)
        assert callback._nan_count == 1

    def test_state_dict(self):
        """Test state dict saving and loading."""
        callback = NaNDetectionCallback()
        callback._nan_count = 10

        state = callback.state_dict()
        assert state["nan_count"] == 10
        assert state["action"] == "abort"

        new_callback = NaNDetectionCallback()
        new_callback.load_state_dict(state)
        assert new_callback._nan_count == 10

    def test_epoch_reset(self):
        """Test that epoch counter resets properly."""
        callback = NaNDetectionCallback()
        callback._nan_count_epoch = 5

        callback.on_train_epoch_start(None, None)
        assert callback._nan_count_epoch == 0
