"""Tests for gradient norm callback."""

import torch
from lightning.pytorch import LightningModule

from maskgit3d.callbacks.gradient_norm import GradientNormCallback


class SimpleModel(LightningModule):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 5)
        self.layer2 = torch.nn.Linear(5, 1)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class TestGradientNormCallback:
    """Test suite for GradientNormCallback."""

    def test_callback_initialization(self):
        """Test callback can be initialized with different parameters."""
        callback = GradientNormCallback(mode="total")
        assert callback.mode == "total"
        assert callback.log_every_n_steps == 1
        assert callback.norm_type == 2.0

        callback = GradientNormCallback(mode="per_layer", log_every_n_steps=10, norm_type=1.0)
        assert callback.mode == "per_layer"
        assert callback.log_every_n_steps == 10
        assert callback.norm_type == 1.0

    def test_total_norm_computation(self):
        """Test total gradient norm computation."""
        callback = GradientNormCallback(mode="total")
        model = SimpleModel()

        # Create gradients
        x = torch.randn(2, 10)
        y = torch.randn(2, 1)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()

        norm = callback._compute_total_norm(model)
        assert norm.item() > 0
        assert torch.isfinite(norm)

    def test_step_counting(self):
        """Test step counting for logging frequency."""
        callback = GradientNormCallback(log_every_n_steps=2)
        assert callback._step_count == 0

        model = SimpleModel()
        callback.on_train_batch_end(None, model, None, None, 0)
        assert callback._step_count == 1

        callback.on_train_batch_end(None, model, None, None, 1)
        assert callback._step_count == 2

    def test_state_dict(self):
        """Test state dict saving and loading."""
        callback = GradientNormCallback()
        callback._step_count = 100

        state = callback.state_dict()
        assert state["step_count"] == 100

        new_callback = GradientNormCallback()
        new_callback.load_state_dict(state)
        assert new_callback._step_count == 100

    def test_epoch_reset(self):
        """Test that step counter resets at epoch start."""
        callback = GradientNormCallback()
        callback._step_count = 50

        callback.on_train_epoch_start(None, None)
        assert callback._step_count == 0
