"""Tests for VQVAE metrics callback."""

from collections.abc import Callable
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer

from maskgit3d.callbacks.vqvae_metrics import VQVAEMetricsCallback


class SimpleModel(LightningModule):
    """Simple model for testing."""

    def __init__(self, use_perceptual: bool = False):
        super().__init__()
        self.use_perceptual_flag = use_perceptual
        self.logged_values: list[tuple[str, torch.Tensor]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def training_step(self, batch: object, batch_idx: int) -> dict[str, torch.Tensor]:
        return {}

    def log(self, *args: object, **kwargs: object) -> None:
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], torch.Tensor):
            self.logged_values.append((args[0], args[1]))


class MockLossFn(nn.Module):
    """Mock loss function for testing."""

    def __init__(self, use_perceptual: bool = False):
        self.use_perceptual = use_perceptual
        self.perceptual_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        super().__init__()
        self.perceptual_loss = self._perceptual_loss if use_perceptual else None

    def _perceptual_loss(self, x_recon: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
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
        if optimizer_idx == 0:
            loss = torch.tensor(1.0)
            log = {"loss_g": loss, "loss_recon": torch.tensor(0.5)}
        else:
            loss = torch.tensor(0.5)
            log = {"loss_d": loss}
        return loss, log


class TestVQVAEMetricsCallback:
    """Test suite for VQVAEMetricsCallback."""

    def test_callback_initialization_default(self):
        """Test callback can be initialized with default parameters."""
        callback = VQVAEMetricsCallback()
        assert callback.log_every_n_steps == 1
        assert callback.log_val_every_n_batches == 1
        assert callback._train_step_count == 0

    def test_callback_initialization_custom(self):
        """Test callback can be initialized with custom parameters."""
        callback = VQVAEMetricsCallback(log_every_n_steps=10, log_val_every_n_batches=5)
        assert callback.log_every_n_steps == 10
        assert callback.log_val_every_n_batches == 5

    def test_on_train_batch_end_with_none_outputs(self):
        """Test on_train_batch_end with None outputs."""
        callback = VQVAEMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()
        model.loss_fn = MockLossFn()

        # Should not raise and should not log
        callback.on_train_batch_end(trainer, model, None, None, 0)
        assert callback._train_step_count == 1

    def test_on_train_batch_end_with_tensor_outputs(self):
        """Test on_train_batch_end with tensor outputs (should skip)."""
        callback = VQVAEMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()
        model.loss_fn = MockLossFn()

        # Tensor outputs should be skipped (not a dict)
        callback.on_train_batch_end(trainer, model, torch.tensor(1.0), None, 0)
        assert callback._train_step_count == 1

    def test_on_train_batch_end_with_dict_outputs(self):
        """Test on_train_batch_end extracts and logs from dict outputs."""
        callback = VQVAEMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()
        model.loss_fn = MockLossFn()

        # Create mock outputs with required keys
        outputs = {
            "x_real": torch.randn(1, 1, 8, 8, 8),
            "x_recon": torch.randn(1, 1, 8, 8, 8),
            "vq_loss": torch.tensor(0.1),
            "last_layer": None,
        }

        callback.on_train_batch_end(trainer, model, outputs, None, 0)

        logged_names = [name for name, _ in model.logged_values]
        assert "loss_g:train" in logged_names
        assert "loss_recon:train" in logged_names
        assert "loss_d:train" in logged_names

    def test_on_train_batch_end_with_missing_keys(self):
        """Test on_train_batch_end with dict but missing required keys."""
        callback = VQVAEMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()
        model.loss_fn = MockLossFn()

        # Dict without required keys should be skipped - no exception
        outputs = {"some_key": "some_value"}
        callback.on_train_batch_end(trainer, model, outputs, None, 0)
        assert model.logged_values == []

    def test_on_train_batch_end_log_every_n_steps(self):
        """Test on_train_batch_end respects log_every_n_steps."""
        callback = VQVAEMetricsCallback(log_every_n_steps=2)
        trainer = MagicMock()
        model = SimpleModel()
        model.loss_fn = MockLossFn()

        outputs = {
            "x_real": torch.randn(1, 1, 8, 8, 8),
            "x_recon": torch.randn(1, 1, 8, 8, 8),
            "vq_loss": torch.tensor(0.1),
            "last_layer": None,
        }

        # First call - should not log (step 1, not divisible by 2)
        callback.on_train_batch_end(trainer, model, outputs, None, 0)
        assert callback._train_step_count == 1
        assert model.logged_values == []

        # Second call - should log (step 2, divisible by 2)
        callback.on_train_batch_end(trainer, model, outputs, None, 1)
        assert callback._train_step_count == 2
        assert len(model.logged_values) == 3

    def test_on_validation_batch_end_with_none_outputs(self):
        """Test on_validation_batch_end with None outputs."""
        callback = VQVAEMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()

        # Should not raise
        callback.on_validation_batch_end(trainer, model, None, None, 0)

    def test_on_validation_batch_end_with_tensor_outputs(self):
        """Test on_validation_batch_end with tensor outputs (should skip)."""
        callback = VQVAEMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()

        # Tensor outputs should be skipped
        callback.on_validation_batch_end(trainer, model, torch.tensor(1.0), None, 0)

    def test_on_validation_batch_end_with_dict_outputs(self):
        """Test on_validation_batch_end computes and logs L1 loss."""
        callback = VQVAEMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()

        model.loss_fn = MockLossFn()

        # Create mock outputs with required keys
        x_real = torch.randn(1, 1, 8, 8, 8)
        x_recon = x_real + torch.randn_like(x_real) * 0.1  # Slightly different
        outputs = {
            "x_real": x_real,
            "x_recon": x_recon,
            "vq_loss": torch.tensor(0.1),
        }

        callback.on_validation_batch_end(trainer, model, outputs, None, 0)

        logged_names = [name for name, _ in model.logged_values]
        assert "val_loss:val" in logged_names
        assert "loss_l1:val" in logged_names
        assert "loss_vq:val" in logged_names

    def test_on_validation_batch_end_with_missing_keys(self):
        """Test on_validation_batch_end with dict but missing keys."""
        callback = VQVAEMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel()

        # Dict without required keys should be skipped - no exception
        outputs = {"some_key": "some_value"}
        callback.on_validation_batch_end(trainer, model, outputs, None, 0)
        assert model.logged_values == []

    def test_on_validation_batch_end_with_perceptual_loss(self):
        """Test on_validation_batch_end computes perceptual loss when enabled."""
        callback = VQVAEMetricsCallback()
        trainer = MagicMock()
        model = SimpleModel(use_perceptual=True)
        model.loss_fn = MockLossFn(use_perceptual=True)

        x_real = torch.randn(1, 1, 8, 8, 8)
        x_recon = x_real + torch.randn_like(x_real) * 0.1
        outputs = {
            "x_real": x_real,
            "x_recon": x_recon,
            "vq_loss": torch.tensor(0.1),
        }

        callback.on_validation_batch_end(trainer, model, outputs, None, 0)

        logged_names = [name for name, _ in model.logged_values]
        assert "loss_perceptual:val" in logged_names

    def test_on_train_epoch_start_resets_counter(self):
        """Test that on_train_epoch_start resets step counter."""
        callback = VQVAEMetricsCallback()
        callback._train_step_count = 50

        model = SimpleModel()
        trainer = MagicMock(spec=Trainer)
        callback.on_train_epoch_start(trainer, model)

        assert callback._train_step_count == 0
