"""VQVAE training loss callback for logging loss components."""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer


class TrainingLossCallback(Callback):
    """Callback for logging a single training loss metric.

    This is a single-metric callback that can be composed with other
    TrainingLossCallback instances to log multiple loss components.

    Args:
        metric_name: The key in outputs dict to log (e.g., "loss", "loss_g")
        log_name: The name to use for logging (e.g., "train/loss")
        prog_bar: Whether to show in progress bar

    Example:
        >>> callbacks = [
        ...     TrainingLossCallback("loss", "train/loss", prog_bar=True),
        ...     TrainingLossCallback("loss_g", "train/loss_g"),
        ...     TrainingLossCallback("rec_loss", "train/rec_loss"),
        ... ]
    """

    def __init__(
        self,
        metric_name: str,
        log_name: str | None = None,
        prog_bar: bool = False,
        log_every_n_steps: int = 1,
    ) -> None:
        super().__init__()
        self.metric_name = metric_name
        self.log_name = log_name or f"train/{metric_name}"
        self.prog_bar = prog_bar
        self.log_every_n_steps = log_every_n_steps
        self._step_count = 0

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log single training loss component from outputs dict."""
        self._step_count += 1

        if self.log_every_n_steps > 1 and self._step_count % self.log_every_n_steps != 0:
            return

        if outputs is None or not isinstance(outputs, dict):
            return

        value = outputs.get(self.metric_name)
        if value is not None:
            pl_module.log(self.log_name, value, prog_bar=self.prog_bar, logger=True)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset step counter at epoch start."""
        self._step_count = 0


class VQVAETrainingLossCallback(Callback):
    """Composite callback for logging all VQVAE training loss components.

    This is a convenience callback that combines multiple TrainingLossCallback
    instances for the standard VQVAE loss components.

    Args:
        log_every_n_steps: Log every N steps (default: 1).
    """

    def __init__(self, log_every_n_steps: int = 1) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._callbacks: list[TrainingLossCallback] = [
            TrainingLossCallback(
                "loss", "train/loss", prog_bar=True, log_every_n_steps=log_every_n_steps
            ),
            TrainingLossCallback("loss_g", "train/loss_g", log_every_n_steps=log_every_n_steps),
            TrainingLossCallback("loss_d", "train/loss_d", log_every_n_steps=log_every_n_steps),
            TrainingLossCallback("nll_loss", "train/nll_loss", log_every_n_steps=log_every_n_steps),
            TrainingLossCallback("rec_loss", "train/rec_loss", log_every_n_steps=log_every_n_steps),
            TrainingLossCallback("p_loss", "train/p_loss", log_every_n_steps=log_every_n_steps),
            TrainingLossCallback("g_loss", "train/g_loss", log_every_n_steps=log_every_n_steps),
            TrainingLossCallback("vq_loss", "train/vq_loss", log_every_n_steps=log_every_n_steps),
            TrainingLossCallback(
                "disc_loss", "train/disc_loss", log_every_n_steps=log_every_n_steps
            ),
        ]

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Delegate to individual metric callbacks."""
        for callback in self._callbacks:
            callback.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset step counter at epoch start."""
        for callback in self._callbacks:
            callback.on_train_epoch_start(trainer, pl_module)
