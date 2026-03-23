"""VQVAE training loss callback for logging loss components."""

from __future__ import annotations

from typing import Any

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
