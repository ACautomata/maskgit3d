"""VQVAE training loss callback for logging loss components."""

from __future__ import annotations

from typing import Any

from lightning.pytorch import Callback, LightningModule, Trainer


class VQVAETrainingLossCallback(Callback):
    """Callback for logging VQVAE training loss components.

    Logs individual loss components from the training_step output dict.
    The output dict contains: loss, loss_g, loss_d, nll_loss, rec_loss,
    p_loss, g_loss, vq_loss, disc_loss.

    Args:
        log_every_n_steps: Log every N steps (default: 1).

    Example:
        >>> callback = VQVAETrainingLossCallback(log_every_n_steps=10)
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(self, log_every_n_steps: int = 1) -> None:
        super().__init__()
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
        """Log training loss components from outputs dict."""
        self._step_count += 1

        if self.log_every_n_steps > 1 and self._step_count % self.log_every_n_steps != 0:
            return

        if outputs is None or not isinstance(outputs, dict):
            return

        # Log each loss component as train/{key}
        for key in (
            "loss",
            "loss_g",
            "loss_d",
            "nll_loss",
            "rec_loss",
            "p_loss",
            "g_loss",
            "vq_loss",
            "disc_loss",
        ):
            value = outputs.get(key)
            if value is not None:
                pl_module.log(f"train/{key}", value, prog_bar=(key == "loss"), logger=True)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset step counter at epoch start."""
        self._step_count = 0
