"""Callback for computing and logging masked cross-entropy loss for MaskGIT."""

from typing import Any

import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer


class MaskedCrossEntropyCallback(Callback):
    """Computes and logs masked cross-entropy loss for MaskGIT.

    Logs loss for train/val/test splits by accessing outputs directly
    rather than through callback_payload.

    Args:
        log_train_every_n_steps: Log training loss every N steps (default: 1).
    """

    def __init__(
        self,
        log_train_every_n_steps: int = 1,
    ) -> None:
        super().__init__()
        self.log_train_every_n_steps = log_train_every_n_steps
        self._train_step_count = 0

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log training loss from outputs dict."""
        self._train_step_count += 1

        if (
            self.log_train_every_n_steps > 1
            and self._train_step_count % self.log_train_every_n_steps != 0
        ):
            return

        if outputs is None:
            return

        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
            if isinstance(loss, torch.Tensor):
                pl_module.log("train/loss", loss, prog_bar=True)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log validation masked cross-entropy loss from outputs dict."""
        if outputs is None:
            return

        masked_logits = outputs.get("masked_logits")
        masked_targets = outputs.get("masked_targets")

        if masked_logits is not None and masked_targets is not None:
            if isinstance(masked_logits, torch.Tensor) and isinstance(masked_targets, torch.Tensor):
                loss = F.cross_entropy(masked_logits, masked_targets)
                pl_module.log("val_loss", loss, prog_bar=True)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log test masked cross-entropy loss from outputs dict."""
        if outputs is None:
            return

        masked_logits = outputs.get("masked_logits")
        masked_targets = outputs.get("masked_targets")

        if masked_logits is not None and masked_targets is not None:
            if isinstance(masked_logits, torch.Tensor) and isinstance(masked_targets, torch.Tensor):
                loss = F.cross_entropy(masked_logits, masked_targets)
                pl_module.log("loss:test", loss, prog_bar=True)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset step counter at epoch start."""
        self._train_step_count = 0
