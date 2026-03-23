"""Callback for computing and logging mask prediction accuracy for MaskGIT."""

from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer


class MaskAccuracyCallback(Callback):
    """Computes and logs mask prediction accuracy for MaskGIT.

    Logs accuracy for val/test splits by accessing outputs directly
    rather than through callback_payload.

    Args:
        log_val_every_n_batches: Log validation accuracy every N batches (default: 1).
    """

    def __init__(
        self,
        log_val_every_n_batches: int = 1,
    ) -> None:
        super().__init__()
        self.log_val_every_n_batches = log_val_every_n_batches

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log validation mask accuracy from outputs dict."""
        if outputs is None:
            return

        if self.log_val_every_n_batches > 1 and batch_idx % self.log_val_every_n_batches != 0:
            return

        masked_logits = outputs.get("masked_logits")
        masked_targets = outputs.get("masked_targets")

        if (
            masked_logits is not None
            and masked_targets is not None
            and isinstance(masked_logits, torch.Tensor)
            and isinstance(masked_targets, torch.Tensor)
        ):
            predictions = masked_logits.argmax(dim=-1)
            accuracy = (predictions == masked_targets).float().mean()
            pl_module.log("val_mask_acc", accuracy, prog_bar=True)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log test mask accuracy from outputs dict."""
        if outputs is None:
            return

        masked_logits = outputs.get("masked_logits")
        masked_targets = outputs.get("masked_targets")

        if (
            masked_logits is not None
            and masked_targets is not None
            and isinstance(masked_logits, torch.Tensor)
            and isinstance(masked_targets, torch.Tensor)
        ):
            predictions = masked_logits.argmax(dim=-1)
            accuracy = (predictions == masked_targets).float().mean()
            pl_module.log("mask_acc:test", accuracy, prog_bar=True)
