"""Reconstruction loss callback for validation and test phases."""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer


class ReconstructionLossCallback(Callback):
    """Callback for computing L1 reconstruction loss during validation/test.

    Extracts x_real and x_recon from outputs dict and computes mean L1 loss.

    Example:
        >>> callback = ReconstructionLossCallback()
        >>> trainer = Trainer(callbacks=[callback])
    """

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute and log L1 reconstruction loss on validation batch."""
        if outputs is None or not isinstance(outputs, dict):
            return

        x_real = outputs.get("x_real")
        x_recon = outputs.get("x_recon")

        if x_real is not None and x_recon is not None:
            rec_loss = F.l1_loss(x_recon, x_real)
            pl_module.log("val_rec_loss", rec_loss, prog_bar=True)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute and log L1 reconstruction loss on test batch."""
        if outputs is None or not isinstance(outputs, dict):
            return

        x_real = outputs.get("x_real")
        x_recon = outputs.get("x_recon")

        if x_real is not None and x_recon is not None:
            rec_loss = F.l1_loss(x_recon, x_real)
            pl_module.log("loss_l1:test", rec_loss, prog_bar=False)
