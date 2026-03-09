"""MaskGIT metrics callback for logging training/validation/test metrics.

This callback handles all metric logging for MaskGIT training, separating
metrics computation from the Task class.
"""

from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

if TYPE_CHECKING:
    pass


class MaskGITMetricsCallback(Callback):
    """Callback for logging MaskGIT-specific metrics.

    This callback receives log data from MaskGITTask's step methods and logs
    them in the format metric_name:split.

    Args:
        log_every_n_steps: Log training metrics every N steps (0 = every step).
        log_val_every_n_batches: Log validation metrics every N batches.

    Example:
        >>> callback = MaskGITMetricsCallback()
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        log_every_n_steps: int = 1,
        log_val_every_n_batches: int = 1,
    ) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_val_every_n_batches = log_val_every_n_batches
        self._train_step_count = 0

    def _compute_mask_accuracy(
        self, masked_logits: torch.Tensor, masked_targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute mask prediction accuracy from raw logits and targets.

        Args:
            masked_logits: Logits for masked positions [N, vocab_size]
            masked_targets: Target token indices for masked positions [N]

        Returns:
            Accuracy as a scalar tensor
        """
        with torch.no_grad():
            preds = masked_logits.argmax(dim=-1)
            acc = (preds == masked_targets).float().mean()
        return acc

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log training metrics from outputs."""
        self._train_step_count += 1

        if self.log_every_n_steps > 1 and self._train_step_count % self.log_every_n_steps != 0:
            return

        if outputs is None:
            return

        # Handle tensor outputs (just loss)
        if isinstance(outputs, torch.Tensor):
            pl_module.log("loss:train", outputs, prog_bar=True)
            return

        # Log main loss
        if "loss" in outputs:
            pl_module.log("loss:train", outputs["loss"], prog_bar=True)

        # Compute and log metrics from raw data
        if "log_data" in outputs and isinstance(outputs["log_data"], dict):
            log_data = outputs["log_data"]

            # Compute mask accuracy from raw logits and targets
            if "masked_logits" in log_data and "masked_targets" in log_data:
                acc = self._compute_mask_accuracy(
                    log_data["masked_logits"], log_data["masked_targets"]
                )
                pl_module.log("mask_acc:train", acc, prog_bar=True)

            # Log mask ratio
            if "mask_ratio" in log_data:
                mask_ratio = log_data["mask_ratio"]
                if isinstance(mask_ratio, torch.Tensor):
                    pl_module.log("mask_ratio:train", mask_ratio, prog_bar=False)
                elif isinstance(mask_ratio, (int, float)):
                    pl_module.log("mask_ratio:train", torch.tensor(mask_ratio), prog_bar=False)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset step counter at epoch start."""
        self._train_step_count = 0

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log validation metrics from outputs."""
        if outputs is None:
            return

        # Handle tensor outputs
        if isinstance(outputs, torch.Tensor):
            pl_module.log("loss:val", outputs, prog_bar=True)
            return

        # Log main loss
        if "loss" in outputs:
            pl_module.log("loss:val", outputs["loss"], prog_bar=True)

        # Compute and log metrics from raw data
        if "log_data" in outputs and isinstance(outputs["log_data"], dict):
            log_data = outputs["log_data"]

            # Compute mask accuracy from raw logits and targets
            if "masked_logits" in log_data and "masked_targets" in log_data:
                acc = self._compute_mask_accuracy(
                    log_data["masked_logits"], log_data["masked_targets"]
                )
                pl_module.log("mask_acc:val", acc, prog_bar=True)

        # Log sample shape if available
        if batch_idx == 0:
            with torch.no_grad():
                sample = pl_module.maskgit.generate(shape=(1, 4, 4, 4), num_iterations=12)  # type: ignore[attr-defined,union-attr,operator]
                pl_module.log("sample_shape:val", torch.tensor(sample.shape[0]), prog_bar=False)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log test metrics from outputs."""
        if outputs is None:
            return

        # Handle tensor outputs
        if isinstance(outputs, torch.Tensor):
            pl_module.log("loss:test", outputs, prog_bar=True)
            return

        # Log main loss
        if "loss" in outputs:
            pl_module.log("loss:test", outputs["loss"], prog_bar=True)

        # Log data from log_data dict
        if "log_data" in outputs and isinstance(outputs["log_data"], dict):
            log_data = outputs["log_data"]
            if "mask_acc" in log_data:
                value = log_data["mask_acc"]
                if isinstance(value, torch.Tensor):
                    pl_module.log("mask_acc:test", value, prog_bar=True)
                elif isinstance(value, (int, float)):
                    pl_module.log("mask_acc:test", torch.tensor(value), prog_bar=True)

        maskgit_model = getattr(pl_module, "maskgit", None)
        if maskgit_model:
            sliding_window_cfg = getattr(maskgit_model, "sliding_window_cfg", None)
            if sliding_window_cfg and sliding_window_cfg.get("enabled", False):
                pl_module.log("sliding_window_enabled:test", torch.tensor(1.0), prog_bar=False)
