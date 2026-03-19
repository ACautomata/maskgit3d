from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
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

        loss: torch.Tensor | None = None
        if isinstance(outputs, torch.Tensor):
            loss = outputs
        elif isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]

        if loss is not None:
            pl_module.log("loss:train", loss, prog_bar=True)

        callback_payload = self._pop_callback_payload(pl_module, "train")
        if callback_payload is None:
            return

        masked_logits = callback_payload.get("masked_logits")
        masked_targets = callback_payload.get("masked_targets")
        mask_ratio = callback_payload.get("mask_ratio")

        if isinstance(masked_logits, torch.Tensor) and isinstance(masked_targets, torch.Tensor):
            predictions = masked_logits.argmax(dim=-1)
            accuracy = (predictions == masked_targets).float().mean()
            pl_module.log("mask_acc:train", accuracy, prog_bar=True)

        if isinstance(mask_ratio, int | float):
            pl_module.log("mask_ratio:train", torch.tensor(float(mask_ratio)), prog_bar=False)

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
        if not isinstance(outputs, dict):
            return

        masked_logits = outputs.get("masked_logits")
        masked_targets = outputs.get("masked_targets")
        mask_ratio = outputs.get("mask_ratio")
        generated_images = outputs.get("generated_images")

        if isinstance(masked_logits, torch.Tensor) and isinstance(masked_targets, torch.Tensor):
            loss = F.cross_entropy(masked_logits, masked_targets)
            accuracy = (masked_logits.argmax(dim=-1) == masked_targets).float().mean()
            pl_module.log("val_loss", loss, prog_bar=True)
            pl_module.log("val_mask_acc", accuracy, prog_bar=True)

        if isinstance(mask_ratio, int | float):
            pl_module.log("val_mask_ratio", torch.tensor(float(mask_ratio)), prog_bar=False)

        if batch_idx == 0 and isinstance(generated_images, torch.Tensor):
            pl_module.log(
                "sample_shape:val", torch.tensor(float(generated_images.shape[0])), prog_bar=False
            )

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
        if not isinstance(outputs, dict):
            return

        masked_logits = outputs.get("masked_logits")
        masked_targets = outputs.get("masked_targets")

        if isinstance(masked_logits, torch.Tensor) and isinstance(masked_targets, torch.Tensor):
            loss = F.cross_entropy(masked_logits, masked_targets)
            accuracy = (masked_logits.argmax(dim=-1) == masked_targets).float().mean()
            pl_module.log("loss:test", loss, prog_bar=True)
            pl_module.log("mask_acc:test", accuracy, prog_bar=True)

        maskgit_model = getattr(pl_module, "maskgit", None)
        if maskgit_model:
            sliding_window_cfg = getattr(maskgit_model, "sliding_window_cfg", None)
            if sliding_window_cfg and sliding_window_cfg.get("enabled", False):
                pl_module.log("sliding_window_enabled:test", torch.tensor(1.0), prog_bar=False)

    def _pop_callback_payload(
        self,
        pl_module: LightningModule,
        stage: str,
    ) -> dict[str, Any] | None:
        pop_payload = getattr(pl_module, "pop_callback_payload", None)
        if callable(pop_payload):
            payload = pop_payload(stage)
            if isinstance(payload, dict):
                return payload
        return None
