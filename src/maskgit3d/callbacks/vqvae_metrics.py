from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer

if TYPE_CHECKING:
    pass


class VQVAEMetricsCallback(Callback):
    """Callback for computing and logging VQVAE-specific metrics.

    This callback receives raw data (x_real, x_recon, vq_loss, etc.) from
    VQVAETask's step methods and computes all metrics in the callback.

    Args:
        log_every_n_steps: Log training metrics every N steps (0 = every step).
        log_val_every_n_batches: Log validation metrics every N batches.

    Example:
        >>> callback = VQVAEMetricsCallback()
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
        """Compute and log training metrics from raw outputs."""
        self._train_step_count += 1

        if self.log_every_n_steps > 1 and self._train_step_count % self.log_every_n_steps != 0:
            return

        if outputs is None:
            return

        callback_payload = self._pop_callback_payload(pl_module, "train")
        if callback_payload is None:
            return

        x_real = callback_payload.get("x_real")
        x_recon = callback_payload.get("x_recon")
        vq_loss = callback_payload.get("vq_loss")
        last_layer = callback_payload.get("last_layer")

        if x_real is None or x_recon is None or vq_loss is None:
            return

        # Compute metrics using loss_fn
        loss_fn = pl_module.loss_fn  # type: ignore[attr-defined,union-attr]

        # Compute generator metrics
        loss_g, log_g = loss_fn(  # type: ignore[operator]
            inputs=x_real,
            reconstructions=x_recon,
            vq_loss=vq_loss,
            optimizer_idx=0,
            global_step=pl_module.global_step,
            last_layer=last_layer,
            split="train",
        )

        # Compute discriminator metrics
        _, log_d = loss_fn(  # type: ignore[operator]
            inputs=x_real,
            reconstructions=x_recon,
            vq_loss=vq_loss,
            optimizer_idx=1,
            global_step=pl_module.global_step,
            split="train",
        )

        for name, value in {**log_g, **log_d}.items():
            if isinstance(value, torch.Tensor):
                pl_module.log(name, value, prog_bar=True)

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
        """Compute and log validation metrics from raw outputs."""
        if outputs is None or not isinstance(outputs, dict):
            return

        x_real = outputs.get("x_real")
        x_recon = outputs.get("x_recon")
        vq_loss = outputs.get("vq_loss")

        if x_real is None or x_recon is None or vq_loss is None:
            return

        loss_l1 = F.l1_loss(x_recon, x_real)
        pl_module.log("val_rec_loss", loss_l1, prog_bar=True)

        loss_fn = pl_module.loss_fn  # type: ignore[attr-defined]
        perceptual_loss = torch.tensor(0.0, device=x_real.device)
        if loss_fn.use_perceptual and loss_fn.perceptual_loss is not None:  # type: ignore[attr-defined,union-attr]
            with torch.no_grad():
                perceptual_loss = loss_fn.perceptual_loss(x_recon, x_real)  # type: ignore[attr-defined,union-attr,operator]
        pl_module.log("val_perceptual_loss", perceptual_loss, prog_bar=True)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute and log test metrics from raw outputs."""
        if outputs is None or not isinstance(outputs, dict):
            return

        x_real = outputs.get("x_real")
        x_recon = outputs.get("x_recon")
        vq_loss = outputs.get("vq_loss")
        inference_time = outputs.get("inference_time")
        use_sliding_window = outputs.get("use_sliding_window")

        if x_real is None or x_recon is None or vq_loss is None:
            return

        loss_l1 = F.l1_loss(x_recon, x_real)

        pl_module.log("loss_l1:test", loss_l1, prog_bar=True)
        pl_module.log("loss_vq:test", vq_loss, prog_bar=True)

        if inference_time is not None:
            pl_module.log("inference_time:test", inference_time, prog_bar=True)

        if use_sliding_window is not None:
            pl_module.log("sliding_window_enabled:test", use_sliding_window, prog_bar=False)

        # Log peak memory if CUDA is available
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            pl_module.log("peak_memory_mb:test", torch.tensor(peak_memory), prog_bar=True)

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
