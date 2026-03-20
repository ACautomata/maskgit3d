from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer

from maskgit3d.metrics.fid import FIDMetric
from maskgit3d.tasks.output_contracts import VQVAETrainingOutput, VQVAEValidationOutput

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
        use_fid: bool = True,
    ) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_val_every_n_batches = log_val_every_n_batches
        self._train_step_count = 0
        self.use_fid = use_fid
        self.fid_metric = FIDMetric(spatial_dims=3) if use_fid else None

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

        callback_payload = self._get_callback_payload(pl_module, "train")
        if callback_payload is None:
            return

        try:
            output = VQVAETrainingOutput(**callback_payload)
        except (TypeError, KeyError):
            return

        x_real = output.x_real
        x_recon = output.x_recon
        vq_loss = output.vq_loss
        last_layer = output.last_layer

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
                # Only show total_loss on progress bar, detailed losses are logged but not shown
                is_total_loss = name.endswith("/total_loss")
                pl_module.log(name, value, prog_bar=is_total_loss)

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

        try:
            output = VQVAEValidationOutput(**outputs)
        except (TypeError, KeyError):
            return

        x_real = output.x_real
        x_recon = output.x_recon
        vq_loss = output.vq_loss

        if x_real is None or x_recon is None or vq_loss is None:
            return

        loss_l1 = F.l1_loss(x_recon, x_real)
        pl_module.log("val_rec_loss", loss_l1, prog_bar=False)

        # Accumulate features for FID (compute at epoch end)
        if self.use_fid and self.fid_metric is not None:
            self.fid_metric.update(x_recon, x_real)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log FID once per validation epoch."""
        if self.use_fid and self.fid_metric is not None:
            fid_score = self.fid_metric.compute()
            pl_module.log("val_fid", fid_score["fid"], prog_bar=True)
            self.fid_metric.reset()

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

        try:
            output = VQVAEValidationOutput(**outputs)
        except (TypeError, KeyError):
            return

        x_real = output.x_real
        x_recon = output.x_recon
        vq_loss = output.vq_loss
        inference_time = output.inference_time
        use_sliding_window = output.use_sliding_window

        if x_real is None or x_recon is None or vq_loss is None:
            return

        loss_l1 = F.l1_loss(x_recon, x_real)

        pl_module.log("loss_l1:test", loss_l1, prog_bar=False)
        pl_module.log("loss_vq:test", vq_loss, prog_bar=False)

        if self.use_fid and self.fid_metric is not None:
            self.fid_metric.update(x_recon, x_real)

        if inference_time is not None:
            pl_module.log("inference_time:test", inference_time, prog_bar=True)

        if use_sliding_window is not None:
            pl_module.log("sliding_window_enabled:test", use_sliding_window, prog_bar=False)

        # Log peak memory if CUDA is available
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            pl_module.log("peak_memory_mb:test", torch.tensor(peak_memory), prog_bar=True)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log FID once per test epoch."""
        if self.use_fid and self.fid_metric is not None:
            fid_score = self.fid_metric.compute()
            pl_module.log("fid:test", fid_score["fid"], prog_bar=True)
            self.fid_metric.reset()

    def _get_callback_payload(
        self,
        pl_module: LightningModule,
        stage: str,
    ) -> dict[str, Any] | None:
        get_payload = getattr(pl_module, "get_callback_payload", None)
        if callable(get_payload):
            payload = get_payload(stage)
            if isinstance(payload, dict):
                return payload

        pop_payload = getattr(pl_module, "pop_callback_payload", None)
        if callable(pop_payload):
            payload = pop_payload(stage)
            if isinstance(payload, dict):
                return payload
        return None
