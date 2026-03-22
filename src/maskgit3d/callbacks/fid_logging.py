"""FID callback for computing Fréchet Inception Distance during validation/test."""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

from maskgit3d.metrics.fid import FIDMetric


class FIDCallback(Callback):
    """Callback for computing FID metric during validation and test phases.

    Uses the existing FIDMetric from maskgit3d.metrics.fid with 2.5D approach
    for 3D inputs. Accumulates features during batch end and computes FID
    at epoch end.

    Args:
        spatial_dims: Spatial dimensions (default: 3 for 3D data).
        perceptual_network: Network to use for feature extraction (default: "alex").
            Note: Currently only "alex" (InceptionV3) is supported.

    Example:
        >>> callback = FIDCallback(spatial_dims=3)
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        perceptual_network: str = "alex",
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.perceptual_network = perceptual_network
        self._fid_metric: FIDMetric | None = None

    def _get_fid_metric(self, pl_module: LightningModule) -> FIDMetric:
        """Lazily initialize FIDMetric with correct device."""
        if self._fid_metric is None:
            device = getattr(pl_module, "device", None)
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._fid_metric = FIDMetric(spatial_dims=self.spatial_dims, device=device)

        return self._fid_metric

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate features for FID computation from validation batches."""
        if outputs is None or not isinstance(outputs, dict):
            return

        x_real = outputs.get("x_real")
        x_recon = outputs.get("x_recon")

        if x_real is not None and x_recon is not None:
            fid_metric = self._get_fid_metric(pl_module)
            fid_metric.update(x_recon, x_real)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log FID at validation epoch end."""
        if self._fid_metric is not None:
            fid_score = self._fid_metric.compute()
            pl_module.log("val_fid", fid_score["fid"], prog_bar=True)
            self._fid_metric.reset()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate features for FID computation from test batches."""
        if outputs is None or not isinstance(outputs, dict):
            return

        x_real = outputs.get("x_real")
        x_recon = outputs.get("x_recon")

        if x_real is not None and x_recon is not None:
            fid_metric = self._get_fid_metric(pl_module)
            fid_metric.update(x_recon, x_real)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log FID at test epoch end."""
        if self._fid_metric is not None:
            fid_score = self._fid_metric.compute()
            pl_module.log("fid:test", fid_score["fid"], prog_bar=True)
            self._fid_metric.reset()
