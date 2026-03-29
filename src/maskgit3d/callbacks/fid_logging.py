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
        input_min: Minimum value of input data (passed to FIDMetric).
        input_max: Maximum value of input data (passed to FIDMetric).

    Example:
        >>> callback = FIDCallback(input_min=-1.0, input_max=1.0)
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        input_min: float = -1.0,
        input_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_min = input_min
        self.input_max = input_max
        self._fid_metric: FIDMetric | None = None

    def _get_fid_metric(self, pl_module: LightningModule) -> FIDMetric:
        """Lazily initialize FIDMetric with correct device."""
        if self._fid_metric is None:
            device = getattr(pl_module, "device", None)
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._fid_metric = FIDMetric(
                input_min=self.input_min,
                input_max=self.input_max,
                device=device,
            )

        return self._fid_metric

    @staticmethod
    def _extract_batch_pair(outputs: Any) -> tuple[Any, Any] | None:
        """Extract (x_recon, x_real) pair from LightningModule outputs.

        Returns ``None`` if outputs is not a dict or is missing required keys.
        """
        if outputs is None or not isinstance(outputs, dict):
            return None

        x_real = outputs.get("x_real")
        x_recon = outputs.get("x_recon")

        if x_real is not None and x_recon is not None:
            return x_recon, x_real

        return None

    def _on_batch_end(self, pl_module: LightningModule, outputs: Any) -> None:
        """Common logic for accumulating features at batch end."""
        pair = self._extract_batch_pair(outputs)
        if pair is None:
            return

        x_recon, x_real = pair
        fid_metric = self._get_fid_metric(pl_module)
        fid_metric.update(x_recon, x_real)

    def _on_epoch_end(self, pl_module: LightningModule, log_key: str) -> None:
        """Common logic for computing and logging FID at epoch end."""
        if self._fid_metric is not None:
            fid_score = self._fid_metric.compute()
            pl_module.log(log_key, fid_score["fid"], prog_bar=True)
            self._fid_metric.reset()

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
        self._on_batch_end(pl_module, outputs)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log FID at validation epoch end."""
        self._on_epoch_end(pl_module, "val_fid")

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
        self._on_batch_end(pl_module, outputs)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log FID at test epoch end."""
        self._on_epoch_end(pl_module, "fid:test")
