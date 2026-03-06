"""
FID 2.5D callback for epoch-level metric computation.

This module provides a Fabric-compatible callback that computes FID 2.5D
metrics at the end of validation epochs, storing predictions throughout
the epoch and computing FID once at the end.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from maskgit3d.infrastructure.training.callbacks import Callback

if TYPE_CHECKING:
    from lightning.pytorch import Trainer

logger = logging.getLogger(__name__)


class FID2p5DCallback(Callback):
    """
    Callback for computing FID 2.5D metrics during validation epochs.

    This callback accumulates real and fake images throughout the validation epoch,
    then computes FID metrics at the end of the epoch. This is more efficient
    and accurate than per-batch FID computation.

    Args:
        enabled: Whether to enable FID 2.5D computation.
        model_name: Feature extraction model name ("squeezenet1_1" or "radimagenet_resnet50").
        center_slices_ratio: Ratio of center slices to use (0.0-1.0).
        xy_only: If True, only compute XY plane FID (faster).
        max_samples: Maximum number of samples to use for FID computation per epoch.
        allow_remote_code: Allow downloading RadImageNet models (security risk).

    Example:
        >>> callback = FID2p5DCallback(
        ...     enabled=True,
        ...     model_name="squeezenet1_1",
        ...     xy_only=True,
        ... )
    """

    def __init__(
        self,
        enabled: bool = False,
        model_name: str = "squeezenet1_1",
        center_slices_ratio: float | None = None,
        xy_only: bool = False,
        max_samples: int = 500,
        allow_remote_code: bool = False,
    ):
        super().__init__()
        self.enabled = enabled
        self.model_name = model_name
        self.center_slices_ratio = center_slices_ratio
        self.xy_only = xy_only
        self.max_samples = max_samples
        self.allow_remote_code = allow_remote_code

        self._metric: Any = None
        self._stored_real: list[torch.Tensor] = []
        self._stored_fake: list[torch.Tensor] = []
        self._current_samples: int = 0

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: torch.nn.Module,
    ) -> None:
        """Reset metric and storage at the start of validation epoch."""
        if not self.enabled:
            return

        # Lazy initialization of metric
        if self._metric is None:
            from maskgit3d.infrastructure.metrics.fid_2p5d import FID2p5DMetric

            self._metric = FID2p5DMetric(
                model_name=self.model_name,
                center_slices_ratio=self.center_slices_ratio,
                xy_only=self.xy_only,
                allow_remote_code=self.allow_remote_code,
            )

        self._metric.reset()
        self._stored_real = []
        self._stored_fake = []
        self._current_samples = 0

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: torch.nn.Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:  # type: ignore[override]
        """Store predictions and targets from validation batch."""
        if not self.enabled:
            return

        if self._current_samples >= self.max_samples:
            return

        # Extract images from batch
        images = batch[0] if isinstance(batch, (tuple, list)) else batch

        # Get model predictions from outputs
        # Outputs typically contains reconstructions from validation_step
        if isinstance(outputs, dict) and "reconstructions" in outputs:
            reconstructions = outputs["reconstructions"]
        elif isinstance(outputs, dict) and "x_rec" in outputs:
            reconstructions = outputs["x_rec"]
        elif isinstance(outputs, torch.Tensor):
            reconstructions = outputs
        else:
            # Try to get from module's last output
            return

        # Ensure tensors and compute remaining capacity
        remaining = self.max_samples - self._current_samples
        batch_size = min(images.shape[0], remaining)

        if batch_size <= 0:
            return

        # Store as tensors (avoiding numpy conversion)
        self._stored_real.append(images[:batch_size].detach().cpu())
        self._stored_fake.append(reconstructions[:batch_size].detach().cpu())
        self._current_samples += batch_size

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: torch.nn.Module,
    ) -> None:
        """Compute and log FID metrics at the end of validation epoch."""
        if not self.enabled or self._metric is None or not self._stored_real:
            return

        try:
            all_real = torch.cat(self._stored_real, dim=0)
            all_fake = torch.cat(self._stored_fake, dim=0)

            self._metric.update(all_fake, all_real)

            fid_results = self._metric.compute()

            for key, value in fid_results.items():
                trainer.callback_metrics[f"val_{key}"] = value

            logger.info(f"FID 2.5D Results: {fid_results}")

        except Exception as e:
            logger.warning(f"Failed to compute FID 2.5D: {e}")

        self._stored_real = []
        self._stored_fake = []
