"""Image quality metrics wrappers based on MONAI."""

from __future__ import annotations

from typing import Any, Protocol

import torch


class LightningMetricProtocol(Protocol):
    def update(self, predictions: Any, targets: Any) -> None: ...

    def compute(self) -> dict[str, float]: ...

    def reset(self) -> None: ...


class ImageMetrics:
    """Wrap MONAI PSNR and SSIM metrics with a Lightning-style API."""

    def __init__(
        self,
        data_range: float = 1.0,
        spatial_dims: int = 3,
        input_min: float = -1.0,
        input_max: float = 1.0,
    ) -> None:
        from monai.metrics.regression import PSNRMetric, SSIMMetric

        if input_max <= input_min:
            raise ValueError(
                f"input_max must be greater than input_min, got {input_max} <= {input_min}"
            )

        self.data_range = data_range
        self.spatial_dims = spatial_dims
        self.input_min = input_min
        self.input_max = input_max

        self.psnr_metric = PSNRMetric(max_val=data_range, reduction="mean")
        self.ssim_metric = SSIMMetric(
            spatial_dims=spatial_dims,
            data_range=data_range,
            reduction="mean",
        )

        self._num_updates = 0
        self._num_ssim_updates = 0

    def update(self, predictions: Any, targets: Any) -> None:
        pred = self._to_tensor(predictions)
        target = self._to_tensor(targets)

        if pred.device != target.device:
            pred = pred.to(target.device)

        pred_normalized = self._normalize_to_data_range(pred)
        target_normalized = self._normalize_to_data_range(target)

        self.psnr_metric(pred_normalized, target_normalized)
        try:
            self.ssim_metric(pred_normalized, target_normalized)
            self._num_ssim_updates += 1
        except RuntimeError:
            pass

        self._num_updates += 1

    def __call__(self, predictions: Any, targets: Any) -> dict[str, float]:
        self.update(predictions, targets)
        return self.compute()

    def compute(self) -> dict[str, float]:
        if self._num_updates == 0:
            return {"psnr": 0.0, "ssim": 0.0}

        psnr = self._to_scalar(self.psnr_metric.aggregate())

        ssim = 0.0 if self._num_ssim_updates == 0 else self._to_scalar(self.ssim_metric.aggregate())

        return {"psnr": psnr, "ssim": ssim}

    def reset(self) -> None:
        self.psnr_metric.reset()
        self.ssim_metric.reset()
        self._num_updates = 0
        self._num_ssim_updates = 0

    def _normalize_to_data_range(self, tensor: torch.Tensor) -> torch.Tensor:
        scaled = (tensor - self.input_min) / (self.input_max - self.input_min)
        return scaled * self.data_range

    def _to_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value

        if isinstance(value, dict):
            if "images" in value:
                value = value["images"]
            elif "volumes" in value:
                value = value["volumes"]
            else:
                raise ValueError(
                    f"Unsupported metric input keys: {list(value.keys())}. Expected 'images' or 'volumes'."
                )

        return torch.as_tensor(value)

    def _to_scalar(self, value: torch.Tensor | tuple[Any, ...]) -> float:
        if isinstance(value, tuple):
            first = value[0]
            if isinstance(first, torch.Tensor):
                return float(first.mean().item())
            return float(first)

        if isinstance(value, torch.Tensor):
            return float(value.mean().item())

        return float(value)
