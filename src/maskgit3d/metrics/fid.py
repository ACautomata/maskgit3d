"""FID (Fréchet Inception Distance) metric implementation using MONAI."""

from __future__ import annotations

from typing import Any, Protocol

import torch


class LightningMetricProtocol(Protocol):
    def update(self, predictions: Any, targets: Any) -> None: ...

    def compute(self) -> dict[str, float]: ...

    def reset(self) -> None: ...


class FIDMetric:
    """FID metric with a Lightning-style API.

    FID (Fréchet Inception Distance) measures the similarity between two distributions
    of images by computing the distance between feature vectors extracted from a
    pretrained network.

    This implementation accumulates features across batches and computes FID at the end.

    Args:
        feature_extractor: Pretrained network for feature extraction.
            If None, uses global average pooling as simple feature extraction.
        spatial_dims: Spatial dimensions of input images (2 or 3).
        input_min: Minimum value of input data (for normalization).
        input_max: Maximum value of input data (for normalization).

    Example:
        >>> fid_metric = FIDMetric(spatial_dims=3)
        >>> for batch in dataloader:
        ...     fid_metric.update(generated_images, real_images)
        >>> result = fid_metric.compute()
        >>> print(result["fid"])
    """

    def __init__(
        self,
        feature_extractor: torch.nn.Module | None = None,
        spatial_dims: int = 3,
        input_min: float = -1.0,
        input_max: float = 1.0,
    ) -> None:
        from monai.metrics.fid import get_fid_score

        if input_max <= input_min:
            raise ValueError(
                f"input_max must be greater than input_min, got {input_max} <= {input_min}"
            )

        self.spatial_dims = spatial_dims
        self.input_min = input_min
        self.input_max = input_max
        self.feature_extractor = feature_extractor
        self._get_fid_score = get_fid_score

        self._pred_features: list[torch.Tensor] = []
        self._target_features: list[torch.Tensor] = []

    def _normalize_to_0_1(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.input_min) / (self.input_max - self.input_min)

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.feature_extractor is not None:
            with torch.no_grad():
                return self.feature_extractor(images)

        spatial_dims = list(range(2, 2 + self.spatial_dims))
        features = torch.mean(images, dim=spatial_dims)
        return features

    def update(self, predictions: Any, targets: Any) -> None:
        pred = self._to_tensor(predictions)
        target = self._to_tensor(targets)

        if pred.device != target.device:
            pred = pred.to(target.device)

        pred_normalized = self._normalize_to_0_1(pred)
        target_normalized = self._normalize_to_0_1(target)

        pred_features = self._extract_features(pred_normalized)
        target_features = self._extract_features(target_normalized)

        self._pred_features.append(pred_features.detach().cpu())
        self._target_features.append(target_features.detach().cpu())

    def __call__(self, predictions: Any, targets: Any) -> dict[str, float]:
        self.update(predictions, targets)
        return self.compute()

    def compute(self) -> dict[str, float]:
        if len(self._pred_features) == 0:
            return {"fid": 0.0}

        pred_features = torch.cat(self._pred_features, dim=0)
        target_features = torch.cat(self._target_features, dim=0)

        if pred_features.shape[0] < 2:
            return {"fid": 0.0}

        fid_score = self._get_fid_score(pred_features, target_features)

        return {"fid": float(fid_score.item())}

    def reset(self) -> None:
        self._pred_features.clear()
        self._target_features.clear()

    def _to_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value

        if isinstance(value, dict):
            if "images" in value:
                value = value["images"]
            elif "volumes" in value:
                value = value["volumes"]
            elif "x_recon" in value:
                value = value["x_recon"]
            elif "x_real" in value:
                value = value["x_real"]
            else:
                raise ValueError(
                    f"Unsupported metric input keys: {list(value.keys())}. "
                    "Expected 'images', 'volumes', 'x_recon', or 'x_real'."
                )

        return torch.as_tensor(value)
