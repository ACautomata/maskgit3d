"""FID (Fréchet Inception Distance) metric implementation using MONAI."""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3  # type: ignore[import-untyped]


class LightningMetricProtocol(Protocol):
    def update(self, predictions: Any, targets: Any) -> None: ...

    def compute(self) -> dict[str, float]: ...

    def reset(self) -> None: ...


class InceptionV3FeatureExtractor(nn.Module):
    """InceptionV3 feature extractor for FID with 2.5D support for 3D inputs.

    For 3D inputs, uses a 2.5D approach similar to MONAI's PerceptualLoss:
    extracts uniformly-spaced slices from all three axes for reproducibility.

    Args:
        input_channels: Number of input channels (default: 1).
        device: Device to run the model on.
        slice_ratio: Ratio of slices to sample per axis for 3D inputs (default: 0.3).
    """

    _imagenet_mean: torch.Tensor
    _imagenet_std: torch.Tensor

    def __init__(
        self,
        input_channels: int = 1,
        device: torch.device | None = None,
        slice_ratio: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.slice_ratio = slice_ratio

        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.eval()
        self.inception.fc = nn.Identity()  # type: ignore
        self.inception.dropout = nn.Identity()  # type: ignore

        for param in self.inception.parameters():
            param.requires_grad = False

        self.device = device or torch.device("cpu")
        self.inception.to(device=self.device, dtype=torch.float32)

        # Cache ImageNet normalization as buffers (auto device/dtype transfer)
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _batchify_axis(self, x: torch.Tensor, spatial_axis: int) -> torch.Tensor:
        """Transform slices from one spatial axis into batch dimension.

        Args:
            x: 5D tensor (B, C, D, H, W).
            spatial_axis: Axis to extract slices from (2, 3, or 4).

        Returns:
            4D tensor with slices as batch dimension.
        """
        preserved = [2, 3, 4]
        preserved.remove(spatial_axis)
        perm = (0, spatial_axis, 1) + tuple(preserved)
        slices = x.permute(perm).contiguous()
        slices = slices.view(-1, x.shape[1], x.shape[preserved[0]], x.shape[preserved[1]])
        return slices

    def _extract_axis_features(self, volume: torch.Tensor, spatial_axis: int) -> torch.Tensor:
        """Extract features from slices along one axis.

        Args:
            volume: 5D tensor (B, C, D, H, W).
            spatial_axis: Axis to extract slices from.

        Returns:
            2D feature tensor (N, 2048) where N is sampled slices.
        """
        slices = self._batchify_axis(volume, spatial_axis)

        n_samples = max(1, int(slices.shape[0] * self.slice_ratio))
        indices = torch.linspace(0, slices.shape[0] - 1, n_samples, device=slices.device).long()
        slices = torch.index_select(slices, dim=0, index=indices)

        features = self._extract_2d_features(slices)
        return features

    def _extract_2d_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from 2D images using InceptionV3."""
        images = images.to(self.device).float()

        if images.shape[-2:] != (299, 299):
            images = nn.functional.interpolate(
                images, size=(299, 299), mode="bilinear", align_corners=False
            )

        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        images = (images - self._imagenet_mean) / self._imagenet_std  # type: ignore[operator]

        with torch.no_grad():
            features = self.inception(images)
            features = torch.flatten(features, 1)

        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images.

        Args:
            images: (B, C, H, W) for 2D or (B, C, D, H, W) for 3D.

        Returns:
            (N, 2048) feature tensor where N depends on slice_ratio for 3D.
        """
        if images.dim() == 4:
            return self._extract_2d_features(images)

        if images.dim() == 5:
            feat_d = self._extract_axis_features(images, 2)
            feat_h = self._extract_axis_features(images, 3)
            feat_w = self._extract_axis_features(images, 4)
            return torch.cat([feat_d, feat_h, feat_w], dim=0)

        raise ValueError(f"Expected 4D or 5D input, got {images.dim()}D")


class FIDMetric:
    """FID metric with a Lightning-style API.

    FID (Fréchet Inception Distance) measures the similarity between two distributions
    of images using InceptionV3 features. For 3D inputs, uses 2.5D approach with
    uniform slice sampling for reproducibility.

    Args:
        input_min: Minimum value of input data.
        input_max: Maximum value of input data.
        slice_ratio: Ratio of slices to sample for 3D inputs (default: 0.3).
        device: Device to run the feature extractor on.
    """

    def __init__(
        self,
        input_min: float = -1.0,
        input_max: float = 1.0,
        slice_ratio: float = 0.3,
        device: torch.device | None = None,
    ) -> None:
        from monai.metrics.fid import get_fid_score

        if input_max <= input_min:
            raise ValueError(f"input_max must be > input_min, got {input_max} <= {input_min}")

        self.input_min = input_min
        self.input_max = input_max
        self._get_fid_score = get_fid_score

        self.feature_extractor = InceptionV3FeatureExtractor(
            input_channels=1,
            device=device,
            slice_ratio=slice_ratio,
        )

        self._pred_features: list[torch.Tensor] = []
        self._target_features: list[torch.Tensor] = []

    def _normalize_to_0_1(self, tensor: torch.Tensor) -> torch.Tensor:
        return ((tensor - self.input_min) / (self.input_max - self.input_min)).clamp(0.0, 1.0)

    def update(self, predictions: Any, targets: Any) -> None:
        pred = self._extract_prediction(predictions)
        target = self._extract_target(targets)

        pred_normalized = self._normalize_to_0_1(pred)
        target_normalized = self._normalize_to_0_1(target)

        pred_features = self.feature_extractor(pred_normalized)
        target_features = self.feature_extractor(target_normalized)

        self._pred_features.append(pred_features.detach().cpu())
        self._target_features.append(target_features.detach().cpu())

    def __call__(self, predictions: Any, targets: Any) -> dict[str, float]:
        """Convenience method: update, compute, and reset in one call.

        Note: Resets internal state after computing. For multi-batch accumulation,
        use ``update()`` per batch and ``compute()`` once at epoch end.
        """
        self.update(predictions, targets)
        result = self.compute()
        self.reset()
        return result

    def compute(self) -> dict[str, float]:
        if len(self._pred_features) == 0:
            return {"fid": float("nan")}

        pred_features = torch.cat(self._pred_features, dim=0)
        target_features = torch.cat(self._target_features, dim=0)

        if pred_features.shape[0] < 2:
            return {"fid": float("nan")}

        fid_score = self._get_fid_score(pred_features, target_features)
        return {"fid": float(fid_score.item())}

    def reset(self) -> None:
        self._pred_features.clear()
        self._target_features.clear()

    def _extract_prediction(self, value: Any) -> torch.Tensor:
        """Extract prediction tensor from various input formats.

        Key priority for dicts: ``x_recon`` > ``images`` > ``volumes``.
        """
        if isinstance(value, torch.Tensor):
            return value

        if isinstance(value, dict):
            if "x_recon" in value:
                value = value["x_recon"]
            elif "images" in value:
                value = value["images"]
            elif "volumes" in value:
                value = value["volumes"]
            else:
                raise ValueError(f"No prediction key found in dict: {list(value.keys())}")

        return torch.as_tensor(value)

    def _extract_target(self, value: Any) -> torch.Tensor:
        """Extract target tensor from various input formats.

        Key priority for dicts: ``x_real`` > ``images`` > ``volumes``.
        """
        if isinstance(value, torch.Tensor):
            return value

        if isinstance(value, dict):
            if "x_real" in value:
                value = value["x_real"]
            elif "images" in value:
                value = value["images"]
            elif "volumes" in value:
                value = value["volumes"]
            else:
                raise ValueError(f"No target key found in dict: {list(value.keys())}")

        return torch.as_tensor(value)
