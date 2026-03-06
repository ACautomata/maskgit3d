"""
FID 2.5D metric for 3D medical image evaluation.

This module provides FID (Fréchet Inception Distance) computation
using a 2.5D approach - extracting 2D slices from 3D volumes along
three orthogonal planes (XY, YZ, ZX) and computing features using
a pretrained 2D CNN (e.g., RadImageNet ResNet50).

Based on MONAI's FID 2.5D implementation for medical image generation evaluation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn

from maskgit3d.domain.interfaces import Metrics

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class FID2p5DMetric(Metrics):
    """
    2.5D FID metric for 3D medical image evaluation.

    This metric computes the Fréchet Inception Distance between real and synthetic
    3D volumes by extracting 2D slices along three orthogonal planes (XY, YZ, ZX),
    extracting features using a pretrained 2D CNN, and computing FID for each plane.

    The final FID is the average of the three plane-wise FIDs.

    Args:
        feature_network: Pretrained feature extraction network (e.g., RadImageNet ResNet50).
                         If None, will automatically load RadImageNet ResNet50.
        model_name: Name of the feature extraction model to use if feature_network is None.
                    Options: "radimagenet_resnet50", "squeezenet1_1"
        device: Device to run feature extraction on. If None, uses CUDA if available.
        center_slices_ratio: Ratio of center slices to use (0.0-1.0).
                             If None, uses all slices.
        xy_only: If True, only compute FID for XY plane (faster, less accurate).
        allow_remote_code: If True, allows downloading models from remote repositories.
                          Required for RadImageNet models. Defaults to False for security.

    Example:
        >>> metric = FID2p5DMetric(model_name="squeezenet1_1")
        >>> for batch in dataloader:
        ...     metric.update(predictions, targets)
        >>> results = metric.compute()
        >>> print(f"FID: {results['fid']:.2f}")
    """

    def __init__(
        self,
        feature_network: nn.Module | None = None,
        model_name: str = "squeezenet1_1",
        device: torch.device | str | None = None,
        center_slices_ratio: float | None = None,
        xy_only: bool = False,
        allow_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.xy_only = xy_only
        self.allow_remote_code = allow_remote_code

        # Validate center_slices_ratio
        if center_slices_ratio is not None and not 0.0 < center_slices_ratio <= 1.0:
            raise ValueError(f"center_slices_ratio must be in (0, 1], got {center_slices_ratio}")
        self.center_slices_ratio = center_slices_ratio

        # Initialize feature network
        if feature_network is not None:
            self.feature_network = feature_network.to(self.device)
            # Infer feature dimension from network
            self._feature_dim = self._infer_feature_dim()
        else:
            self.feature_network, self._feature_dim = self._load_feature_network(model_name)

        self.feature_network.eval()
        for param in self.feature_network.parameters():
            param.requires_grad = False

        # Storage for features
        self.real_features_xy: list[torch.Tensor] = []
        self.real_features_yz: list[torch.Tensor] = []
        self.real_features_zx: list[torch.Tensor] = []
        self.fake_features_xy: list[torch.Tensor] = []
        self.fake_features_yz: list[torch.Tensor] = []
        self.fake_features_zx: list[torch.Tensor] = []

        self._count = 0

    def _infer_feature_dim(self) -> int:
        """Infer feature dimension by running a dummy forward pass."""
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            output = self.feature_network(dummy_input)
            if isinstance(output, tuple):
                output = output[0]
        return output.shape[1] if len(output.shape) > 1 else output.shape[0]

    def _load_feature_network(self, model_name: str) -> tuple[nn.Module, int]:
        """Load pretrained feature extraction network."""
        logger.info(f"Loading feature network: {model_name}")

        if model_name == "radimagenet_resnet50":
            if not self.allow_remote_code:
                raise ValueError(
                    "Loading RadImageNet requires allow_remote_code=True. "
                    "This is a security risk as it downloads and executes remote code. "
                    "Only enable this if you trust the source: Warvito/radimagenet-models"
                )
            try:
                network = cast(
                    nn.Module,
                    torch.hub.load(
                        "Warvito/radimagenet-models",
                        model="radimagenet_resnet50",
                        verbose=False,
                        trust_repo=True,
                    ),
                )
                logger.info("Loaded RadImageNet ResNet50")
                # RadImageNet ResNet50 outputs 2048-dim features
                return network, 2048  # type: ignore[return-value]
            except Exception as e:
                logger.error(f"Failed to load RadImageNet: {e}")
                raise RuntimeError(f"Failed to load RadImageNet: {e}") from e

        if model_name == "squeezenet1_1":
            import torchvision

            # Use weights parameter instead of deprecated pretrained
            try:
                from torchvision.models import SqueezeNet1_1_Weights

                network = torchvision.models.squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
            except ImportError:
                # Fallback for older torchvision versions
                network = torchvision.models.squeezenet1_1(pretrained=True)

            logger.info("Loaded SqueezeNet 1.1")
            # SqueezeNet classifier outputs 1000-dim features
            return network, 1000

        raise ValueError(
            f"Unknown model_name: {model_name}. Supported: radimagenet_resnet50, squeezenet1_1"
        )

    def reset(self) -> None:
        """Reset accumulated features."""
        self.real_features_xy = []
        self.real_features_yz = []
        self.real_features_zx = []
        self.fake_features_xy = []
        self.fake_features_yz = []
        self.fake_features_zx = []
        self._count = 0

    def update(
        self,
        predictions: torch.Tensor | dict[str, Any] | np.ndarray,
        targets: torch.Tensor | dict[str, Any] | np.ndarray,
    ) -> None:
        """
        Update metrics with new batch of predictions and targets.

        Args:
            predictions: Model predictions (reconstructed/generated images).
                          Can be dict with "images" key or torch.Tensor.
            targets: Ground truth images. Can be torch.Tensor or numpy array.
        """
        # Extract tensors from inputs
        pred_tensor = self._extract_tensor(predictions)
        target_tensor = self._extract_tensor(targets)

        # Ensure tensors
        if not isinstance(pred_tensor, torch.Tensor):
            pred_tensor = torch.from_numpy(pred_tensor)
        if not isinstance(target_tensor, torch.Tensor):
            target_tensor = torch.from_numpy(target_tensor)

        # Normalize to [-1, 1] range if needed
        pred_tensor = self._normalize_range(pred_tensor)
        target_tensor = self._normalize_range(target_tensor)

        # Move to device
        pred_tensor = pred_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)

        # Extract features
        with torch.no_grad():
            pred_feats = self._extract_features_2p5d(pred_tensor)
            target_feats = self._extract_features_2p5d(target_tensor)

        # Store features - xy is always present, yz/zx only when not xy_only
        self.fake_features_xy.append(pred_feats[0])
        self.real_features_xy.append(target_feats[0])

        if not self.xy_only:
            if pred_feats[1] is not None:
                self.fake_features_yz.append(pred_feats[1])  # type: ignore[arg-type]
            if pred_feats[2] is not None:
                self.fake_features_zx.append(pred_feats[2])  # type: ignore[arg-type]
            if target_feats[1] is not None:
                self.real_features_yz.append(target_feats[1])  # type: ignore[arg-type]
            if target_feats[2] is not None:
                self.real_features_zx.append(target_feats[2])  # type: ignore[arg-type]

        self._count += pred_tensor.shape[0]

    def _extract_tensor(self, data: Any) -> torch.Tensor:
        """Extract torch.Tensor from various input formats."""
        if isinstance(data, torch.Tensor):
            return data

        if isinstance(data, dict):
            if "images" in data:
                data = data["images"]
            elif "volumes" in data:
                data = data["volumes"]
            elif "masks" in data:
                data = data["masks"]
            else:
                raise ValueError(
                    f"Dict must contain 'images', 'volumes', or 'masks' key. Got: {data.keys()}"
                )

            # Convert numpy to tensor if needed
            if hasattr(data, "numpy"):
                return torch.from_numpy(data)
            return data

        # Assume numpy array
        if hasattr(data, "shape") and hasattr(data, "dtype"):
            import numpy as np

            return torch.from_numpy(np.asarray(data))

        raise ValueError(f"Cannot extract tensor from {type(data)}")

    def _normalize_range(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [-1, 1] range if in [0, 1]."""
        # Check if tensor is in [0, 1] range
        if tensor.min() >= 0 and tensor.max() <= 1:
            # Convert from [0, 1] to [-1, 1]
            return tensor * 2 - 1
        return tensor

    def _extract_features_2p5d(
        self,
        volume: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Extract 2.5D features from a 3D volume.

        Args:
            volume: Input volume [B, C, D, H, W] in [-1, 1] range.

        Returns:
            Tuple of (XY_features, YZ_features, ZX_features).
            If xy_only is True, YZ and ZX features will be None.
        """
        # Ensure 5D tensor [B, C, D, H, W]
        if volume.dim() == 4:
            volume = volume.unsqueeze(2)  # Add D dimension

        B, C, D, H, W = volume.shape

        # If single channel, replicate to 3 channels
        if C == 1:
            volume = volume.repeat(1, 3, 1, 1, 1)
        elif C != 3:
            # Take first 3 channels
            volume = volume[:, :3, ...]

        # Convert from RGB to BGR for ImageNet pretrained models
        volume = volume[:, [2, 1, 0], ...]

        # Normalize to [0, 1] for ImageNet models
        volume = (volume + 1) / 2  # [-1, 1] -> [0, 1]

        features_xy = self._extract_xy_features(volume)

        if self.xy_only:
            return features_xy, None, None

        features_yz = self._extract_yz_features(volume)
        features_zx = self._extract_zx_features(volume)

        return features_xy, features_yz, features_zx

    def _extract_xy_features(self, volume: torch.Tensor) -> torch.Tensor:
        """Extract features from XY planes (slicing along D axis)."""
        B, C, D, H, W = volume.shape

        # Get slices along D axis
        if self.center_slices_ratio is not None:
            start_d = int((1.0 - self.center_slices_ratio) / 2.0 * D)
            end_d = int((1.0 + self.center_slices_ratio) / 2.0 * D)
            # Ensure at least one slice
            if start_d >= end_d:
                start_d = 0
                end_d = max(1, D // 2)
            slices = torch.unbind(volume[:, :, start_d:end_d, :, :], dim=2)
        else:
            slices = torch.unbind(volume, dim=2)

        if not slices:
            # Return zero features if no slices
            return torch.zeros(B, self._feature_dim, device=volume.device)

        # Concatenate all slices
        images_2d = torch.cat(slices, dim=0)  # [B * num_slices, C, H, W]

        # Normalize for ImageNet
        images_2d = self._radimagenet_normalize(images_2d)

        # Extract features
        with torch.no_grad():
            feats = self.feature_network.forward(images_2d)
            if isinstance(feats, tuple):
                feats = feats[0]
            feats = self._spatial_average(feats, keepdim=False)

        return feats

    def _extract_yz_features(self, volume: torch.Tensor) -> torch.Tensor:
        """Extract features from YZ planes (slicing along H axis)."""
        B, C, D, H, W = volume.shape

        if self.center_slices_ratio is not None:
            start_h = int((1.0 - self.center_slices_ratio) / 2.0 * H)
            end_h = int((1.0 + self.center_slices_ratio) / 2.0 * H)
            if start_h >= end_h:
                start_h = 0
                end_h = max(1, H // 2)
            slices = torch.unbind(volume[:, :, :, start_h:end_h, :], dim=3)
        else:
            slices = torch.unbind(volume, dim=3)

        if not slices:
            return torch.zeros(B, self._feature_dim, device=volume.device)

        images_2d = torch.cat(slices, dim=0)
        images_2d = self._radimagenet_normalize(images_2d)

        with torch.no_grad():
            feats = self.feature_network.forward(images_2d)
            if isinstance(feats, tuple):
                feats = feats[0]
            feats = self._spatial_average(feats, keepdim=False)

        return feats

    def _extract_zx_features(self, volume: torch.Tensor) -> torch.Tensor:
        """Extract features from ZX planes (slicing along W axis)."""
        B, C, D, H, W = volume.shape

        if self.center_slices_ratio is not None:
            start_w = int((1.0 - self.center_slices_ratio) / 2.0 * W)
            end_w = int((1.0 + self.center_slices_ratio) / 2.0 * W)
            if start_w >= end_w:
                start_w = 0
                end_w = max(1, W // 2)
            slices = torch.unbind(volume[:, :, :, :, start_w:end_w], dim=4)
        else:
            slices = torch.unbind(volume, dim=4)

        if not slices:
            return torch.zeros(B, self._feature_dim, device=volume.device)

        images_2d = torch.cat(slices, dim=0)
        images_2d = self._radimagenet_normalize(images_2d)

        with torch.no_grad():
            feats = self.feature_network.forward(images_2d)
            if isinstance(feats, tuple):
                feats = feats[0]
            feats = self._spatial_average(feats, keepdim=False)

        return feats

    def _radimagenet_normalize(self, images: torch.Tensor) -> torch.Tensor:
        """
        Normalize images for ImageNet pretrained models.

        Args:
            images: Input images [B, C, H, W] in [0, 1] range.

        Returns:
            Normalized images with ImageNet mean subtracted.
        """
        # Min-max normalize to [0, 1]
        min_val = images.min()
        max_val = images.max()
        if max_val > min_val:
            images = (images - min_val) / (max_val - min_val + 1e-10)

        # Subtract ImageNet mean (BGR order)
        mean = torch.tensor([0.406, 0.456, 0.485], device=images.device).view(1, 3, 1, 1)
        images = images - mean

        return images

    def _spatial_average(self, x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        """Average out spatial dimensions."""
        dim = len(x.shape)

        if dim == 2:
            return x
        elif dim == 3:
            return x.mean([2], keepdim=keepdim)
        elif dim == 4:
            return x.mean([2, 3], keepdim=keepdim)
        elif dim == 5:
            return x.mean([2, 3, 4], keepdim=keepdim)

        return x

    def compute(self) -> dict[str, float]:
        """
        Compute FID metrics from accumulated features.

        Returns:
            Dictionary with FID scores:
            - fid: Average FID across all planes
            - fid_xy: FID for XY plane
            - fid_yz: FID for YZ plane (if not xy_only)
            - fid_zx: FID for ZX plane (if not xy_only)
        """
        if self._count == 0:
            return {"fid": float("inf")}

        # Import MONAI FIDMetric
        try:
            from monai.metrics.fid import FIDMetric
        except ImportError as e:
            raise ImportError(
                "MONAI FIDMetric is required for FID computation. "
                "Please install MONAI: pip install monai"
            ) from e

        fid_metric = FIDMetric()

        # Stack features
        real_xy = torch.vstack(self.real_features_xy)
        fake_xy = torch.vstack(self.fake_features_xy)

        # Compute XY FID
        fid_xy = fid_metric(fake_xy, real_xy).item()

        results: dict[str, float] = {
            "fid_xy": fid_xy,
        }

        if not self.xy_only:
            real_yz = torch.vstack(self.real_features_yz)
            fake_yz = torch.vstack(self.fake_features_yz)
            real_zx = torch.vstack(self.real_features_zx)
            fake_zx = torch.vstack(self.fake_features_zx)

            fid_yz = fid_metric(fake_yz, real_yz).item()
            fid_zx = fid_metric(fake_zx, real_zx).item()

            results["fid_yz"] = fid_yz
            results["fid_zx"] = fid_zx
            results["fid"] = (fid_xy + fid_yz + fid_zx) / 3.0
        else:
            results["fid"] = fid_xy

        return results
