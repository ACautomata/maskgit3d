"""
nnUNet-style data augmentation for 3D medical imaging.

This module provides data augmentation transforms following nnUNet's approach,
implemented using MONAI transforms.

Reference:
    - nnUNet: https://github.com/MIC-DKFZ/nnUNet
    - Default augmentation params from nnUNetv2
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import (
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
)
from monai.transforms.spatial.dictionary import (
    RandFlipd,
    RandRotated,
    RandZoomd,
)
from monai.transforms.transform import MapTransform


class NnUNetAugmentationConfig:
    """Configuration for nnUNet-style data augmentation."""

    def __init__(
        self,
        rotation_x: tuple[float, float] = (-15.0 / 360.0, 15.0 / 360.0),
        rotation_y: tuple[float, float] = (-15.0 / 360.0, 15.0 / 360.0),
        rotation_z: tuple[float, float] = (-15.0 / 360.0, 15.0 / 360.0),
        p_rotation: float = 0.2,
        scale_range: tuple[float, float] = (0.85, 1.25),
        p_scale: float = 0.2,
        gamma_range: tuple[float, float] = (0.7, 1.5),
        p_gamma: float = 0.3,
        brightness_range: tuple[float, float] = (0.75, 1.25),
        p_brightness: float = 0.15,
        p_mirror: float = 0.5,
        gaussian_noise_mean: float = 0.0,
        gaussian_noise_std: float = 0.1,
        p_gaussian_noise: float = 0.1,
        gaussian_smooth_sigma: tuple[float, float] = (0.5, 1.0),
        p_gaussian_smooth: float = 0.2,
        prob: float = 1.0,
    ):
        """Initialize nnUNet augmentation configuration."""
        self.rotation_x = rotation_x
        self.rotation_y = rotation_y
        self.rotation_z = rotation_z
        self.p_rotation = p_rotation
        self.scale_range = scale_range
        self.p_scale = p_scale
        self.gamma_range = gamma_range
        self.p_gamma = p_gamma
        self.brightness_range = brightness_range
        self.p_brightness = p_brightness
        self.p_mirror = p_mirror
        self.gaussian_noise_mean = gaussian_noise_mean
        self.gaussian_noise_std = gaussian_noise_std
        self.p_gaussian_noise = p_gaussian_noise
        self.gaussian_smooth_sigma = gaussian_smooth_sigma
        self.p_gaussian_smooth = p_gaussian_smooth
        self.prob = prob

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "rotation_x": self.rotation_x,
            "rotation_y": self.rotation_y,
            "rotation_z": self.rotation_z,
            "p_rotation": self.p_rotation,
            "scale_range": self.scale_range,
            "p_scale": self.p_scale,
            "gamma_range": self.gamma_range,
            "p_gamma": self.p_gamma,
            "brightness_range": self.brightness_range,
            "p_brightness": self.p_brightness,
            "p_mirror": self.p_mirror,
            "gaussian_noise_mean": self.gaussian_noise_mean,
            "gaussian_noise_std": self.gaussian_noise_std,
            "p_gaussian_noise": self.p_gaussian_noise,
            "gaussian_smooth_sigma": self.gaussian_smooth_sigma,
            "p_gaussian_smooth": self.p_gaussian_smooth,
            "prob": self.prob,
        }

    @classmethod
    def default_3d(cls) -> NnUNetAugmentationConfig:
        """Get default 3D augmentation configuration (nnUNet v2 defaults)."""
        return cls()

    @classmethod
    def light_3d(cls) -> NnUNetAugmentationConfig:
        """Get light 3D augmentation configuration (reduced augmentation)."""
        return cls(
            p_rotation=0.1,
            p_scale=0.1,
            p_gamma=0.15,
            p_brightness=0.1,
            p_mirror=0.5,
            p_gaussian_noise=0.05,
            p_gaussian_smooth=0.1,
        )

    @classmethod
    def heavy_3d(cls) -> NnUNetAugmentationConfig:
        """Get heavy 3D augmentation configuration (aggressive augmentation)."""
        return cls(
            rotation_x=(-30.0 / 360.0, 30.0 / 360.0),
            rotation_y=(-30.0 / 360.0, 30.0 / 360.0),
            rotation_z=(-30.0 / 360.0, 30.0 / 360.0),
            p_rotation=0.3,
            scale_range=(0.75, 1.5),
            p_scale=0.3,
            gamma_range=(0.5, 2.0),
            p_gamma=0.4,
            brightness_range=(0.5, 1.5),
            p_brightness=0.25,
            p_mirror=0.5,
            gaussian_noise_std=0.15,
            p_gaussian_noise=0.15,
            p_gaussian_smooth=0.25,
        )


def create_nnunet_augmentation_transforms(
    keys: Sequence[str],
    config: NnUNetAugmentationConfig | None = None,
) -> Compose:
    """Create nnUNet-style augmentation transforms."""
    if config is None:
        config = NnUNetAugmentationConfig.default_3d()

    transforms = []

    transforms.append(
        RandRotated(
            keys=keys,
            range_x=config.rotation_x,
            range_y=config.rotation_y,
            range_z=config.rotation_z,
            prob=config.p_rotation,
            mode=("bilinear", "nearest") if len(keys) > 1 else "bilinear",
            padding_mode="border",
        )
    )

    transforms.append(
        RandZoomd(
            keys=keys,
            min_zoom=config.scale_range[0],
            max_zoom=config.scale_range[1],
            prob=config.p_scale,
            mode=("trilinear", "nearest") if len(keys) > 1 else "trilinear",
            padding_mode="edge",
        )
    )

    transforms.append(
        RandFlipd(
            keys=keys,
            spatial_axis=[0, 1, 2],
            prob=config.p_mirror,
        )
    )

    transforms.append(
        RandAdjustContrastd(
            keys=keys[0],
            prob=config.p_gamma,
            gamma=config.gamma_range,
        )
    )

    transforms.append(
        RandScaleIntensityd(
            keys=keys[0],
            factors=config.brightness_range,
            prob=config.p_brightness,
        )
    )

    transforms.append(
        RandGaussianNoised(
            keys=keys[0],
            prob=config.p_gaussian_noise,
            mean=config.gaussian_noise_mean,
            std=config.gaussian_noise_std,
        )
    )

    transforms.append(
        RandGaussianSmoothd(
            keys=keys[0],
            prob=config.p_gaussian_smooth,
            sigma_x=config.gaussian_smooth_sigma,
            sigma_y=config.gaussian_smooth_sigma,
            sigma_z=config.gaussian_smooth_sigma,
        )
    )

    return Compose(transforms)


def create_training_transforms_with_augmentation(
    keys: Sequence[str],
    crop_size: Sequence[int],
    normalization_transforms: list[MapTransform] | None = None,
    augmentation_config: NnUNetAugmentationConfig | None = None,
) -> Compose:
    """Create complete training transforms with preprocessing and augmentation."""
    from monai.transforms.croppad.dictionary import RandSpatialCropd, SpatialPadd

    transforms = []

    if normalization_transforms:
        transforms.extend(normalization_transforms)

    transforms.append(SpatialPadd(keys=keys, spatial_size=crop_size, mode="constant"))

    transforms.append(
        RandSpatialCropd(
            keys=keys,
            roi_size=crop_size,
            random_center=True,
            random_size=False,
        )
    )

    aug_transforms = create_nnunet_augmentation_transforms(keys, augmentation_config)
    transforms.extend(aug_transforms.transforms)

    return Compose(transforms)


def create_brats_training_transforms_with_augmentation(
    crop_size: Sequence[int] = (128, 128, 128),
    normalize_mode: str = "zscore",
    augmentation_config: NnUNetAugmentationConfig | None = None,
    task: str = "reconstruction",
) -> Compose:
    """Create BraTS training transforms with nnUNet-style augmentation."""
    from monai.transforms.croppad.dictionary import RandSpatialCropd, SpatialPadd
    from monai.transforms.intensity.dictionary import (
        NormalizeIntensityd,
        ScaleIntensityd,
    )
    from monai.transforms.io.dictionary import LoadImaged
    from monai.transforms.utility.dictionary import (
        ConvertToMultiChannelBasedOnBratsClassesd,
        EnsureChannelFirstd,
    )

    if normalize_mode not in ("minmax", "zscore"):
        raise ValueError(f"normalize_mode must be 'minmax' or 'zscore', got '{normalize_mode}'")

    if task not in ("reconstruction", "segmentation"):
        raise ValueError(f"task must be 'reconstruction' or 'segmentation', got '{task}'")

    load_keys = ["image", "label"] if task == "segmentation" else ["image"]
    crop_keys = ["image"] if task == "reconstruction" else ["image", "label"]

    transforms = [
        LoadImaged(keys=load_keys, image_only=True),
        EnsureChannelFirstd(keys="image"),
    ]

    if task == "segmentation":
        transforms.append(ConvertToMultiChannelBasedOnBratsClassesd(keys="label"))

    if normalize_mode == "zscore":
        transforms.append(NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
    else:
        # Scale whatever input range to [-1.0, 1.0]
        transforms.append(
            ScaleIntensityd(
                keys="image",
                minv=-1.0,
                maxv=1.0,
            )
        )

    transforms.append(SpatialPadd(keys=crop_keys, spatial_size=crop_size, mode="constant"))
    transforms.append(
        RandSpatialCropd(
            keys=crop_keys,
            roi_size=crop_size,
            random_center=True,
            random_size=False,
        )
    )

    aug_transforms = create_nnunet_augmentation_transforms(crop_keys, augmentation_config)
    transforms.extend(aug_transforms.transforms)

    return Compose(transforms)
