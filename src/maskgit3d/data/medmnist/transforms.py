"""MONAI transforms for MedMNIST-3D."""

from typing import Callable

from monai.transforms import Compose, EnsureType, ScaleIntensityRange, SpatialPad, RandSpatialCrop

from .config import MedMNISTConfig
from .validators import validate_crop_size_for_vqvae


def create_training_transforms(config: MedMNISTConfig) -> Callable:
    """Create training transforms pipeline.

    Pipeline:
    1. EnsureType - Ensure tensor type
    2. ScaleIntensityRange - Normalize [0,255] to [-1,1]
    3. SpatialPad - Pad to crop_size if smaller
    4. RandSpatialCrop - Random crop to crop_size

    Args:
        config: MedMNIST configuration

    Returns:
        Composed transform callable
    """
    crop_size = config.crop_size

    # Validate crop_size compatibility
    validate_crop_size_for_vqvae(crop_size)

    return Compose(
        [
            EnsureType(),
            ScaleIntensityRange(
                a_min=0.0,
                a_max=255.0,
                b_min=-1.0,
                b_max=1.0,
            ),
            SpatialPad(spatial_size=crop_size, mode="constant"),
            RandSpatialCrop(
                roi_size=crop_size,
                random_center=True,
                random_size=False,
            ),
        ]
    )


def create_inference_transforms(config: MedMNISTConfig) -> Callable:
    """Create inference transforms pipeline.

    Pipeline:
    1. EnsureType - Ensure tensor type
    2. ScaleIntensityRange - Normalize [0,255] to [-1,1]

    Note: No cropping is applied during inference to preserve
    original dimensions for sliding window inference.

    Args:
        config: MedMNIST configuration

    Returns:
        Composed transform callable
    """
    return Compose(
        [
            EnsureType(),
            ScaleIntensityRange(
                a_min=0.0,
                a_max=255.0,
                b_min=-1.0,
                b_max=1.0,
            ),
        ]
    )
