"""MONAI transforms for MedMNIST-3D."""

from collections.abc import Callable

from monai.transforms.compose import Compose
from monai.transforms.croppad.array import RandSpatialCrop, SpatialPad
from monai.transforms.intensity.array import ScaleIntensity, ScaleIntensityRange
from monai.transforms.spatial.array import RandFlip, Resize
from monai.transforms.utility.array import EnsureType

from .config import MedMNISTConfig
from .validators import validate_crop_size_for_vqvae


def create_training_transforms(config: MedMNISTConfig) -> Callable:
    """Create training transforms pipeline.

    Pipeline:
    1. EnsureType - Ensure tensor type
    2. SpatialPad - Pad to crop_size if smaller
    3. ScaleIntensityRange - Normalize [0,255] to [-1,1]
    4. RandFlip - Random flip along each spatial axis (prob=0.5)
    5. RandSpatialCrop - Random crop to crop_size

    Args:
        config: MedMNIST configuration

    Returns:
        Composed transform callable
    """
    crop_size = config.crop_size

    validate_crop_size_for_vqvae(crop_size)

    return Compose(
        [
            EnsureType(),
            SpatialPad(spatial_size=crop_size, mode="constant"),
            ScaleIntensityRange(
                a_min=0.0,
                a_max=255.0,
                b_min=-1.0,
                b_max=1.0,
            ),
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=2),
            RandSpatialCrop(
                roi_size=crop_size,
                random_center=True,
                random_size=False,
            ),
        ]
    )


def create_validation_transforms(config: MedMNISTConfig) -> Callable:
    """Create validation transforms pipeline.

    No spatial cropping - preserves original input size for sliding window inference.
    Use this when validation should operate on full-size images with sliding window.

    Pipeline:
    1. EnsureType - Ensure tensor type
    2. ScaleIntensityRange - Normalize [0,255] to [-1,1]

    Note:
        Unlike training transforms, this does NOT crop the data.
        The VQVAE reconstructor handles sliding window inference for large inputs.

    Args:
        config: MedMNIST configuration

    Returns:
        Composed transform callable
    """
    crop_size = config.crop_size

    return Compose(
        [
            EnsureType(),
            SpatialPad(spatial_size=crop_size, mode="constant"),
            ScaleIntensityRange(
                a_min=0.0,
                a_max=255.0,
                b_min=-1.0,
                b_max=1.0,
            ),
        ]
    )


def create_inference_transforms(config: MedMNISTConfig) -> Callable:
    """Create inference/test transforms pipeline.

    No spatial cropping - preserves original input size for test/inference.
    Use this for test set evaluation where full-volume reconstruction is needed.

    Pipeline:
    1. EnsureType - Ensure tensor type
    2. SpatialPad - Pad to crop_size if smaller (for divisibility)
    3. ScaleIntensityRange - Normalize [0,255] to [-1,1]

    Args:
        config: MedMNIST configuration

    Returns:
        Composed transform callable
    """
    crop_size = config.crop_size

    return Compose(
        [
            EnsureType(),
            SpatialPad(spatial_size=crop_size, mode="constant"),
            ScaleIntensityRange(
                a_min=0.0,
                a_max=255.0,
                b_min=-1.0,
                b_max=1.0,
            ),
        ]
    )


def create_medmnist_preprocessing(
    spatial_size: tuple[int, int, int] = (64, 64, 64),
    input_size: int = 28,
) -> Compose:
    """Create preprocessing pipeline for MedMNIST3D data.

    MedMNIST3D data comes as numpy arrays with values in [0, 255].

    Args:
        spatial_size: Target spatial dimensions (D, H, W)
        input_size: Input size (28 or 64 for MedMNIST3D)

    Returns:
        MONAI Compose object with MedMNIST3D-specific preprocessing
    """
    transforms = [
        EnsureType(),
        ScaleIntensity(minv=-1.0, maxv=1.0),
        Resize(spatial_size=spatial_size, mode="trilinear")
        if spatial_size != (input_size, input_size, input_size)
        else lambda x: x,
    ]

    return Compose([t for t in transforms if callable(t)])


def create_medmnist_training_preprocessing(
    crop_size: tuple[int, int, int] = (32, 32, 32),
    input_size: int = 28,
) -> Compose:
    """Create training preprocessing pipeline for MedMNIST3D data with random crop.

    Args:
        crop_size: Crop spatial dimensions (D, H, W) for training
        input_size: Input size (28 or 64 for MedMNIST3D)

    Returns:
        MONAI Compose object with MedMNIST3D training preprocessing
    """
    import warnings

    downsampling_factor = 16
    for dim in crop_size:
        if dim % downsampling_factor != 0:
            warnings.warn(
                f"crop_size {crop_size} is not divisible by {downsampling_factor}. "
                "This may cause issues with VQVAE decoder.",
                UserWarning,
                stacklevel=2,
            )
            break

    transforms = [
        EnsureType(),
        ScaleIntensity(minv=-1.0, maxv=1.0),
        SpatialPad(spatial_size=crop_size, mode="constant"),
        RandSpatialCrop(roi_size=crop_size, random_center=True, random_size=False),
    ]
    return Compose(transforms)


def create_medmnist_inference_preprocessing() -> Compose:
    """Create inference preprocessing pipeline for MedMNIST3D data.

    Returns:
        MONAI Compose object with MedMNIST3D inference preprocessing
    """
    transforms = [
        EnsureType(),
        ScaleIntensity(minv=-1.0, maxv=1.0),
    ]
    return Compose(transforms)
