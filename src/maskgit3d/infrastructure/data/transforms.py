"""
Shared preprocessing transforms for 3D medical imaging data.

This module provides common MONAI-based preprocessing pipelines for
various medical imaging datasets (BraTS, MedMnist3D, etc.).
"""

from typing import Tuple

from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureType,
    NormalizeIntensity,
    Resize,
    ScaleIntensityRange,
)


def create_3d_preprocessing(
    spatial_size: Tuple[int, int, int] = (64, 64, 64),
    normalize_mode: str = "minmax",
    output_range: Tuple[float, float] = (-1.0, 1.0),
) -> Compose:
    """
    Create a standard 3D preprocessing pipeline.

    Args:
        spatial_size: Target spatial dimensions (D, H, W)
        normalize_mode: Normalization mode, either "minmax" or "zscore"
        output_range: Output intensity range for minmax normalization

    Returns:
        MONAI Compose object with preprocessing transforms

    Raises:
        ValueError: If normalize_mode is not "minmax" or "zscore"
    """
    if normalize_mode not in ("minmax", "zscore"):
        raise ValueError(f"normalize_mode must be 'minmax' or 'zscore', got '{normalize_mode}'")

    transforms = [
        EnsureType(),
        EnsureChannelFirst(channel_dim="no_channel"),
    ]

    if normalize_mode == "minmax":
        # Scale intensity to [0, 1] first, then to output range
        transforms.append(
            ScaleIntensityRange(
                a_min=None,
                a_max=None,
                b_min=output_range[0],
                b_max=output_range[1],
                clip=True,
            )
        )
    else:  # zscore
        transforms.append(NormalizeIntensity())

    # Resize to target spatial size
    transforms.append(Resize(spatial_size=spatial_size, mode="trilinear"))

    return Compose(transforms)


def normalize_to_neg_one_one(x):
    """
    Normalize tensor from [0, 1] range to [-1, 1] range.

    Args:
        x: Input tensor assumed to be in [0, 1] range

    Returns:
        Tensor normalized to [-1, 1] range
    """
    return x * 2.0 - 1.0


def create_brats_preprocessing(
    spatial_size: Tuple[int, int, int] = (64, 64, 64),
    normalize_mode: str = "zscore",
) -> Compose:
    """
    Create preprocessing pipeline for BraTS MRI data.

    BraTS data is typically stored in NIfTI format with varying
    spatial dimensions and intensity ranges.

    Args:
        spatial_size: Target spatial dimensions (D, H, W)
        normalize_mode: Normalization mode ("minmax" or "zscore")

    Returns:
        MONAI Compose object with BraTS-specific preprocessing
    """
    transforms = [
        EnsureType(),
        EnsureChannelFirst(channel_dim="no_channel"),
        # Normalize intensity (z-score is common for MRI)
        NormalizeIntensity() if normalize_mode == "zscore"
        else ScaleIntensityRange(a_min=None, a_max=None, b_min=-1.0, b_max=1.0, clip=True),
        # Resize to target size
        Resize(spatial_size=spatial_size, mode="trilinear"),
    ]

    return Compose(transforms)


def create_medmnist_preprocessing(
    spatial_size: Tuple[int, int, int] = (64, 64, 64),
    input_size: int = 28,
) -> Compose:
    """
    Create preprocessing pipeline for MedMnist3D data.

    MedMnist3D data comes as numpy arrays with shape (D, H, W) or (D, H, W, C)
    and intensity values in [0, 255].

    Args:
        spatial_size: Target spatial dimensions (D, H, W)
        input_size: Input size (28 or 64 for MedMnist3D)

    Returns:
        MONAI Compose object with MedMnist3D-specific preprocessing
    """
    transforms = [
        EnsureType(),
        EnsureChannelFirst(channel_dim="no_channel"),
        # Scale from [0, 255] to [-1, 1]
        ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
        # Resize if target size differs from input
        Resize(spatial_size=spatial_size, mode="trilinear") if spatial_size != (input_size, input_size, input_size)
        else lambda x: x,
    ]

    return Compose([t for t in transforms if callable(t)])