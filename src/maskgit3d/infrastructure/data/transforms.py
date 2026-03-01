"""
Shared preprocessing transforms for 3D medical imaging data.

This module provides common MONAI-based preprocessing pipelines for
various medical imaging datasets (BraTS, MedMnist3D, etc.).
"""

from typing import Tuple

from monai.transforms import (
    Compose,
    ConcatItemsd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    DeleteItemsd,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    EnsureType,
    Lambdad,
    LoadImaged,
    MapTransform,
    NormalizeIntensity,
    NormalizeIntensityd,
    RandSpatialCrop,
    RandSpatialCropd,
    Resize,
    Resized,
    ScaleIntensity,
    ScaleIntensityd,
    ScaleIntensityRange,
    ScaleIntensityRanged,
    SpatialPad,
    SpatialPadd,
)

_DICT_TRANSFORM_EXTENSIONS = (
    ConcatItemsd,
    DeleteItemsd,
    Lambdad,
    MapTransform,
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
        # Scale whatever intensity range to [0, 1] then to output range
        transforms.extend([
            ScaleIntensity(),
            ScaleIntensityRange(
                a_min=0.0,
                a_max=1.0,
                b_min=output_range[0],
                b_max=output_range[1],
                clip=True,
            )
        ])
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
        NormalizeIntensity()
        if normalize_mode == "zscore"
        else ScaleIntensity(minv=-1.0, maxv=1.0),
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
        ScaleIntensity(minv=-1.0, maxv=1.0),
        # Resize if target size differs from input
        Resize(spatial_size=spatial_size, mode="trilinear")
        if spatial_size != (input_size, input_size, input_size)
        else lambda x: x,
    ]

    return Compose([t for t in transforms if callable(t)])


def create_brats2023_preprocessing(
    spatial_size: Tuple[int, int, int] = (64, 64, 64),
    normalize_mode: str = "zscore",
    task: str = "reconstruction",
) -> Compose:
    """
    Create dictionary-based preprocessing pipeline for BraTS 2023 data.

    Args:
        spatial_size: Target spatial dimensions (D, H, W)
        normalize_mode: Normalization mode ("minmax" or "zscore")
        task: Task type ("reconstruction" or "segmentation")

    Returns:
        MONAI Compose object with dictionary-based preprocessing transforms

    Raises:
        ValueError: If normalize_mode or task are invalid
    """
    if normalize_mode not in ("minmax", "zscore"):
        raise ValueError(f"normalize_mode must be 'minmax' or 'zscore', got '{normalize_mode}'")

    if task not in ("reconstruction", "segmentation"):
        raise ValueError(f"task must be 'reconstruction' or 'segmentation', got '{task}'")

    load_keys = ["image", "label"] if task == "segmentation" else ["image"]

    transforms = [
        LoadImaged(keys=load_keys, image_only=True),
        EnsureChannelFirstd(keys="image"),
    ]

    if task == "segmentation":
        transforms.append(ConvertToMultiChannelBasedOnBratsClassesd(keys="label"))

    if normalize_mode == "zscore":
        transforms.append(NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
    else:
        transforms.append(
            ScaleIntensityd(
                keys="image",
                minv=-1.0,
                maxv=1.0,
            )
        )

    if task == "segmentation":
        transforms.append(
            Resized(
                keys=["image", "label"],
                spatial_size=spatial_size,
                mode=("trilinear", "nearest"),
            )
        )
    else:
        transforms.append(Resized(keys="image", spatial_size=spatial_size, mode="trilinear"))

    return Compose(transforms)


# =============================================================================
# Training Transforms with Random Crop
# =============================================================================


def create_brats_training_preprocessing(
    crop_size: Tuple[int, int, int] = (128, 128, 128),
    normalize_mode: str = "zscore",
) -> Compose:
    """
    Create training preprocessing pipeline for BraTS MRI data with random crop.

    Args:
        crop_size: Crop spatial dimensions (D, H, W) for training
        normalize_mode: Normalization mode ("minmax" or "zscore")

    Returns:
        MONAI Compose object with BraTS training preprocessing
    """
    transforms = [
        EnsureType(),
        EnsureChannelFirst(channel_dim="no_channel"),
        NormalizeIntensity()
        if normalize_mode == "zscore"
        else ScaleIntensity(minv=-1.0, maxv=1.0),
        # Pad if image smaller than crop size, then random crop
        SpatialPad(spatial_size=crop_size, mode="constant"),
        RandSpatialCrop(roi_size=crop_size, random_center=True, random_size=False),
    ]
    return Compose(transforms)


def create_brats_inference_preprocessing(
    normalize_mode: str = "zscore",
) -> Compose:
    """
    Create inference preprocessing pipeline for BraTS MRI data.

    No resize/crop - keep original size for sliding window inference.

    Args:
        normalize_mode: Normalization mode ("minmax" or "zscore")

    Returns:
        MONAI Compose object with BraTS inference preprocessing
    """
    transforms = [
        EnsureType(),
        EnsureChannelFirst(channel_dim="no_channel"),
        NormalizeIntensity()
        if normalize_mode == "zscore"
        else ScaleIntensity(minv=-1.0, maxv=1.0),
    ]
    return Compose(transforms)


def create_medmnist_training_preprocessing(
    crop_size: Tuple[int, int, int] = (32, 32, 32),
    input_size: int = 28,
) -> Compose:
    """
    Create training preprocessing pipeline for MedMnist3D data with random crop.

    Args:
        crop_size: Crop spatial dimensions (D, H, W) for training
        input_size: Input size (28 or 64 for MedMnist3D)

    Returns:
        MONAI Compose object with MedMnist3D training preprocessing
    """
    transforms = [
        EnsureType(),
        EnsureChannelFirst(channel_dim="no_channel"),
        ScaleIntensity(minv=-1.0, maxv=1.0),
        # Pad if smaller than crop size, then random crop
        SpatialPad(spatial_size=crop_size, mode="constant"),
        RandSpatialCrop(roi_size=crop_size, random_center=True, random_size=False),
    ]
    return Compose(transforms)


def create_medmnist_inference_preprocessing() -> Compose:
    """
    Create inference preprocessing pipeline for MedMnist3D data.

    No resize/crop - keep original size for sliding window inference.

    Returns:
        MONAI Compose object with MedMnist3D inference preprocessing
    """
    transforms = [
        EnsureType(),
        EnsureChannelFirst(channel_dim="no_channel"),
        ScaleIntensity(minv=-1.0, maxv=1.0),
    ]
    return Compose(transforms)


def create_brats2023_training_preprocessing(
    crop_size: Tuple[int, int, int] = (128, 128, 128),
    normalize_mode: str = "zscore",
    task: str = "reconstruction",
) -> Compose:
    """
    Create training preprocessing pipeline for BraTS 2023 data with random crop.

    Args:
        crop_size: Crop spatial dimensions (D, H, W) for training
        normalize_mode: Normalization mode ("minmax" or "zscore")
        task: Task type ("reconstruction" or "segmentation")

    Returns:
        MONAI Compose object with dictionary-based training preprocessing
    """
    if normalize_mode not in ("minmax", "zscore"):
        raise ValueError(f"normalize_mode must be 'minmax' or 'zscore', got '{normalize_mode}'")

    if task not in ("reconstruction", "segmentation"):
        raise ValueError(f"task must be 'reconstruction' or 'segmentation', got '{task}'")

    load_keys = ["image", "label"] if task == "segmentation" else ["image"]

    transforms = [
        LoadImaged(keys=load_keys, image_only=True),
        EnsureChannelFirstd(keys="image"),
    ]

    if task == "segmentation":
        transforms.append(ConvertToMultiChannelBasedOnBratsClassesd(keys="label"))

    if normalize_mode == "zscore":
        transforms.append(NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
    else:
        transforms.append(
            ScaleIntensityd(
                keys="image",
                minv=-1.0,
                maxv=1.0,
            )
        )

    # Pad and random crop
    crop_keys = ["image"] if task == "reconstruction" else ["image", "label"]
    transforms.extend([
        SpatialPadd(keys=crop_keys, spatial_size=crop_size, mode="constant"),
        RandSpatialCropd(keys=crop_keys, roi_size=crop_size, random_center=True, random_size=False),
    ])

    return Compose(transforms)


def create_brats2023_inference_preprocessing(
    normalize_mode: str = "zscore",
    task: str = "reconstruction",
) -> Compose:
    """
    Create inference preprocessing pipeline for BraTS 2023 data.

    No resize/crop - keep original size for sliding window inference.

    Args:
        normalize_mode: Normalization mode ("minmax" or "zscore")
        task: Task type ("reconstruction" or "segmentation")

    Returns:
        MONAI Compose object with dictionary-based inference preprocessing
    """
    if normalize_mode not in ("minmax", "zscore"):
        raise ValueError(f"normalize_mode must be 'minmax' or 'zscore', got '{normalize_mode}'")

    if task not in ("reconstruction", "segmentation"):
        raise ValueError(f"task must be 'reconstruction' or 'segmentation', got '{task}'")

    load_keys = ["image", "label"] if task == "segmentation" else ["image"]

    transforms = [
        LoadImaged(keys=load_keys, image_only=True),
        EnsureChannelFirstd(keys="image"),
    ]

    if task == "segmentation":
        transforms.append(ConvertToMultiChannelBasedOnBratsClassesd(keys="label"))

    if normalize_mode == "zscore":
        transforms.append(NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
    else:
        transforms.append(
            ScaleIntensityd(
                keys="image",
                minv=-1.0,
                maxv=1.0,
            )
        )

    return Compose(transforms)
