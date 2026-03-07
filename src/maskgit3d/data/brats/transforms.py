"""BraTS dataset preprocessing transforms."""

from monai.transforms.compose import Compose
from monai.transforms.croppad.array import RandSpatialCrop, SpatialPad
from monai.transforms.croppad.dictionary import RandSpatialCropd, SpatialPadd
from monai.transforms.intensity.array import NormalizeIntensity, ScaleIntensity
from monai.transforms.intensity.dictionary import NormalizeIntensityd, ScaleIntensityd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.array import Resize
from monai.transforms.spatial.dictionary import Resized
from monai.transforms.utility.array import EnsureChannelFirst, EnsureType
from monai.transforms.utility.dictionary import (
    ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureChannelFirstd,
)


def create_brats_preprocessing(
    spatial_size: tuple[int, int, int] = (64, 64, 64),
    normalize_mode: str = "zscore",
) -> Compose:
    """Create preprocessing pipeline for BraTS MRI data.

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
        NormalizeIntensity() if normalize_mode == "zscore" else ScaleIntensity(minv=-1.0, maxv=1.0),
        Resize(spatial_size=spatial_size, mode="trilinear"),
    ]

    return Compose(transforms)


def create_brats_training_preprocessing(
    crop_size: tuple[int, int, int] = (128, 128, 128),
    normalize_mode: str = "zscore",
) -> Compose:
    """Create training preprocessing pipeline for BraTS MRI data with random crop.

    Args:
        crop_size: Crop spatial dimensions (D, H, W) for training
        normalize_mode: Normalization mode ("minmax" or "zscore")

    Returns:
        MONAI Compose object with BraTS training preprocessing
    """
    transforms = [
        EnsureType(),
        EnsureChannelFirst(channel_dim="no_channel"),
        NormalizeIntensity() if normalize_mode == "zscore" else ScaleIntensity(minv=-1.0, maxv=1.0),
        SpatialPad(spatial_size=crop_size, mode="constant"),
        RandSpatialCrop(roi_size=crop_size, random_center=True, random_size=False),
    ]
    return Compose(transforms)


def create_brats_inference_preprocessing(
    normalize_mode: str = "zscore",
) -> Compose:
    """Create inference preprocessing pipeline for BraTS MRI data.

    Args:
        normalize_mode: Normalization mode ("minmax" or "zscore")

    Returns:
        MONAI Compose object with BraTS inference preprocessing
    """
    transforms = [
        EnsureType(),
        EnsureChannelFirst(channel_dim="no_channel"),
        NormalizeIntensity() if normalize_mode == "zscore" else ScaleIntensity(minv=-1.0, maxv=1.0),
    ]
    return Compose(transforms)


def create_brats2023_preprocessing(
    spatial_size: tuple[int, int, int] = (64, 64, 64),
    normalize_mode: str = "zscore",
    task: str = "reconstruction",
) -> Compose:
    """Create dictionary-based preprocessing pipeline for BraTS 2023 data.

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


def create_brats2023_training_preprocessing(
    crop_size: tuple[int, int, int] = (128, 128, 128),
    normalize_mode: str = "zscore",
    task: str = "reconstruction",
) -> Compose:
    """Create training preprocessing pipeline for BraTS 2023 data with random crop.

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

    crop_keys = ["image"] if task == "reconstruction" else ["image", "label"]
    transforms.extend(
        [
            SpatialPadd(keys=crop_keys, spatial_size=crop_size, mode="constant"),
            RandSpatialCropd(
                keys=crop_keys, roi_size=crop_size, random_center=True, random_size=False
            ),
        ]
    )

    return Compose(transforms)


def create_brats2023_inference_preprocessing(
    normalize_mode: str = "zscore",
    task: str = "reconstruction",
) -> Compose:
    """Create inference preprocessing pipeline for BraTS 2023 data.

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
