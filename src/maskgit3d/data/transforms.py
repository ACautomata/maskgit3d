"""Shared preprocessing transforms for 3D medical imaging data.

This module provides general-purpose MONAI-based preprocessing pipelines.
Dataset-specific transforms have been moved to their respective modules:
- BraTS: data/brats/transforms.py
- MedMNIST: data/medmnist/transforms.py
- VQVAE: tasks/vqvae_task.py
"""

from monai.transforms.compose import Compose
from monai.transforms.intensity.array import NormalizeIntensity, ScaleIntensity, ScaleIntensityRange
from monai.transforms.spatial.array import Resize
from monai.transforms.utility.array import EnsureChannelFirst, EnsureType


def create_3d_preprocessing(
    spatial_size: tuple[int, int, int] = (64, 64, 64),
    normalize_mode: str = "minmax",
    output_range: tuple[float, float] = (-1.0, 1.0),
) -> Compose:
    """Create a standard 3D preprocessing pipeline.

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
        transforms.extend(
            [
                ScaleIntensity(),
                ScaleIntensityRange(
                    a_min=0.0,
                    a_max=1.0,
                    b_min=output_range[0],
                    b_max=output_range[1],
                    clip=True,
                ),
            ]
        )
    else:
        transforms.append(NormalizeIntensity())

    transforms.append(Resize(spatial_size=spatial_size, mode="trilinear"))

    return Compose(transforms)


def normalize_to_neg_one_one(x):
    """Normalize tensor from [0, 1] range to [-1, 1] range.

    Args:
        x: Input tensor assumed to be in [0, 1] range

    Returns:
        Tensor normalized to [-1, 1] range
    """
    return x * 2.0 - 1.0
