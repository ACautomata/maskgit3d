"""Utilities for computing MAISI tensor splitting parameters.

This module provides functions to calculate the maximum supported `num_splits`
based on spatial dimensions, downsampling factors, and convolution kernel sizes.

Key constraints from MONAI MaisiConvolution:
- Split size = spatial_dim // num_splits (floor division)
- Each split chunk must be >= kernel_size + padding
- Default kernel_size = 3, padding = 3 (for boundary handling)
- So minimum split size = 6 voxels
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal


def compute_downsampling_factor(num_channels: tuple[int, ...] | list[int]) -> int:
    """Compute the total spatial downsampling factor from encoder channel config.

    The MAISI/VQVAE encoder applies downsampling at each level.
    Each level with a downsampling operation reduces spatial dimensions by 2x.
    Number of downsampling stages = len(num_channels) - 1

    Args:
        num_channels: Channel configuration tuple, e.g., (64, 128, 256)

    Returns:
        Total downsampling factor (power of 2), e.g., 4 for 3 levels

    Example:
        >>> compute_downsampling_factor((64, 128, 256))
        4  # 2^(3-1) = 4
    """
    if len(num_channels) < 1:
        raise ValueError("num_channels must have at least one element")
    # Each transition between levels applies 2x downsampling
    num_downsamplings = len(num_channels) - 1
    return int(2**num_downsamplings)


def compute_max_num_splits(
    spatial_size: int,
    kernel_size: int = 3,
    split_padding: int = 3,
) -> int:
    """Compute maximum supported num_splits for a single spatial dimension.

    The constraint is that each split chunk must be large enough to handle
    the convolution with boundary padding:
    - split_size = spatial_size // num_splits
    - split_size must be >= kernel_size + split_padding

    Args:
        spatial_size: Spatial dimension size (D, H, or W)
        kernel_size: Convolution kernel size (default: 3)
        split_padding: Padding added for split boundary handling (default: 3)

    Returns:
        Maximum supported num_splits (at least 1)

    Example:
        >>> compute_max_num_splits(32, kernel_size=3, split_padding=3)
        5  # 32 // 6 = 5 (each split ~= 6.4 voxels)
    """
    min_split_size = kernel_size + split_padding
    if spatial_size < min_split_size:
        return 1
    return max(1, spatial_size // min_split_size)


def compute_max_num_splits_for_spatial_dims(
    spatial_dims: tuple[int, int, int],
    kernel_size: int = 3,
    split_padding: int = 3,
    dim_split: int = 0,
) -> int:
    """Compute maximum supported num_splits for the specified split dimension.

    Args:
        spatial_dims: Spatial dimensions (D, H, W)
        kernel_size: Convolution kernel size (default: 3)
        split_padding: Padding for split boundary handling (default: 3)
        dim_split: Dimension along which to split (0=D, 1=H, 2=W)

    Returns:
        Maximum supported num_splits for the specified dimension

    Raises:
        ValueError: If dim_split is not 0, 1, or 2
    """
    if dim_split not in (0, 1, 2):
        raise ValueError(f"dim_split must be 0, 1, or 2, got {dim_split}")

    target_dim = spatial_dims[dim_split]
    return compute_max_num_splits(target_dim, kernel_size, split_padding)


def resolve_num_splits(
    crop_size: tuple[int, int, int] | None,
    roi_size: tuple[int, int, int] | None,
    num_channels: Sequence[int],
    kernel_size: int = 3,
    split_padding: int = 3,
    dim_split: int = 0,
    requested_num_splits: int | None = None,
    prefer_crop_size: bool = True,
) -> tuple[int, str]:
    """Resolve the optimal num_splits value based on spatial dimensions.

    Computes the maximum supported num_splits considering both crop_size
    (training) and roi_size (inference). The constraint is that after
    all downsampling stages, the minimum spatial dimension must still be
    large enough to support splitting.

    IMPORTANT: num_splits is applied at ALL encoder layers, including those
    with reduced spatial dimensions after downsampling. Therefore, we must
    ensure the minimum dimension (after downsampling) can support splitting.

    Args:
        crop_size: Training crop size (D, H, W), optional
        roi_size: Inference ROI size (D, H, W), optional
        num_channels: Encoder channel configuration for computing downsampling
        kernel_size: Convolution kernel size (default: 3)
        split_padding: Padding for split boundary handling (default: 3)
        dim_split: Dimension along which to split (0=D, 1=H, 2=W)
        requested_num_splits: User-requested value, None for auto
        prefer_crop_size: If True, use crop_size for auto-calculation when both provided

    Returns:
        Tuple of (resolved_num_splits, reason_string)

    Raises:
        ValueError: If requested_num_splits exceeds the computed maximum

    Example:
        >>> resolve_num_splits(
        ...     crop_size=(32, 32, 32),
        ...     roi_size=(64, 64, 64),
        ...     num_channels=(64, 128, 256),
        ...     dim_split=0
        ... )
        (1, 'auto: min latent size too small for splitting')
    """

    num_downsamplings = len(num_channels) - 1 if num_channels else 0

    def is_valid_num_splits(dim_size: int, num_s: int) -> bool:
        for stage in range(num_downsamplings + 1):
            stage_dim = dim_size // (2**stage)
            if stage_dim % num_s != 0:
                return False
            split_size = stage_dim // num_s
            if split_size < kernel_size:
                return False
        return True

    def compute_max_for_size(spatial_size: tuple[int, int, int]) -> int:
        downsampling_factor = compute_downsampling_factor(list(num_channels))
        dim_size = spatial_size[dim_split]
        latent_dim = dim_size // downsampling_factor

        if latent_dim < kernel_size:
            return 1

        theoretical_max = latent_dim // kernel_size

        for num_s in range(theoretical_max, 0, -1):
            if num_s == 1:
                return 1
            if is_valid_num_splits(dim_size, num_s):
                return num_s

        return 1

    if crop_size is not None and roi_size is not None:
        max_from_crop = compute_max_for_size(crop_size)
        max_from_roi = compute_max_for_size(roi_size)
        auto_max = min(max_from_crop, max_from_roi)
        reason = (
            f"auto: min of crop_size[{dim_split}]={crop_size[dim_split]}->{max_from_crop}, "
            f"roi_size[{dim_split}]={roi_size[dim_split]}->{max_from_roi}"
        )
    elif crop_size is not None:
        max_from_crop = compute_max_for_size(crop_size)
        auto_max = max_from_crop
        reason = f"auto: from crop_size[{dim_split}]={crop_size[dim_split]}->{max_from_crop}"
    elif roi_size is not None:
        max_from_roi = compute_max_for_size(roi_size)
        auto_max = max_from_roi
        reason = f"auto: from roi_size[{dim_split}]={roi_size[dim_split]}->{max_from_roi}"
    else:
        return 1, "default: no spatial size provided"

    if requested_num_splits is not None:
        if requested_num_splits < 1:
            raise ValueError(f"num_splits must be >= 1, got {requested_num_splits}")

        if crop_size is not None and not is_valid_num_splits(
            crop_size[dim_split], requested_num_splits
        ):
            raise ValueError(
                f"Requested num_splits={requested_num_splits} is not valid for "
                f"crop_size[{dim_split}]={crop_size[dim_split]} with "
                f"num_channels={list(num_channels)}. "
                f"Each encoder stage must be divisible by num_splits and "
                f"split_size must be >= kernel_size={kernel_size}."
            )

        if roi_size is not None and not is_valid_num_splits(
            roi_size[dim_split], requested_num_splits
        ):
            raise ValueError(
                f"Requested num_splits={requested_num_splits} is not valid for "
                f"roi_size[{dim_split}]={roi_size[dim_split]} with "
                f"num_channels={list(num_channels)}. "
                f"Each encoder stage must be divisible by num_splits and "
                f"split_size must be >= kernel_size={kernel_size}."
            )

        return (
            requested_num_splits,
            f"explicit: user requested {requested_num_splits} (max allowed: {auto_max})",
        )

    return auto_max, reason


def validate_num_splits_for_all_dims(
    spatial_dims: tuple[int, int, int],
    num_splits: int,
    kernel_size: int = 3,
    split_padding: int = 3,
) -> dict[Literal["D", "H", "W"], bool]:
    """Validate num_splits against all spatial dimensions.

    Returns a dict indicating whether num_splits is valid for each dimension.
    Useful for helping users choose the appropriate dim_split.

    Args:
        spatial_dims: Spatial dimensions (D, H, W)
        num_splits: Number of splits to validate
        kernel_size: Convolution kernel size
        split_padding: Padding for split boundary handling

    Returns:
        Dict mapping dimension name to validity boolean

    Example:
        >>> validate_num_splits_for_all_dims((32, 64, 64), num_splits=10)
        {'D': False, 'H': True, 'W': True}
    """
    dim_names: list[Literal["D", "H", "W"]] = ["D", "H", "W"]
    result = {}
    for i, name in enumerate(dim_names):
        max_splits = compute_max_num_splits(spatial_dims[i], kernel_size, split_padding)
        result[name] = num_splits <= max_splits
    return result
