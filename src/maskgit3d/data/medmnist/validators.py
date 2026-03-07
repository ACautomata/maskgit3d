"""Validators for MedMNIST data configuration."""

import warnings
from typing import Tuple

VQVAE_DOWNSAMPLING_FACTOR = 16


def validate_crop_size_for_vqvae(
    crop_size: Tuple[int, int, int],
    raise_error: bool = False,
) -> bool:
    """Validate that crop_size is compatible with VQVAE downsampling.

    VQVAE has 4 downsampling layers (2^4=16), so crop_size must be
    divisible by 16 to avoid dimension mismatch during encode/decode.

    Args:
        crop_size: Spatial dimensions (D, H, W)
        raise_error: If True, raise ValueError on invalid size.
                    If False, emit warning and return False.

    Returns:
        True if valid, False if invalid (and raise_error=False)

    Raises:
        ValueError: If invalid and raise_error=True
    """
    for dim, size in enumerate(crop_size):
        if size % VQVAE_DOWNSAMPLING_FACTOR != 0:
            suggested = [s - s % VQVAE_DOWNSAMPLING_FACTOR for s in crop_size]
            msg = (
                f"crop_size[{dim}]={size} is not divisible by "
                f"{VQVAE_DOWNSAMPLING_FACTOR}. This may cause VQVAE "
                f"encode/decode dimension mismatch. Consider using "
                f"{tuple(suggested)} instead."
            )
            if raise_error:
                raise ValueError(msg)
            warnings.warn(msg, UserWarning)
            return False
    return True


def validate_roi_size_for_vqvae(
    roi_size: Tuple[int, int, int],
    raise_error: bool = False,
) -> bool:
    """Validate that roi_size is compatible with VQVAE downsampling.

    See validate_crop_size_for_vqvae for details.
    """
    return validate_crop_size_for_vqvae(roi_size, raise_error)
