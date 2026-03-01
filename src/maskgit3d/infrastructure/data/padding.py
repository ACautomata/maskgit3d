"""
Padding utilities for VQVAE-compatible preprocessing.

This module provides padding utilities to ensure that image dimensions
are compatible with VQVAE's downsampling factor and MONAI's sliding
window inference requirements.

Key requirements:
1. VQVAE encoder downsamples by a factor determined by the number of
   downsampling layers (typically 16 for 4 layers).
2. MONAI sliding window inference requires `overlap * roi_size * zoom_scale`
   to be an integer for proper reconstruction.
3. For VQVAE reconstruction, input dimensions must be divisible by the
   downsampling factor.
"""



def compute_downsampling_factor(
    channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4),
) -> int:
    """
    Compute the downsampling factor from VQVAE channel multipliers.

    The downsampling factor is 2^n where n is the number of downsampling
    layers (len(channel_multipliers) - 1).

    Args:
        channel_multipliers: Channel multipliers for encoder/decoder.
            Default is (1, 1, 2, 2, 4) which gives 4 downsampling layers.

    Returns:
        Downsampling factor (e.g., 16 for 4 downsampling layers)

    Example:
        >>> compute_downsampling_factor((1, 1, 2, 2, 4))
        16
        >>> compute_downsampling_factor((1, 2, 4))
        4
    """
    num_down_layers = len(channel_multipliers) - 1
    return 2**num_down_layers


def validate_crop_size(
    crop_size: tuple[int, int, int],
    downsampling_factor: int = 16,
) -> tuple[int, int, int]:
    """
    Validate and adjust crop size to be divisible by downsampling factor.

    Args:
        crop_size: Original crop size (D, H, W)
        downsampling_factor: VQVAE downsampling factor (default 16)

    Returns:
        Validated crop size divisible by downsampling factor

    Raises:
        ValueError: If crop size would need significant adjustment

    Example:
        >>> validate_crop_size((128, 128, 128), 16)
        (128, 128, 128)
        >>> validate_crop_size((64, 64, 64), 16)
        (64, 64, 64)
        >>> validate_crop_size((100, 100, 100), 16)  # Would round to 96 or 112
        ValueError: Crop size (100, 100, 100) is not divisible by 16
    """
    for dim, val in zip(("D", "H", "W"), crop_size, strict=True):
        if val % downsampling_factor != 0:
            raise ValueError(
                f"Crop size {crop_size} has {dim}={val} not divisible by {downsampling_factor}. "
                f"Use a size divisible by {downsampling_factor}."
            )

    return crop_size


def validate_roi_size(
    roi_size: tuple[int, int, int],
    overlap: float,
    downsampling_factor: int = 16,
) -> tuple[int, int, int]:
    """
    Validate and adjust ROI size for sliding window inference.

    Ensures:
    1. ROI size is divisible by downsampling factor (for VQVAE)
    2. overlap * roi_size is an integer (for MONAI sliding window)

    Args:
        roi_size: ROI size (D, H, W) for sliding window
        overlap: Overlap ratio (0-1), default is 0.25
        downsampling_factor: VQVAE downsampling factor (default 16)

    Returns:
        Validated ROI size

    Raises:
        ValueError: If ROI size requirements cannot be met

    Example:
        >>> validate_roi_size((128, 128, 128), 0.25, 16)
        (128, 128, 128)
        >>> validate_roi_size((64, 64, 64), 0.25, 16)
        (64, 64, 64)
    """
    if not (0 <= overlap < 1):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    for dim, val in zip(("D", "H", "W"), roi_size, strict=True):
        if val <= 0:
            raise ValueError(f"ROI size has {dim}={val}, must be positive")
        if val % downsampling_factor != 0:
            raise ValueError(
                f"ROI size {roi_size} has {dim}={val} not divisible by {downsampling_factor}. "
                f"Use a size divisible by {downsampling_factor}."
            )

    for dim, val in zip(("D", "H", "W"), roi_size, strict=True):
        product = overlap * val
        if abs(product - round(product)) > 1e-6:
            raise ValueError(
                f"overlap * roi_size must be integer for proper sliding window reconstruction. "
                f"Got overlap={overlap}, {dim}={val}, product={product}. "
                f"Adjust overlap or roi_size."
            )

    return roi_size


def compute_padded_size(
    input_size: tuple[int, int, int],
    downsampling_factor: int = 16,
) -> tuple[int, int, int]:
    """
    Compute the minimum padded size divisible by downsampling factor.

    Args:
        input_size: Input spatial size (D, H, W)
        downsampling_factor: VQVAE downsampling factor (default 16)

    Returns:
        Padded size where each dimension is divisible by downsampling factor

    Example:
        >>> compute_padded_size((100, 100, 100), 16)
        (112, 112, 112)
        >>> compute_padded_size((64, 64, 64), 16)
        (64, 64, 64)
        >>> compute_padded_size((17, 31, 47), 16)
        (32, 32, 48)
    """
    d, h, w = input_size

    def ceil_div(x: int, divisor: int) -> int:
        return (x + divisor - 1) // divisor * divisor

    return (
        ceil_div(d, downsampling_factor),
        ceil_div(h, downsampling_factor),
        ceil_div(w, downsampling_factor),
    )


def compute_output_crop(
    original_size: tuple[int, int, int],
    padded_size: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    """
    Compute slicing to crop output back to original size.

    Assumes symmetric padding (pad both sides equally, extra on end).

    Args:
        original_size: Original input size (D, H, W)
        padded_size: Padded size (D, H, W)

    Returns:
        Tuple of slices for cropping along (D, H, W) dimensions

    Raises:
        ValueError: If original_size > padded_size in any dimension

    Example:
        >>> compute_output_crop((100, 100, 100), (112, 112, 112))
        (slice(6, 106, None), slice(6, 106, None), slice(6, 106, None))
    """
    for dim, orig, padded in zip(("D", "H", "W"), original_size, padded_size, strict=True):
        if orig > padded:
            raise ValueError(
                f"original_size ({original_size}) must be <= padded_size ({padded_size}), "
                f"but {dim} has {orig} > {padded}"
            )

    slices = []
    for orig, padded in zip(original_size, padded_size, strict=True):
        pad_total = padded - orig
        pad_start = pad_total // 2
        pad_end = pad_start + orig
        slices.append(slice(pad_start, pad_end))

    return tuple(slices)  # type: ignore

