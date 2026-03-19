"""Geometry utilities for 3D medical image processing."""


def validate_crop_size(
    crop_size: tuple[int, int, int],
    downsampling_factor: int = 16,
) -> tuple[int, int, int]:
    """Validate and adjust crop size to be divisible by downsampling factor.

    Args:
        crop_size: Original crop size (D, H, W)
        downsampling_factor: VQVAE downsampling factor (default 16)

    Returns:
        Validated crop size divisible by downsampling factor

    Raises:
        ValueError: If crop size would need significant adjustment
    """
    for dim, val in zip(("D", "H", "W"), crop_size, strict=True):
        if val % downsampling_factor != 0:
            raise ValueError(
                f"Crop size {crop_size} has {dim}={val} not divisible by {downsampling_factor}. "
                f"Use a size divisible by {downsampling_factor}."
            )

    return crop_size


def compute_padded_size(
    input_size: tuple[int, int, int],
    downsampling_factor: int = 16,
) -> tuple[int, int, int]:
    """Compute the minimum padded size divisible by downsampling factor.

    Args:
        input_size: Input spatial size (D, H, W)
        downsampling_factor: VQVAE downsampling factor (default 16)

    Returns:
        Padded size where each dimension is divisible by downsampling factor
    """
    d, h, w = input_size

    def ceil_div(x: int, divisor: int) -> int:
        return (x + divisor - 1) // divisor * divisor

    return (
        ceil_div(d, downsampling_factor),
        ceil_div(h, downsampling_factor),
        ceil_div(w, downsampling_factor),
    )
