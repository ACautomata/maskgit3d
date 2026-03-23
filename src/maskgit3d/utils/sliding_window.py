"""Sliding window utilities for 3D medical image inference."""

from typing import Any

import torch
import torch.nn.functional as F
from monai.inferers.inferer import SlidingWindowInferer


def create_sliding_window_inferer(cfg: dict[str, Any]) -> SlidingWindowInferer | None:
    """Create a MONAI SlidingWindowInferer from configuration.

    Args:
        cfg: Configuration dictionary with keys:
            - enabled: Whether sliding window is enabled
            - roi_size: Tuple or list of 3 ints for window size (default: [64, 64, 64])
            - overlap: Overlap ratio between windows (default: 0.25)
            - mode: Inference mode, "gaussian" or "constant" (default: "gaussian")
            - sigma_scale: Scale factor for Gaussian sigma (default: 0.125)
            - sw_batch_size: Batch size for sliding window (default: 1)
            - sw_device: Device for window processing, None = input tensor's device (default: None)
            - device: Device for output aggregation (default: "cpu")

    Returns:
        SlidingWindowInferer instance if enabled, None otherwise.
    """
    if not cfg.get("enabled", False):
        return None

    roi_size = tuple(cfg.get("roi_size", [64, 64, 64]))
    overlap = cfg.get("overlap", 0.25)
    mode = cfg.get("mode", "gaussian")
    sigma_scale = cfg.get("sigma_scale", 0.125)
    sw_batch_size = cfg.get("sw_batch_size", 1)
    sw_device = cfg.get("sw_device")
    device = cfg.get("device")

    return SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode=mode,
        sigma_scale=sigma_scale,
        padding_mode="constant",
        cval=0.0,
        sw_device=sw_device,
        device=device,
    )


def pad_to_divisible(x: torch.Tensor, k: int, pad_value: float = -1.0) -> torch.Tensor:
    """Pad 5D tensor (B, C, D, H, W) to be divisible by k in all spatial dimensions.

    Uses symmetric padding with the specified pad value. Padding is distributed
    evenly between the start and end of each dimension.

    Args:
        x: Input tensor of shape (B, C, D, H, W).
        k: Divisibility factor (e.g., 16 for VQVAE with 4 downsampling layers).
        pad_value: Value to use for padding (default: -1.0).

    Returns:
        Padded tensor with spatial dimensions divisible by k.
    """
    B, C, D, H, W = x.shape

    d_new = ((D + k - 1) // k) * k
    h_new = ((H + k - 1) // k) * k
    w_new = ((W + k - 1) // k) * k

    pad_d = d_new - D
    pad_h = h_new - H
    pad_w = w_new - W

    # PyTorch pad order: (W_start, W_end, H_start, H_end, D_start, D_end)
    pad = (
        pad_w // 2,
        pad_w - pad_w // 2,
        pad_h // 2,
        pad_h - pad_h // 2,
        pad_d // 2,
        pad_d - pad_d // 2,
    )

    return F.pad(x, pad, mode="constant", value=pad_value)
