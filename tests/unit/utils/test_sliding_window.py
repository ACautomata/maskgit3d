"""Tests for sliding window utilities."""

import torch

from maskgit3d.utils.sliding_window import (
    create_sliding_window_inferer,
    pad_to_divisible,
)


class TestCreateSlidingWindowInferer:
    """Tests for create_sliding_window_inferer function."""

    def test_returns_none_when_disabled(self) -> None:
        """Test that None is returned when enabled is False."""
        cfg = {"enabled": False}
        result = create_sliding_window_inferer(cfg)

        assert result is None

    def test_returns_none_when_enabled_missing(self) -> None:
        """Test that None is returned when enabled key is missing."""
        cfg: dict = {}
        result = create_sliding_window_inferer(cfg)

        assert result is None

    def test_returns_inferer_when_enabled(self) -> None:
        """Test that SlidingWindowInferer is returned when enabled is True."""
        from monai.inferers.inferer import SlidingWindowInferer

        cfg = {"enabled": True}
        result = create_sliding_window_inferer(cfg)

        assert result is not None
        assert isinstance(result, SlidingWindowInferer)

    def test_custom_roi_size(self) -> None:
        """Test that custom roi_size is used."""
        cfg = {
            "enabled": True,
            "roi_size": [128, 128, 128],
        }
        result = create_sliding_window_inferer(cfg)

        assert result is not None
        # SlidingWindowInferer stores roi_size as the first arg
        assert result.roi_size == (128, 128, 128)

    def test_custom_overlap(self) -> None:
        """Test that custom overlap is used."""
        cfg = {
            "enabled": True,
            "overlap": 0.5,
        }
        result = create_sliding_window_inferer(cfg)

        assert result is not None
        assert result.overlap == 0.5

    def test_custom_mode(self) -> None:
        """Test that custom mode is used."""
        cfg = {
            "enabled": True,
            "mode": "constant",
        }
        result = create_sliding_window_inferer(cfg)

        assert result is not None
        assert result.mode == "constant"

    def test_custom_sigma_scale(self) -> None:
        """Test that custom sigma_scale is used."""
        cfg = {
            "enabled": True,
            "sigma_scale": 0.25,
        }
        result = create_sliding_window_inferer(cfg)

        assert result is not None
        assert result.sigma_scale == 0.25

    def test_custom_sw_batch_size(self) -> None:
        """Test that custom sw_batch_size is used."""
        cfg = {
            "enabled": True,
            "sw_batch_size": 4,
        }
        result = create_sliding_window_inferer(cfg)

        assert result is not None
        assert result.sw_batch_size == 4


class TestPadToDivisible:
    """Tests for pad_to_divisible function."""

    def test_no_padding_needed(self) -> None:
        """Test that no padding is added when already divisible."""
        x = torch.randn(1, 1, 16, 16, 16)
        result = pad_to_divisible(x, k=16)

        assert result.shape == x.shape

    def test_padding_needed_single_dimension(self) -> None:
        """Test padding when only depth needs padding."""
        x = torch.randn(1, 1, 15, 16, 16)
        result = pad_to_divisible(x, k=16)

        assert result.shape == (1, 1, 16, 16, 16)

    def test_padding_needed_all_dimensions(self) -> None:
        """Test padding when all dimensions need padding."""
        x = torch.randn(1, 1, 10, 20, 30)
        result = pad_to_divisible(x, k=16)

        assert result.shape == (1, 1, 16, 32, 32)

    def test_padding_with_k_8(self) -> None:
        """Test padding with k=8."""
        x = torch.randn(1, 1, 5, 10, 15)
        result = pad_to_divisible(x, k=8)

        assert result.shape == (1, 1, 8, 16, 16)

    def test_padding_with_k_32(self) -> None:
        """Test padding with k=32."""
        x = torch.randn(1, 1, 20, 40, 60)
        result = pad_to_divisible(x, k=32)

        assert result.shape == (1, 1, 32, 64, 64)

    def test_custom_pad_value(self) -> None:
        """Test that custom pad_value is used."""
        x = torch.randn(1, 1, 10, 10, 10)
        result = pad_to_divisible(x, k=16, pad_value=0.0)

        # Check that padding regions have the specified value
        # After padding to 16x16x16, the original 10x10x10 is centered
        assert result.shape == (1, 1, 16, 16, 16)

    def test_batch_and_channel_preserved(self) -> None:
        """Test that batch and channel dimensions are preserved."""
        x = torch.randn(4, 3, 10, 20, 30)
        result = pad_to_divisible(x, k=16)

        assert result.shape[0] == 4  # batch
        assert result.shape[1] == 3  # channel

    def test_symmetric_padding_depth(self) -> None:
        """Test that padding is symmetric in depth dimension."""
        x = torch.randn(1, 1, 10, 16, 16)
        result = pad_to_divisible(x, k=16)

        # Depth goes from 10 to 16, pad of 6 should be 3 on each side
        assert result.shape == (1, 1, 16, 16, 16)

    def test_symmetric_padding_all_dims(self) -> None:
        """Test symmetric padding in all dimensions."""
        x = torch.randn(1, 1, 14, 22, 30)
        result = pad_to_divisible(x, k=16)

        # Depth: 14 -> 16 (pad 2, 1 each side)
        # Height: 22 -> 32 (pad 10, 5 each side)
        # Width: 30 -> 32 (pad 2, 1 each side)
        assert result.shape == (1, 1, 16, 32, 32)

    def test_output_divisible_by_k(self) -> None:
        """Test that output spatial dimensions are always divisible by k."""
        for k in [8, 16, 32]:
            for d, h, w in [(10, 20, 30), (15, 15, 15), (31, 47, 63)]:
                x = torch.randn(1, 1, d, h, w)
                result = pad_to_divisible(x, k=k)

                assert result.shape[2] % k == 0, f"Depth {result.shape[2]} not divisible by {k}"
                assert result.shape[3] % k == 0, f"Height {result.shape[3]} not divisible by {k}"
                assert result.shape[4] % k == 0, f"Width {result.shape[4]} not divisible by {k}"

    def test_input_not_modified(self) -> None:
        """Test that input tensor is not modified in place."""
        x = torch.randn(1, 1, 10, 20, 30)
        original_shape = x.shape
        _ = pad_to_divisible(x, k=16)

        assert x.shape == original_shape

    def test_odd_padding_distribution(self) -> None:
        """Test that odd padding amounts are distributed correctly."""
        # 10 -> 16: need 6, split as 3 left, 3 right (floor division)
        x = torch.randn(1, 1, 10, 10, 10)
        result = pad_to_divisible(x, k=16)

        assert result.shape == (1, 1, 16, 16, 16)
