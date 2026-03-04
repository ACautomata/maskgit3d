"""Tests for data/padding.py utilities."""

import pytest

from maskgit3d.infrastructure.data.padding import (
    compute_downsampling_factor,
    compute_output_crop,
    compute_padded_size,
    validate_crop_size,
    validate_roi_size,
)


class TestComputeDownsamplingFactor:
    """Test compute_downsampling_factor function."""

    @pytest.mark.parametrize(
        "channel_multipliers,expected",
        [
            ((1, 1, 2, 2, 4), 16),  # 4 downsampling layers -> 2^4 = 16
            ((1, 2, 4), 4),  # 2 downsampling layers -> 2^2 = 4
            ((1, 2), 2),  # 1 downsampling layer -> 2^1 = 2
            ((1,), 1),  # 0 downsampling layers -> 2^0 = 1
            ((1, 1, 1, 1, 1, 1), 32),  # 5 downsampling layers -> 2^5 = 32
        ],
    )
    def test_compute_downsampling_factor(self, channel_multipliers, expected):
        """Test downsampling factor calculation."""
        assert compute_downsampling_factor(channel_multipliers) == expected

    def test_default_value(self):
        """Test default channel multipliers."""
        # Default is (1, 1, 2, 2, 4) -> 16
        assert compute_downsampling_factor() == 16


class TestComputePaddedSize:
    """Test compute_padded_size function."""

    @pytest.mark.parametrize(
        "input_size,factor,expected",
        [
            ((100, 100, 100), 16, (112, 112, 112)),
            ((64, 64, 64), 16, (64, 64, 64)),
            ((17, 31, 47), 16, (32, 32, 48)),
            ((15, 15, 15), 8, (16, 16, 16)),
            ((1, 1, 1), 16, (16, 16, 16)),
        ],
    )
    def test_compute_padded_size(self, input_size, factor, expected):
        """Test padded size calculation."""
        assert compute_padded_size(input_size, factor) == expected

    def test_default_factor(self):
        """Test with default downsampling factor."""
        result = compute_padded_size((32, 32, 32))
        assert result == (32, 32, 32)


class TestValidateCropSize:
    """Test validate_crop_size function."""

    def test_valid_crop_size(self):
        """Test validation of valid crop sizes."""
        assert validate_crop_size((128, 128, 128), 16) == (128, 128, 128)
        assert validate_crop_size((64, 64, 64), 16) == (64, 64, 64)

    @pytest.mark.parametrize(
        "crop_size,factor",
        [
            ((100, 100, 100), 16),
            ((65, 64, 64), 16),
            ((64, 65, 64), 16),
            ((64, 64, 65), 16),
        ],
    )
    def test_invalid_crop_size_raises(self, crop_size, factor):
        """Test that invalid crop sizes raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_crop_size(crop_size, factor)
        assert "not divisible by" in str(exc_info.value)


class TestValidateRoiSize:
    """Test validate_roi_size function."""

    def test_valid_roi_size(self):
        """Test validation of valid ROI sizes."""
        assert validate_roi_size((128, 128, 128), 0.25, 16) == (128, 128, 128)

    def test_valid_roi_overlap_product(self):
        """Test ROI size where overlap * size is integer."""
        # 0.25 * 128 = 32 (integer)
        assert validate_roi_size((128, 128, 128), 0.25, 16) == (128, 128, 128)

    def test_roi_not_divisible_raises(self):
        """Test that ROI not divisible by factor raises."""
        with pytest.raises(ValueError) as exc_info:
            validate_roi_size((100, 100, 100), 0.25, 16)
        assert "not divisible by" in str(exc_info.value)

    def test_invalid_overlap_raises(self):
        """Test that invalid overlap values raise."""
        with pytest.raises(ValueError) as exc_info:
            validate_roi_size((128, 128, 128), 1.5, 16)
        assert "overlap must be in [0, 1)" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            validate_roi_size((128, 128, 128), -0.1, 16)
        assert "overlap must be in [0, 1)" in str(exc_info.value)


class TestComputeOutputCrop:
    """Test compute_output_crop function."""

    def test_symmetric_padding(self):
        """Test cropping after symmetric padding."""
        original = (100, 100, 100)
        padded = (112, 112, 112)
        slices = compute_output_crop(original, padded)

        # For symmetric padding: pad = (112-100)/2 = 6 on each side
        assert slices == (slice(6, 106), slice(6, 106), slice(6, 106))

    def test_no_padding_needed(self):
        """Test when no padding was applied."""
        original = (64, 64, 64)
        padded = (64, 64, 64)
        slices = compute_output_crop(original, padded)
        assert slices == (slice(0, 64), slice(0, 64), slice(0, 64))

    def test_asymmetric_padding(self):
        """Test with odd padding (extra on end)."""
        original = (100, 100, 100)
        padded = (113, 113, 113)  # 13 extra, 6 on start, 7 on end
        slices = compute_output_crop(original, padded)
        assert slices == (slice(6, 106), slice(6, 106), slice(6, 106))

    def test_original_larger_raises(self):
        """Test that original > padded raises."""
        with pytest.raises(ValueError) as exc_info:
            compute_output_crop((100, 100, 100), (64, 64, 64))
        assert "must be <= padded_size" in str(exc_info.value)
