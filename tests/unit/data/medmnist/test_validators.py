"""Tests for MedMNIST validators."""

import warnings

from maskgit3d.data.medmnist.validators import (
    validate_crop_size_for_vqvae,
)


class TestValidateCropSizeForVQVAE:
    def test_valid_sizes_pass(self):
        """Test that sizes divisible by 16 pass validation."""
        assert validate_crop_size_for_vqvae((16, 16, 16)) is True
        assert validate_crop_size_for_vqvae((32, 32, 32)) is True
        assert validate_crop_size_for_vqvae((64, 64, 64)) is True
        assert validate_crop_size_for_vqvae((16, 32, 48)) is True

    def test_invalid_sizes_warn(self):
        """Test that invalid sizes trigger warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_crop_size_for_vqvae((28, 28, 28))
            assert result is False
            assert len(w) == 1
            assert "16" in str(w[0].message)
            assert "28" in str(w[0].message)

    def test_invalid_sizes_raises_when_requested(self):
        """Test that invalid sizes raise error when raise_error=True."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            validate_crop_size_for_vqvae((28, 28, 28), raise_error=True)
        assert "16" in str(exc_info.value)
        assert "28" in str(exc_info.value)

    def test_mixed_valid_invalid(self):
        """Test that partially valid sizes warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_crop_size_for_vqvae((32, 28, 32))
            assert result is False
            assert len(w) == 1
            assert "crop_size[1]=28" in str(w[0].message)
