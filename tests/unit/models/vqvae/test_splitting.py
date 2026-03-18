"""Tests for num_splits calculation utilities."""

import pytest

from maskgit3d.models.vqvae.splitting import (
    compute_downsampling_factor,
    compute_max_num_splits,
    compute_max_num_splits_for_spatial_dims,
    resolve_num_splits,
    validate_num_splits_for_all_dims,
)


class TestComputeDownsamplingFactor:
    def test_single_channel(self):
        assert compute_downsampling_factor([64]) == 1

    def test_two_levels(self):
        assert compute_downsampling_factor([32, 64]) == 2

    def test_three_levels(self):
        assert compute_downsampling_factor([64, 128, 256]) == 4

    def test_four_levels(self):
        assert compute_downsampling_factor([32, 64, 128, 256]) == 8


class TestComputeMaxNumSplits:
    def test_small_size_returns_1(self):
        assert compute_max_num_splits(5) == 1

    def test_exact_multiple(self):
        assert compute_max_num_splits(12, kernel_size=3) == 2

    def test_non_divisible(self):
        assert compute_max_num_splits(10, kernel_size=3) == 1


class TestComputeMaxNumSplitsForSpatialDims:
    def test_dim_split_0(self):
        result = compute_max_num_splits_for_spatial_dims((32, 64, 64), dim_split=0)
        assert result == 5

    def test_dim_split_1(self):
        result = compute_max_num_splits_for_spatial_dims((32, 64, 64), dim_split=1)
        assert result == 10

    def test_dim_split_2(self):
        result = compute_max_num_splits_for_spatial_dims((32, 64, 64), dim_split=2)
        assert result == 10

    def test_invalid_dim_split(self):
        with pytest.raises(ValueError, match="dim_split must be 0, 1, or 2"):
            compute_max_num_splits_for_spatial_dims((32, 32, 32), dim_split=3)


class TestResolveNumSplits:
    def test_no_spatial_size_returns_1(self):
        result, reason = resolve_num_splits(
            crop_size=None,
            roi_size=None,
            num_channels=[64, 128, 256],
        )
        assert result == 1
        assert "no spatial size" in reason

    def test_crop_size_only(self):
        result, reason = resolve_num_splits(
            crop_size=(32, 32, 32),
            roi_size=None,
            num_channels=[64, 128, 256],
        )
        assert result == 2

    def test_roi_size_only(self):
        result, reason = resolve_num_splits(
            crop_size=None,
            roi_size=(64, 64, 64),
            num_channels=[64, 128, 256],
        )
        assert result == 4

    def test_min_of_crop_and_roi(self):
        result, reason = resolve_num_splits(
            crop_size=(64, 64, 64),
            roi_size=(32, 32, 32),
            num_channels=[64, 128, 256],
        )
        assert result == 2

    def test_multi_stage_divisibility(self):
        result, _ = resolve_num_splits(
            crop_size=(44, 44, 44),
            roi_size=None,
            num_channels=[32, 64],
        )
        assert result == 2
        assert 44 % result == 0
        assert 22 % result == 0

    def test_explicit_valid_num_splits(self):
        result, reason = resolve_num_splits(
            crop_size=(64, 64, 64),
            roi_size=None,
            num_channels=[64, 128, 256],
            requested_num_splits=2,
        )
        assert result == 2
        assert "explicit" in reason

    def test_explicit_oversize_rejected(self):
        with pytest.raises(ValueError, match="not valid"):
            resolve_num_splits(
                crop_size=(32, 32, 32),
                roi_size=None,
                num_channels=[64, 128, 256],
                requested_num_splits=10,
            )

    def test_explicit_valid_for_crop_invalid_for_roi(self):
        with pytest.raises(ValueError, match="roi_size"):
            resolve_num_splits(
                crop_size=(64, 64, 64),
                roi_size=(32, 32, 32),
                num_channels=[64, 128, 256],
                requested_num_splits=4,
            )

    def test_explicit_zero_rejected(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            resolve_num_splits(
                crop_size=(32, 32, 32),
                roi_size=None,
                num_channels=[64, 128, 256],
                requested_num_splits=0,
            )


class TestValidateNumSplitsForAllDims:
    def test_uniform_dims(self):
        result = validate_num_splits_for_all_dims((32, 32, 32), num_splits=2)
        assert result == {"D": True, "H": True, "W": True}

    def test_non_uniform_dims(self):
        result = validate_num_splits_for_all_dims((16, 64, 64), num_splits=8)
        assert result == {"D": False, "H": True, "W": True}

    def test_all_valid(self):
        result = validate_num_splits_for_all_dims((64, 64, 64), num_splits=4)
        assert all(result.values())
