"""Tests for BraTS2023 transforms."""

import pytest
from monai.transforms import Compose

from maskgit3d.data.brats.transforms import (
    create_brats2023_training_transforms,
    create_brats2023_validation_transforms,
)


class TestCreateBrats2023TrainingTransforms:
    """Test training transforms composition."""

    def test_returns_compose_object(self) -> None:
        """Test that function returns MONAI Compose object."""
        transforms = create_brats2023_training_transforms()
        assert isinstance(transforms, Compose)

    def test_contains_load_transform(self) -> None:
        """Test that transforms include LoadImaged."""
        transforms = create_brats2023_training_transforms()
        transform_names = [type(t).__name__ for t in transforms.transforms]
        assert "LoadImaged" in transform_names

    def test_contains_spatial_crop(self) -> None:
        """Test that training transforms include RandSpatialCropd."""
        transforms = create_brats2023_training_transforms()
        transform_names = [type(t).__name__ for t in transforms.transforms]
        assert "RandSpatialCropd" in transform_names

    def test_contains_affine_augmentation(self) -> None:
        """Test that training transforms include RandAffined."""
        transforms = create_brats2023_training_transforms()
        transform_names = [type(t).__name__ for t in transforms.transforms]
        assert "RandAffined" in transform_names

    def test_contains_elastic_augmentation(self) -> None:
        """Test that training transforms include Rand3DElasticd."""
        transforms = create_brats2023_training_transforms()
        transform_names = [type(t).__name__ for t in transforms.transforms]
        assert "Rand3DElasticd" in transform_names

    def test_contains_gaussian_noise(self) -> None:
        """Test that training transforms include RandGaussianNoised."""
        transforms = create_brats2023_training_transforms()
        transform_names = [type(t).__name__ for t in transforms.transforms]
        assert "RandGaussianNoised" in transform_names

    def test_contains_contrast_adjustment(self) -> None:
        """Test that training transforms include RandScaleIntensityd or similar."""
        transforms = create_brats2023_training_transforms()
        transform_names = [type(t).__name__ for t in transforms.transforms]
        # Check for any intensity/contrast adjustment
        intensity_transforms = ["RandScaleIntensityd", "RandAdjustContrastd", "RandGammaTransform"]
        assert any(t in transform_names for t in intensity_transforms)

    def test_default_crop_size(self) -> None:
        """Test that default crop size is (128, 128, 128)."""
        transforms = create_brats2023_training_transforms()
        for t in transforms.transforms:
            if type(t).__name__ == "RandSpatialCropd":
                assert t.cropper.roi_size == (128, 128, 128)
                return
        pytest.fail("RandSpatialCropd not found in transforms")

    def test_custom_crop_size(self) -> None:
        """Test that custom crop size is respected."""
        transforms = create_brats2023_training_transforms(crop_size=(64, 64, 64))
        for t in transforms.transforms:
            if type(t).__name__ == "RandSpatialCropd":
                assert t.cropper.roi_size == (64, 64, 64)
                return
        pytest.fail("RandSpatialCropd not found in transforms")


class TestCreateBrats2023ValidationTransforms:
    """Test validation/test transforms composition."""

    def test_returns_compose_object(self) -> None:
        """Test that function returns MONAI Compose object."""
        transforms = create_brats2023_validation_transforms()
        assert isinstance(transforms, Compose)

    def test_contains_load_transform(self) -> None:
        """Test that transforms include LoadImaged."""
        transforms = create_brats2023_validation_transforms()
        transform_names = [type(t).__name__ for t in transforms.transforms]
        assert "LoadImaged" in transform_names

    def test_contains_normalization(self) -> None:
        """Test that transforms include normalization."""
        transforms = create_brats2023_validation_transforms()
        transform_names = [type(t).__name__ for t in transforms.transforms]
        assert "NormalizeIntensityd" in transform_names

    def test_no_spatial_crop(self) -> None:
        """Test that validation transforms do NOT include spatial crop."""
        transforms = create_brats2023_validation_transforms()
        transform_names = [type(t).__name__ for t in transforms.transforms]
        assert "RandSpatialCropd" not in transform_names
        assert "SpatialCropd" not in transform_names

    def test_no_random_augmentations(self) -> None:
        """Test that validation transforms do NOT include random augmentations."""
        transforms = create_brats2023_validation_transforms()
        transform_names = [type(t).__name__ for t in transforms.transforms]
        random_transforms = [
            "RandAffined",
            "Rand3DElasticd",
            "RandGaussianNoised",
            "RandScaleIntensityd",
            "RandAdjustContrastd",
            "RandFlipd",
            "RandRotated",
        ]
        for rand_t in random_transforms:
            assert rand_t not in transform_names, (
                f"Found random transform {rand_t} in validation pipeline"
            )

    def test_channel_wise_normalization(self) -> None:
        """Test that normalization is channel-wise."""
        transforms = create_brats2023_validation_transforms()
        for t in transforms.transforms:
            if type(t).__name__ == "NormalizeIntensityd":
                assert t.normalizer.channel_wise is True
                return
        pytest.fail("NormalizeIntensityd not found in transforms")
