"""Tests for data/augmentation.py"""

import pytest
from maskgit3d.infrastructure.data.augmentation import (
    NnUNetAugmentationConfig,
    create_nnunet_augmentation_transforms,
    create_training_transforms_with_augmentation,
    create_brats_training_transforms_with_augmentation,
)


class TestNnUNetAugmentationConfig:
    """Test NnUNetAugmentationConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = NnUNetAugmentationConfig()
        assert config.p_rotation == 0.2
        assert config.p_scale == 0.2
        assert config.p_gamma == 0.3
        assert config.p_mirror == 0.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = NnUNetAugmentationConfig()
        config_dict = config.to_dict()
        assert "rotation_x" in config_dict
        assert "p_rotation" in config_dict
        assert config_dict["p_rotation"] == 0.2

    def test_default_3d_factory(self):
        """Test default_3d class method."""
        config = NnUNetAugmentationConfig.default_3d()
        assert isinstance(config, NnUNetAugmentationConfig)
        assert config.p_rotation == 0.2

    def test_light_3d_factory(self):
        """Test light_3d class method."""
        config = NnUNetAugmentationConfig.light_3d()
        assert config.p_rotation == 0.1
        assert config.p_scale == 0.1
        assert config.p_gamma == 0.15

    def test_heavy_3d_factory(self):
        """Test heavy_3d class method."""
        config = NnUNetAugmentationConfig.heavy_3d()
        assert config.p_rotation == 0.3
        assert config.p_scale == 0.3
        assert config.p_gamma == 0.4


class TestCreateAugmentationTransforms:
    """Test create_nnunet_augmentation_transforms function."""

    def test_create_with_default_config(self):
        """Test creating transforms with default config."""
        transforms = create_nnunet_augmentation_transforms(keys=["image"])
        assert transforms is not None

    def test_create_with_custom_config(self):
        """Test creating transforms with custom config."""
        config = NnUNetAugmentationConfig.light_3d()
        transforms = create_nnunet_augmentation_transforms(
            keys=["image", "label"],
            config=config,
        )
        assert transforms is not None

    def test_create_with_single_key(self):
        """Test creating transforms with single key."""
        transforms = create_nnunet_augmentation_transforms(keys=["image"])
        assert transforms is not None


class TestCreateTrainingTransforms:
    """Test create_training_transforms_with_augmentation function."""

    def test_create_training_transforms(self):
        """Test creating complete training transforms."""
        transforms = create_training_transforms_with_augmentation(
            keys=["image", "label"],
            crop_size=(64, 64, 64),
        )
        assert transforms is not None

    def test_create_with_normalization(self):
        """Test creating transforms with normalization."""
        # Note: This would require actual MONAI transforms
        transforms = create_training_transforms_with_augmentation(
            keys=["image"],
            crop_size=(32, 32, 32),
            normalization_transforms=None,
        )
        assert transforms is not None


class TestCreateBratsTrainingTransforms:
    """Test create_brats_training_transforms_with_augmentation function."""

    def test_create_brats_transforms_reconstruction(self):
        """Test creating BraTS transforms for reconstruction task."""
        transforms = create_brats_training_transforms_with_augmentation(
            crop_size=(64, 64, 64),
            normalize_mode="zscore",
            task="reconstruction",
        )
        assert transforms is not None

    def test_create_brats_transforms_segmentation(self):
        """Test creating BraTS transforms for segmentation task."""
        transforms = create_brats_training_transforms_with_augmentation(
            crop_size=(64, 64, 64),
            normalize_mode="minmax",
            task="segmentation",
        )
        assert transforms is not None

    def test_invalid_normalize_mode_raises(self):
        """Test that invalid normalize_mode raises."""
        with pytest.raises(ValueError) as exc_info:
            create_brats_training_transforms_with_augmentation(
                crop_size=(64, 64, 64),
                normalize_mode="invalid",
            )
        assert "normalize_mode must be 'minmax' or 'zscore'" in str(exc_info.value)

    def test_invalid_task_raises(self):
        """Test that invalid task raises."""
        with pytest.raises(ValueError) as exc_info:
            create_brats_training_transforms_with_augmentation(
                crop_size=(64, 64, 64),
                task="classification",
            )
        assert "task must be 'reconstruction' or 'segmentation'" in str(exc_info.value)
