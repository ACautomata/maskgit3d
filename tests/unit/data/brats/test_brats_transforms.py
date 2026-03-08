"""Tests for BraTS transforms."""

import pytest
from monai.transforms.compose import Compose
from monai.transforms.croppad.array import RandSpatialCrop, SpatialPad
from monai.transforms.intensity.array import NormalizeIntensity, ScaleIntensity
from monai.transforms.spatial.array import Resize
from monai.transforms.utility.array import EnsureChannelFirst, EnsureType

from maskgit3d.data.brats.transforms import (
    create_brats2023_inference_preprocessing,
    create_brats2023_preprocessing,
    create_brats2023_training_preprocessing,
    create_brats_inference_preprocessing,
    create_brats_preprocessing,
    create_brats_training_preprocessing,
)


class TestBraTSPreprocessing:
    """Tests for basic BraTS preprocessing functions."""

    def test_create_brats_preprocessing_default(self) -> None:
        """Test default preprocessing pipeline."""
        pipeline = create_brats_preprocessing()

        assert isinstance(pipeline, Compose)
        assert len(pipeline.transforms) == 4
        assert isinstance(pipeline.transforms[0], EnsureType)
        assert isinstance(pipeline.transforms[1], EnsureChannelFirst)
        assert isinstance(pipeline.transforms[2], NormalizeIntensity)
        assert isinstance(pipeline.transforms[3], Resize)

    def test_create_brats_preprocessing_minmax(self) -> None:
        """Test preprocessing with minmax normalization."""
        pipeline = create_brats_preprocessing(normalize_mode="minmax")

        assert isinstance(pipeline.transforms[2], ScaleIntensity)

    def test_create_brats_preprocessing_custom_size(self) -> None:
        """Test preprocessing with custom spatial size."""
        pipeline = create_brats_preprocessing(spatial_size=(128, 128, 128))

        assert pipeline.transforms[3].spatial_size == (128, 128, 128)


class TestBraTSTrainingPreprocessing:
    """Tests for BraTS training preprocessing."""

    def test_create_brats_training_preprocessing_default(self) -> None:
        """Test default training preprocessing."""
        pipeline = create_brats_training_preprocessing()

        assert isinstance(pipeline, Compose)
        assert len(pipeline.transforms) == 5
        assert isinstance(pipeline.transforms[4], RandSpatialCrop)

    def test_create_brats_training_preprocessing_minmax(self) -> None:
        """Test training preprocessing with minmax normalization."""
        pipeline = create_brats_training_preprocessing(normalize_mode="minmax")

        assert isinstance(pipeline.transforms[2], ScaleIntensity)


class TestBraTSInferencePreprocessing:
    """Tests for BraTS inference preprocessing."""

    def test_create_brats_inference_preprocessing_default(self) -> None:
        """Test default inference preprocessing."""
        pipeline = create_brats_inference_preprocessing()

        assert isinstance(pipeline, Compose)
        assert len(pipeline.transforms) == 3
        assert isinstance(pipeline.transforms[0], EnsureType)

    def test_create_brats_inference_preprocessing_minmax(self) -> None:
        """Test inference preprocessing with minmax."""
        pipeline = create_brats_inference_preprocessing(normalize_mode="minmax")

        assert isinstance(pipeline.transforms[2], ScaleIntensity)


class TestBraTS2023Preprocessing:
    """Tests for BraTS 2023 dictionary-based preprocessing."""

    def test_create_brats2023_preprocessing_default(self) -> None:
        """Test default BraTS 2023 preprocessing."""
        pipeline = create_brats2023_preprocessing()

        assert isinstance(pipeline, Compose)
        # Should have LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, Resized
        assert len(pipeline.transforms) >= 3

    def test_create_brats2023_preprocessing_segmentation(self) -> None:
        """Test BraTS 2023 preprocessing for segmentation task."""
        pipeline = create_brats2023_preprocessing(task="segmentation")

        assert isinstance(pipeline, Compose)
        # Segmentation task should include label transforms

    def test_create_brats2023_preprocessing_invalid_normalize_mode(self) -> None:
        """Test that invalid normalize_mode raises error."""
        with pytest.raises(ValueError, match="normalize_mode must be"):
            create_brats2023_preprocessing(normalize_mode="invalid")

    def test_create_brats2023_preprocessing_invalid_task(self) -> None:
        """Test that invalid task raises error."""
        with pytest.raises(ValueError, match="task must be"):
            create_brats2023_preprocessing(task="invalid")


class TestBraTS2023TrainingPreprocessing:
    """Tests for BraTS 2023 training preprocessing."""

    def test_create_brats2023_training_preprocessing_default(self) -> None:
        """Test default training preprocessing."""
        pipeline = create_brats2023_training_preprocessing()

        assert isinstance(pipeline, Compose)

    def test_create_brats2023_training_preprocessing_segmentation(self) -> None:
        """Test training preprocessing for segmentation."""
        pipeline = create_brats2023_training_preprocessing(task="segmentation")

        assert isinstance(pipeline, Compose)

    def test_create_brats2023_training_preprocessing_invalid_mode(self) -> None:
        """Test invalid normalize_mode raises error."""
        with pytest.raises(ValueError, match="normalize_mode must be"):
            create_brats2023_training_preprocessing(normalize_mode="invalid")

    def test_create_brats2023_training_preprocessing_invalid_task(self) -> None:
        """Test invalid task raises error."""
        with pytest.raises(ValueError, match="task must be"):
            create_brats2023_training_preprocessing(task="invalid")


class TestBraTS2023InferencePreprocessing:
    """Tests for BraTS 2023 inference preprocessing."""

    def test_create_brats2023_inference_preprocessing_default(self) -> None:
        """Test default inference preprocessing."""
        pipeline = create_brats2023_inference_preprocessing()

        assert isinstance(pipeline, Compose)

    def test_create_brats2023_inference_preprocessing_segmentation(self) -> None:
        """Test inference preprocessing for segmentation."""
        pipeline = create_brats2023_inference_preprocessing(task="segmentation")

        assert isinstance(pipeline, Compose)

    def test_create_brats2023_inference_preprocessing_invalid_mode(self) -> None:
        """Test invalid normalize_mode raises error."""
        with pytest.raises(ValueError, match="normalize_mode must be"):
            create_brats2023_inference_preprocessing(normalize_mode="invalid")

    def test_create_brats2023_inference_preprocessing_invalid_task(self) -> None:
        """Test invalid task raises error."""
        with pytest.raises(ValueError, match="task must be"):
            create_brats2023_inference_preprocessing(task="invalid")
