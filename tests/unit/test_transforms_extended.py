"""Extended tests for transforms module to improve coverage."""

import numpy as np
import pytest
import torch
from monai.transforms.compose import Compose

from maskgit3d.infrastructure.data.transforms import (
    create_3d_preprocessing,
    create_brats_preprocessing,
    create_brats2023_inference_preprocessing,
    create_brats2023_preprocessing,
    create_brats2023_training_preprocessing,
    create_brats_inference_preprocessing,
    create_brats_training_preprocessing,
    create_medmnist_inference_preprocessing,
    create_medmnist_preprocessing,
    create_medmnist_training_preprocessing,
    create_vqvae2023_inference_preprocessing,
    create_vqvae2023_training_preprocessing,
    create_vqvae_inference_preprocessing,
    create_vqvae_sliding_window_inference_preprocessing,
    create_vqvae_training_preprocessing,
    normalize_to_neg_one_one,
)


class TestNormalizeToNegOneOne:
    """Tests for normalize_to_neg_one_one function."""

    def test_normalize_basic(self):
        """Test basic normalization."""
        x = torch.tensor([0.0, 0.5, 1.0])
        result = normalize_to_neg_one_one(x)
        expected = torch.tensor([-1.0, 0.0, 1.0])
        assert torch.allclose(result, expected)

    def test_normalize_3d(self):
        """Test normalization with 3D tensor."""
        x = torch.rand(2, 4, 8, 8, 8)  # [B, C, D, H, W]
        result = normalize_to_neg_one_one(x)
        assert result.shape == x.shape
        assert result.min() >= -1.0
        assert result.max() <= 1.0


class TestCreate3DPreprocessing:
    """Tests for create_3d_preprocessing function."""

    def test_create_3d_preprocessing_minmax(self):
        """Test creating preprocessing with minmax normalization."""
        transforms = create_3d_preprocessing(
            spatial_size=(64, 64, 64),
            normalize_mode="minmax",
            output_range=(-1.0, 1.0),
        )
        assert isinstance(transforms, Compose)

    def test_create_3d_preprocessing_zscore(self):
        """Test creating preprocessing with zscore normalization."""
        transforms = create_3d_preprocessing(
            spatial_size=(64, 64, 64),
            normalize_mode="zscore",
        )
        assert isinstance(transforms, Compose)

    def test_create_3d_preprocessing_invalid_mode(self):
        """Test that invalid normalization mode raises ValueError."""
        with pytest.raises(ValueError, match="normalize_mode must be 'minmax' or 'zscore'"):
            create_3d_preprocessing(normalize_mode="invalid")

    def test_create_3d_preprocessing_applies(self):
        """Test that preprocessing actually transforms data."""
        transforms = create_3d_preprocessing(spatial_size=(32, 32, 32))
        # Create sample 3D data [D, H, W]
        data = np.random.rand(64, 64, 64).astype(np.float32)
        result = transforms(data)
        assert result.shape == (1, 32, 32, 32)  # [C, D, H, W]


class TestCreateBratsPreprocessing:
    """Tests for BraTS preprocessing functions."""

    def test_create_brats_preprocessing_zscore(self):
        """Test BraTS preprocessing with zscore."""
        transforms = create_brats_preprocessing(
            spatial_size=(64, 64, 64),
            normalize_mode="zscore",
        )
        assert isinstance(transforms, Compose)

    def test_create_brats_preprocessing_minmax(self):
        """Test BraTS preprocessing with minmax."""
        transforms = create_brats_preprocessing(
            spatial_size=(64, 64, 64),
            normalize_mode="minmax",
        )
        assert isinstance(transforms, Compose)

    def test_create_brats_training_preprocessing(self):
        """Test BraTS training preprocessing."""
        transforms = create_brats_training_preprocessing(
            crop_size=(128, 128, 128),
            normalize_mode="zscore",
        )
        assert isinstance(transforms, Compose)

    def test_create_brats_inference_preprocessing(self):
        """Test BraTS inference preprocessing."""
        transforms = create_brats_inference_preprocessing(normalize_mode="zscore")
        assert isinstance(transforms, Compose)


class TestCreateMedMnistPreprocessing:
    """Tests for MedMnist preprocessing functions."""

    def test_create_medmnist_preprocessing(self):
        """Test MedMnist preprocessing."""
        transforms = create_medmnist_preprocessing(
            spatial_size=(28, 28, 28),
            input_size=28,
        )
        assert isinstance(transforms, Compose)

    def test_create_medmnist_preprocessing_resize(self):
        """Test MedMnist preprocessing with resize."""
        transforms = create_medmnist_preprocessing(
            spatial_size=(64, 64, 64),
            input_size=28,
        )
        assert isinstance(transforms, Compose)

    def test_create_medmnist_training_preprocessing(self):
        """Test MedMnist training preprocessing."""
        transforms = create_medmnist_training_preprocessing(
            crop_size=(32, 32, 32),
            input_size=28,
        )
        assert isinstance(transforms, Compose)

    def test_create_medmnist_inference_preprocessing(self):
        """Test MedMnist inference preprocessing."""
        transforms = create_medmnist_inference_preprocessing()
        assert isinstance(transforms, Compose)

    def test_medmnist_preprocessing_applies(self):
        """Test that MedMnist preprocessing transforms data."""
        transforms = create_medmnist_preprocessing(spatial_size=(28, 28, 28))
        # Create sample MedMnist-style data [D, H, W]
        data = np.random.randint(0, 256, (28, 28, 28)).astype(np.float32)
        result = transforms(data)
        assert result.shape == (1, 28, 28, 28)


class TestCreateBrats2023Preprocessing:
    """Tests for BraTS 2023 preprocessing functions."""

    def test_create_brats2023_preprocessing_reconstruction(self):
        """Test BraTS 2023 preprocessing for reconstruction task."""
        transforms = create_brats2023_preprocessing(
            spatial_size=(64, 64, 64),
            normalize_mode="zscore",
            task="reconstruction",
        )
        assert isinstance(transforms, Compose)

    def test_create_brats2023_preprocessing_segmentation(self):
        """Test BraTS 2023 preprocessing for segmentation task."""
        transforms = create_brats2023_preprocessing(
            spatial_size=(64, 64, 64),
            normalize_mode="zscore",
            task="segmentation",
        )
        assert isinstance(transforms, Compose)

    def test_create_brats2023_preprocessing_invalid_mode(self):
        """Test that invalid normalize_mode raises ValueError."""
        with pytest.raises(ValueError, match="normalize_mode must be 'minmax' or 'zscore'"):
            create_brats2023_preprocessing(normalize_mode="invalid")

    def test_create_brats2023_preprocessing_invalid_task(self):
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be 'reconstruction' or 'segmentation'"):
            create_brats2023_preprocessing(task="invalid")

    def test_create_brats2023_training_preprocessing(self):
        """Test BraTS 2023 training preprocessing."""
        transforms = create_brats2023_training_preprocessing(
            crop_size=(128, 128, 128),
            normalize_mode="zscore",
            task="reconstruction",
        )
        assert isinstance(transforms, Compose)

    def test_create_brats2023_training_preprocessing_segmentation(self):
        """Test BraTS 2023 training preprocessing for segmentation."""
        transforms = create_brats2023_training_preprocessing(
            crop_size=(128, 128, 128),
            normalize_mode="minmax",
            task="segmentation",
        )
        assert isinstance(transforms, Compose)

    def test_create_brats2023_inference_preprocessing(self):
        """Test BraTS 2023 inference preprocessing."""
        transforms = create_brats2023_inference_preprocessing(
            normalize_mode="zscore",
            task="reconstruction",
        )
        assert isinstance(transforms, Compose)


class TestCreateVqvaePreprocessing:
    """Tests for VQVAE-aware preprocessing functions."""

    def test_create_vqvae_training_preprocessing(self):
        """Test VQVAE training preprocessing."""
        transforms = create_vqvae_training_preprocessing(
            crop_size=(128, 128, 128),
            normalize_mode="zscore",
            channel_multipliers=(1, 1, 2, 2, 4),
        )
        assert isinstance(transforms, Compose)

    def test_create_vqvae_training_preprocessing_invalid_mode(self):
        """Test that invalid normalize_mode raises ValueError."""
        with pytest.raises(ValueError, match="normalize_mode must be 'minmax' or 'zscore'"):
            create_vqvae_training_preprocessing(normalize_mode="invalid")

    def test_create_vqvae_inference_preprocessing(self):
        """Test VQVAE inference preprocessing."""
        transforms = create_vqvae_inference_preprocessing(
            normalize_mode="zscore",
            channel_multipliers=(1, 1, 2, 2, 4),
        )
        assert isinstance(transforms, Compose)

    def test_create_vqvae_inference_preprocessing_minmax(self):
        """Test VQVAE inference preprocessing with minmax."""
        transforms = create_vqvae_inference_preprocessing(
            normalize_mode="minmax",
            channel_multipliers=(1, 1, 2, 2, 4),
        )
        assert isinstance(transforms, Compose)

    def test_create_vqvae_sliding_window_inference_preprocessing(self):
        """Test VQVAE sliding window inference preprocessing."""
        transforms = create_vqvae_sliding_window_inference_preprocessing(
            normalize_mode="zscore",
            channel_multipliers=(1, 1, 2, 2, 4),
        )
        assert isinstance(transforms, Compose)


class TestCreateVqvae2023Preprocessing:
    """Tests for VQVAE 2023 preprocessing functions."""

    def test_create_vqvae2023_training_preprocessing(self):
        """Test VQVAE 2023 training preprocessing."""
        transforms = create_vqvae2023_training_preprocessing(
            crop_size=(128, 128, 128),
            normalize_mode="zscore",
            task="reconstruction",
            channel_multipliers=(1, 1, 2, 2, 4),
        )
        assert isinstance(transforms, Compose)

    def test_create_vqvae2023_training_preprocessing_segmentation(self):
        """Test VQVAE 2023 training preprocessing for segmentation."""
        transforms = create_vqvae2023_training_preprocessing(
            crop_size=(128, 128, 128),
            normalize_mode="minmax",
            task="segmentation",
            channel_multipliers=(1, 1, 2, 2, 4),
        )
        assert isinstance(transforms, Compose)

    def test_create_vqvae2023_training_preprocessing_invalid_mode(self):
        """Test that invalid normalize_mode raises ValueError."""
        with pytest.raises(ValueError, match="normalize_mode must be 'minmax' or 'zscore'"):
            create_vqvae2023_training_preprocessing(normalize_mode="invalid")

    def test_create_vqvae2023_training_preprocessing_invalid_task(self):
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be 'reconstruction' or 'segmentation'"):
            create_vqvae2023_training_preprocessing(task="invalid")

    def test_create_vqvae2023_inference_preprocessing(self):
        """Test VQVAE 2023 inference preprocessing."""
        transforms = create_vqvae2023_inference_preprocessing(
            normalize_mode="zscore",
            task="reconstruction",
            channel_multipliers=(1, 1, 2, 2, 4),
        )
        assert isinstance(transforms, Compose)

    def test_create_vqvae2023_inference_preprocessing_segmentation(self):
        """Test VQVAE 2023 inference preprocessing for segmentation."""
        transforms = create_vqvae2023_inference_preprocessing(
            normalize_mode="minmax",
            task="segmentation",
            channel_multipliers=(1, 1, 2, 2, 4),
        )
        assert isinstance(transforms, Compose)

    def test_create_vqvae2023_inference_preprocessing_invalid_mode(self):
        """Test that invalid normalize_mode raises ValueError."""
        with pytest.raises(ValueError, match="normalize_mode must be 'minmax' or 'zscore'"):
            create_vqvae2023_inference_preprocessing(normalize_mode="invalid")

    def test_create_vqvae2023_inference_preprocessing_invalid_task(self):
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be 'reconstruction' or 'segmentation'"):
            create_vqvae2023_inference_preprocessing(task="invalid")


class TestPreprocessingApplication:
    """Tests for applying preprocessing transforms."""

    def test_brats_preprocessing_applies(self):
        """Test that BraTS preprocessing transforms data correctly."""
        transforms = create_brats_preprocessing(spatial_size=(64, 64, 64))
        # Create sample 3D MRI data [D, H, W]
        data = np.random.rand(128, 128, 128).astype(np.float32)
        result = transforms(data)
        assert result.shape == (1, 64, 64, 64)  # [C, D, H, W]

    def test_brats_training_preprocessing_applies(self):
        """Test that BraTS training preprocessing transforms data."""
        transforms = create_brats_training_preprocessing(crop_size=(64, 64, 64))
        data = np.random.rand(128, 128, 128).astype(np.float32)
        result = transforms(data)
        assert result.shape == (1, 64, 64, 64)

    def test_brats_inference_preprocessing_applies(self):
        """Test that BraTS inference preprocessing transforms data."""
        transforms = create_brats_inference_preprocessing()
        data = np.random.rand(128, 128, 128).astype(np.float32)
        result = transforms(data)
        assert result.shape[0] == 1  # Channel dimension added

    def test_vqvae_inference_preprocessing_applies(self):
        """Test that VQVAE inference preprocessing transforms data."""
        transforms = create_vqvae_inference_preprocessing()
        data = np.random.rand(100, 100, 100).astype(np.float32)
        result = transforms(data)
        assert result.shape[0] == 1  # Channel dimension added
