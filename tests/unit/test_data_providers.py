"""
Tests for data providers: MedMnist3DDataProvider and BraTSDataProvider.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
import tempfile
import os

from maskgit3d.infrastructure.data.transforms import (
    create_3d_preprocessing,
    create_brats_preprocessing,
    create_medmnist_preprocessing,
    normalize_to_neg_one_one,
)


class TestTransforms:
    """Tests for preprocessing transforms."""

    def test_create_3d_preprocessing_minmax(self):
        """Test creating 3D preprocessing pipeline with minmax normalization."""
        transform = create_3d_preprocessing(
            spatial_size=(64, 64, 64),
            normalize_mode="minmax",
        )
        assert transform is not None

    def test_create_3d_preprocessing_zscore(self):
        """Test creating 3D preprocessing pipeline with zscore normalization."""
        transform = create_3d_preprocessing(
            spatial_size=(64, 64, 64),
            normalize_mode="zscore",
        )
        assert transform is not None

    def test_create_3d_preprocessing_invalid_mode(self):
        """Test that invalid normalization mode raises ValueError."""
        with pytest.raises(ValueError, match="normalize_mode"):
            create_3d_preprocessing(normalize_mode="invalid")

    def test_normalize_to_neg_one_one(self):
        """Test normalization from [0, 1] to [-1, 1]."""
        x = torch.tensor([0.0, 0.5, 1.0])
        result = normalize_to_neg_one_one(x)
        expected = torch.tensor([-1.0, 0.0, 1.0])
        assert torch.allclose(result, expected)

    def test_create_brats_preprocessing(self):
        """Test creating BraTS preprocessing pipeline."""
        transform = create_brats_preprocessing(
            spatial_size=(64, 64, 64),
            normalize_mode="zscore",
        )
        assert transform is not None

    def test_create_medmnist_preprocessing(self):
        """Test creating MedMNIST preprocessing pipeline."""
        transform = create_medmnist_preprocessing(
            spatial_size=(64, 64, 64),
            input_size=28,
        )
        assert transform is not None


class TestMedMnist3DDataProvider:
    """Tests for MedMnist3DDataProvider."""

    def test_invalid_dataset_type(self):
        """Test that invalid dataset type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            from maskgit3d.infrastructure.data.medmnist_provider import (
                MedMnist3DDataProvider,
            )
            MedMnist3DDataProvider(dataset_type="invalid_type")

    def test_invalid_input_size(self):
        """Test that invalid input size raises ValueError."""
        with pytest.raises(ValueError, match="input_size"):
            from maskgit3d.infrastructure.data.medmnist_provider import (
                MedMnist3DDataProvider,
            )
            MedMnist3DDataProvider(dataset_type="organ", input_size=32)

    @patch("maskgit3d.infrastructure.data.medmnist_provider._get_dataset_class")
    def test_supported_dataset_types(self, mock_get_class):
        """Test that all documented dataset types are accepted."""
        from maskgit3d.infrastructure.data.medmnist_provider import (
            MedMnist3DDataProvider,
            MedMNIST3DDataset,
        )

        # Mock the dataset class
        mock_dataset_class = MagicMock
        mock_dataset_class.num_classes = 11
        mock_get_class.return_value = mock_dataset_class

        supported_types = ["organ", "nodule", "adrenal", "vessel", "fracture", "synapse"]

        for dataset_type in supported_types:
            # Should not raise
            provider = MedMnist3DDataProvider(
                dataset_type=dataset_type,
                spatial_size=(28, 28, 28),
                batch_size=1,
            )
            assert provider.dataset_type.value == dataset_type


class TestBraTSDataProvider:
    """Tests for BraTSDataProvider."""

    def test_missing_data_dir(self):
        """Test that missing data directory raises FileNotFoundError."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with pytest.raises(FileNotFoundError, match="BraTS data directory not found"):
            BraTSDataProvider(data_dir="/nonexistent/path")

    def test_invalid_modality(self):
        """Test that invalid modality raises ValueError."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid-looking directory structure
            patient_dir = Path(tmpdir) / "BraTS2021_00001"
            patient_dir.mkdir()

            with pytest.raises(ValueError, match="Invalid modalities"):
                BraTSDataProvider(
                    data_dir=tmpdir,
                    modalities=["invalid_modality"],
                )

    def test_invalid_ratios(self):
        """Test that ratios not summing to 1.0 raise ValueError."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            patient_dir = Path(tmpdir) / "BraTS2021_00001"
            patient_dir.mkdir()

            # Create a dummy NIfTI file
            import nibabel as nib
            import numpy as np

            data = np.random.rand(10, 10, 10).astype(np.float32)
            affine = np.eye(4)
            img = nib.Nifti1Image(data, affine)
            nib.save(img, str(patient_dir / "BraTS2021_00001_t1.nii.gz"))

            with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
                BraTSDataProvider(
                    data_dir=tmpdir,
                    train_ratio=0.5,
                    val_ratio=0.3,
                    test_ratio=0.3,  # Sum = 1.1
                )

    def test_valid_initialization(self):
        """Test successful initialization with valid parameters."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid directory structure
            patient_dir = Path(tmpdir) / "BraTS2021_00001"
            patient_dir.mkdir()

            import nibabel as nib
            import numpy as np

            data = np.random.rand(10, 10, 10).astype(np.float32)
            affine = np.eye(4)
            img = nib.Nifti1Image(data, affine)
            nib.save(img, str(patient_dir / "BraTS2021_00001_t1.nii.gz"))

            provider = BraTSDataProvider(
                data_dir=tmpdir,
                modalities=["t1"],
                spatial_size=(32, 32, 32),
                batch_size=1,
            )

            assert provider.num_modalities == 1
            assert provider.num_train_samples + provider.num_val_samples + provider.num_test_samples == 1


class TestDataProviderIntegration:
    """Integration tests for data providers."""

    def test_data_shape_format(self):
        """Test that data providers return correct tensor shape [B, C, D, H, W]."""
        from maskgit3d.infrastructure.data.dataset import SimpleDataProvider

        provider = SimpleDataProvider(
            num_train=5,
            num_val=2,
            num_test=2,
            batch_size=2,
            in_channels=1,
            spatial_size=(64, 64, 64),
        )

        # Check training loader
        for inputs, targets in provider.train_loader():
            assert inputs.shape == (2, 1, 64, 64, 64)
            assert targets.shape == (2, 1, 64, 64, 64)
            break

    def test_config_module_registration(self):
        """Test that new providers are registered in config module."""
        from maskgit3d.config.modules import DataModule

        # Check that providers dict contains expected keys
        config = DataModule(data_config={"type": "simple", "params": {}})
        provider = config.provide_data_provider()

        # Verify it's a DataProvider instance
        from maskgit3d.domain.interfaces import DataProvider

        assert isinstance(provider, DataProvider)


class TestMedMNISTDatasetWrapper:
    """Tests for MedMNIST3DDatasetWrapper."""

    def test_wrapper_shape_handling(self):
        """Test that wrapper correctly handles tensor shapes."""
        from maskgit3d.infrastructure.data.medmnist_provider import (
            MedMNIST3DDatasetWrapper,
        )

        # Create a mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        # Return 3D tensor without channel dimension
        sample_image = torch.rand(28, 28, 28)
        mock_dataset.__getitem__ = MagicMock(return_value=(sample_image, 0))

        wrapper = MedMNIST3DDatasetWrapper(
            dataset=mock_dataset,
            spatial_size=(28, 28, 28),
        )

        result = wrapper[0]
        input_tensor, target_tensor = result

        # Should have channel dimension added
        assert input_tensor.dim() == 4
        assert input_tensor.shape[0] == 1  # Channel dimension


class TestBraTSDataset:
    """Tests for BraTSDataset."""

    def test_missing_modality_handling(self):
        """Test handling of missing modalities."""
        from maskgit3d.infrastructure.data.brats_provider import (
            BraTSDataset,
            BRATS_MODALITIES,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            patient_dir = Path(tmpdir) / "BraTS2021_00001"
            patient_dir.mkdir()

            # Create only T1 modality
            import nibabel as nib
            import numpy as np

            data = np.random.rand(10, 10, 10).astype(np.float32)
            affine = np.eye(4)
            img = nib.Nifti1Image(data, affine)
            nib.save(img, str(patient_dir / "BraTS2021_00001_t1.nii.gz"))

            dataset = BraTSDataset(
                data_dir=Path(tmpdir),
                patient_ids=["BraTS2021_00001"],
                modalities=["t1", "t2"],  # T2 is missing
                spatial_size=(32, 32, 32),
            )

            # Should not raise, but log warning
            assert len(dataset) == 1