"""Tests for MedMNIST Dataset."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from maskgit3d.data.medmnist.config import MedMNISTConfig, MedMNISTDatasetName, TaskType
from maskgit3d.data.medmnist.dataset import MedMNIST3DDataset
from maskgit3d.data.medmnist.downloader import MedMNISTDownloader


class TestMedMNIST3DDataset:
    @pytest.fixture
    def mock_downloader(self, tmp_path):
        """Create mock downloader."""
        downloader = Mock(spec=MedMNISTDownloader)
        downloader.ensure_data_available.return_value = tmp_path / "data.npz"
        return downloader

    @pytest.fixture
    def fake_medmnist_data(self, tmp_path):
        """Create fake MedMNIST data file."""
        data_file = tmp_path / "organmnist3d.npz"
        images = np.random.randint(0, 255, size=(10, 28, 28, 28), dtype=np.uint8)
        labels = np.random.randint(0, 11, size=(10, 1))
        np.savez(
            data_file,
            train_images=images,
            train_labels=labels,
            val_images=images,
            val_labels=labels,
            test_images=images,
            test_labels=labels,
        )
        return data_file

    @pytest.fixture
    def config(self):
        """Create test config."""
        return MedMNISTConfig(
            dataset_name=MedMNISTDatasetName.ORGAN,
            task_type=TaskType.RECONSTRUCTION,
            data_dir="/tmp/medmnist",
            download=False,
        )

    def test_init_with_reconstruction_task(self, config, tmp_path):
        """Test dataset initialization with reconstruction task."""
        data_file = tmp_path / "organmnist3d.npz"
        images = np.random.randint(0, 255, size=(10, 28, 28, 28), dtype=np.uint8)
        labels = np.random.randint(0, 11, size=(10, 1))
        np.savez(
            data_file,
            train_images=images,
            train_labels=labels,
            val_images=images,
            val_labels=labels,
            test_images=images,
            test_labels=labels,
        )

        mock_downloader = Mock()
        mock_downloader.ensure_data_available.return_value = data_file

        dataset = MedMNIST3DDataset(
            config=config,
            split="train",
            downloader=mock_downloader,
        )
        assert dataset.split == "train"
        assert dataset.task_type == TaskType.RECONSTRUCTION

    def test_init_with_classification_task(self, config, tmp_path):
        """Test dataset initialization with classification task."""
        config.task_type = TaskType.CLASSIFICATION
        data_file = tmp_path / "organmnist3d.npz"
        images = np.random.randint(0, 255, size=(10, 28, 28, 28), dtype=np.uint8)
        labels = np.random.randint(0, 11, size=(10, 1))
        np.savez(
            data_file,
            train_images=images,
            train_labels=labels,
            val_images=images,
            val_labels=labels,
            test_images=images,
            test_labels=labels,
        )

        mock_downloader = Mock()
        mock_downloader.ensure_data_available.return_value = data_file

        dataset = MedMNIST3DDataset(
            config=config,
            split="train",
            downloader=mock_downloader,
        )
        assert dataset.task_type == TaskType.CLASSIFICATION

    def test_len_returns_correct_count(self, config, tmp_path):
        """Test __len__ returns correct sample count."""
        mock_downloader = Mock()
        data_file = tmp_path / "organmnist3d.npz"
        images = np.random.randint(0, 255, size=(10, 28, 28, 28), dtype=np.uint8)
        labels = np.random.randint(0, 11, size=(10, 1))
        np.savez(
            data_file,
            train_images=images,
            train_labels=labels,
            val_images=images,
            val_labels=labels,
            test_images=images,
            test_labels=labels,
        )
        mock_downloader.ensure_data_available.return_value = data_file

        dataset = MedMNIST3DDataset(
            config=config,
            split="train",
            downloader=mock_downloader,
        )
        assert len(dataset) == 10

    def test_getitem_reconstruction_returns_image_and_target(self, config, tmp_path):
        """Test __getitem__ returns image and image.clone() for reconstruction."""
        config.task_type = TaskType.RECONSTRUCTION
        mock_downloader = Mock()
        data_file = tmp_path / "organmnist3d.npz"
        images = np.random.randint(0, 255, size=(10, 28, 28, 28), dtype=np.uint8)
        labels = np.random.randint(0, 11, size=(10, 1))
        np.savez(
            data_file,
            train_images=images,
            train_labels=labels,
            val_images=images,
            val_labels=labels,
            test_images=images,
            test_labels=labels,
        )
        mock_downloader.ensure_data_available.return_value = data_file

        dataset = MedMNIST3DDataset(
            config=config,
            split="train",
            downloader=mock_downloader,
        )

        image, target = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (1, 28, 28, 28)
        assert isinstance(target, torch.Tensor)
        assert target.shape == image.shape
        assert torch.equal(image, target)

    def test_getitem_classification_returns_image_and_label(self, config, tmp_path):
        """Test __getitem__ returns image and label for classification."""
        config.task_type = TaskType.CLASSIFICATION
        mock_downloader = Mock()
        data_file = tmp_path / "organmnist3d.npz"
        images = np.random.randint(0, 255, size=(10, 28, 28, 28), dtype=np.uint8)
        labels = np.random.randint(0, 11, size=(10, 1))
        np.savez(
            data_file,
            train_images=images,
            train_labels=labels,
            val_images=images,
            val_labels=labels,
            test_images=images,
            test_labels=labels,
        )
        mock_downloader.ensure_data_available.return_value = data_file

        dataset = MedMNIST3DDataset(
            config=config,
            split="train",
            downloader=mock_downloader,
        )

        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert label.item() >= 0 and label.item() < 11
