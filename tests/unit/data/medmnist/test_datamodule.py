"""Tests for MedMNIST DataModule."""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytorch_lightning as pl

from src.maskgit3d.data.medmnist.config import MedMNISTConfig, MedMNISTDatasetName, TaskType
from src.maskgit3d.data.medmnist.datamodule import MedMNIST3DDataModule


class TestMedMNIST3DDataModule:
    @pytest.fixture
    def fake_data(self, tmp_path):
        """Create fake MedMNIST data files."""
        splits = ["train", "val", "test"]
        for split in splits:
            data_file = tmp_path / f"organmnist3d_{split}.npz"
            images = np.random.randint(0, 255, size=(10, 28, 28, 28), dtype=np.uint8)
            labels = np.random.randint(0, 11, size=(10, 1))
            np.savez(data_file, **{f"{split}_images": images, f"{split}_labels": labels})
        return tmp_path

    def test_init_with_default_params(self, tmp_path):
        """Test datamodule initialization with default params."""
        datamodule = MedMNIST3DDataModule(
            dataset_name="organmnist3d",
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            download=False,
        )
        assert datamodule.batch_size == 4
        assert datamodule.num_workers == 0

    def test_init_with_custom_params(self, tmp_path):
        """Test datamodule initialization with custom params."""
        datamodule = MedMNIST3DDataModule(
            dataset_name="nodulemnist3d",
            task_type="classification",
            data_dir=str(tmp_path),
            batch_size=8,
            num_workers=2,
            download=False,
        )
        assert datamodule.batch_size == 8
        assert datamodule.num_workers == 2
        assert datamodule.config.task_type == TaskType.CLASSIFICATION

    @patch("src.maskgit3d.data.medmnist.datamodule.MedMNIST3DDataset")
    def test_setup_fit(self, mock_dataset_cls, tmp_path, fake_data):
        """Test setup with fit stage."""
        mock_dataset_cls.return_value = Mock(__len__=Mock(return_value=10))

        datamodule = MedMNIST3DDataModule(
            dataset_name="organmnist3d",
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            download=False,
        )
        datamodule.setup(stage="fit")

        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None

    @patch("src.maskgit3d.data.medmnist.datamodule.MedMNIST3DDataset")
    def test_setup_test(self, mock_dataset_cls, tmp_path, fake_data):
        """Test setup with test stage."""
        mock_dataset_cls.return_value = Mock(__len__=Mock(return_value=10))

        datamodule = MedMNIST3DDataModule(
            dataset_name="organmnist3d",
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            download=False,
        )
        datamodule.setup(stage="test")

        assert datamodule.test_dataset is not None

    @patch("src.maskgit3d.data.medmnist.datamodule.MedMNIST3DDataset")
    def test_train_dataloader(self, mock_dataset_cls, tmp_path, fake_data):
        """Test train_dataloader creation."""
        mock_ds = Mock(__len__=Mock(return_value=10))
        mock_dataset_cls.return_value = mock_ds

        datamodule = MedMNIST3DDataModule(
            dataset_name="organmnist3d",
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            download=False,
        )
        datamodule.setup(stage="fit")

        loader = datamodule.train_dataloader()

        assert loader is not None
        assert loader.batch_size == 4

    @patch("src.maskgit3d.data.medmnist.datamodule.MedMNIST3DDataset")
    def test_val_dataloader(self, mock_dataset_cls, tmp_path, fake_data):
        """Test val_dataloader creation."""
        mock_ds = Mock(__len__=Mock(return_value=10))
        mock_dataset_cls.return_value = mock_ds

        datamodule = MedMNIST3DDataModule(
            dataset_name="organmnist3d",
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            download=False,
        )
        datamodule.setup(stage="fit")

        loader = datamodule.val_dataloader()

        assert loader is not None

    @patch("src.maskgit3d.data.medmnist.datamodule.MedMNIST3DDataset")
    def test_test_dataloader(self, mock_dataset_cls, tmp_path, fake_data):
        """Test test_dataloader creation."""
        mock_ds = Mock(__len__=Mock(return_value=10))
        mock_dataset_cls.return_value = mock_ds

        datamodule = MedMNIST3DDataModule(
            dataset_name="organmnist3d",
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            download=False,
        )
        datamodule.setup(stage="test")

        loader = datamodule.test_dataloader()

        assert loader is not None
