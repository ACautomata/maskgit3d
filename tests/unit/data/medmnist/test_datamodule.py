"""Tests for MedMNIST DataModule."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from maskgit3d.data.medmnist.config import TaskType
from maskgit3d.data.medmnist.datamodule import MedMNIST3DDataModule


class TestMedMNIST3DDataModule:
    @pytest.fixture
    def fake_data(self, tmp_path):
        """Create fake MedMNIST data files."""
        data_file = tmp_path / "organmnist3d.npz"
        archive = {}
        for split in ["train", "val", "test"]:
            images = np.random.randint(0, 255, size=(10, 28, 28, 28), dtype=np.uint8)
            labels = np.random.randint(0, 11, size=(10, 1))
            archive[f"{split}_images"] = images
            archive[f"{split}_labels"] = labels
        np.savez(data_file, **archive)
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

    @patch("maskgit3d.data.medmnist.datamodule.MedMNIST3DDataset")
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

    @patch("maskgit3d.data.medmnist.datamodule.MedMNIST3DDataset")
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

    @patch("maskgit3d.data.medmnist.datamodule.MedMNIST3DDataset")
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

    @patch("maskgit3d.data.medmnist.datamodule.MedMNIST3DDataset")
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

    @patch("maskgit3d.data.medmnist.datamodule.MedMNIST3DDataset")
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

    def test_train_dataloader_without_setup(self, tmp_path):
        """Test train_dataloader raises error without setup."""
        datamodule = MedMNIST3DDataModule(
            dataset_name="organmnist3d",
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            download=False,
        )

        with pytest.raises(RuntimeError, match="train_dataset is None"):
            datamodule.train_dataloader()

    def test_val_dataloader_without_setup(self, tmp_path):
        """Test val_dataloader raises error without setup."""
        datamodule = MedMNIST3DDataModule(
            dataset_name="organmnist3d",
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            download=False,
        )

        with pytest.raises(RuntimeError, match="val_dataset is None"):
            datamodule.val_dataloader()

    def test_test_dataloader_without_setup(self, tmp_path):
        """Test test_dataloader raises error without setup."""
        datamodule = MedMNIST3DDataModule(
            dataset_name="organmnist3d",
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            download=False,
        )

        with pytest.raises(RuntimeError, match="test_dataset is None"):
            datamodule.test_dataloader()

    def test_predict_dataloader_without_setup(self, tmp_path):
        """Test predict_dataloader raises error without setup."""
        datamodule = MedMNIST3DDataModule(
            dataset_name="organmnist3d",
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            download=False,
        )

        with pytest.raises(RuntimeError, match="test_dataset is None"):
            datamodule.predict_dataloader()

    @patch("maskgit3d.data.medmnist.datamodule.MedMNIST3DDataset")
    def test_predict_dataloader(self, mock_dataset_cls, tmp_path, fake_data):
        """Test predict_dataloader creation."""
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

        loader = datamodule.predict_dataloader()

        assert loader is not None

    @patch("maskgit3d.data.medmnist.datamodule.MedMNIST3DDataset")
    def test_setup_validate_stage(self, mock_dataset_cls, tmp_path, fake_data):
        """Test setup with validate stage."""
        mock_dataset_cls.return_value = Mock(__len__=Mock(return_value=10))

        datamodule = MedMNIST3DDataModule(
            dataset_name="organmnist3d",
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            download=False,
        )
        datamodule.setup(stage="validate")

        assert datamodule.val_dataset is not None
