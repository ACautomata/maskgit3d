"""Tests for BraTS2023 DataModule."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from maskgit3d.data.brats.config import BraTSSubDataset
from maskgit3d.data.brats.datamodule import BraTS2023DataModule
from maskgit3d.data.brats.dataset import BraTS2023CaseRecord


class TestBraTS2023DataModule:
    """Test BraTS2023 DataModule."""

    def create_mock_cases(self, n_gli: int = 3, n_men: int = 2, n_met: int = 2) -> list:
        """Helper to create mock case records."""
        cases = []
        for i in range(n_gli):
            cases.append(
                BraTS2023CaseRecord(
                    case_id=f"BraTS-GLI-{i:05d}-000",
                    subdataset=BraTSSubDataset.GLI,
                    image_paths=[
                        Path(f"/data/BraTS-GLI-{i:05d}-000/BraTS-GLI-{i:05d}-000-{mod}.nii.gz")
                        for mod in ["t1n", "t1c", "t2w", "t2f"]
                    ],
                )
            )
        for i in range(n_men):
            cases.append(
                BraTS2023CaseRecord(
                    case_id=f"BraTS-MEN-{i:05d}-000",
                    subdataset=BraTSSubDataset.MEN,
                    image_paths=[
                        Path(f"/data/BraTS-MEN-{i:05d}-000/BraTS-MEN-{i:05d}-000-{mod}.nii.gz")
                        for mod in ["t1n", "t1c", "t2w", "t2f"]
                    ],
                )
            )
        for i in range(n_met):
            cases.append(
                BraTS2023CaseRecord(
                    case_id=f"BraTS-MET-{i:05d}-000",
                    subdataset=BraTSSubDataset.MET,
                    image_paths=[
                        Path(f"/data/BraTS-MET-{i:05d}-000/BraTS-MET-{i:05d}-000-{mod}.nii.gz")
                        for mod in ["t1n", "t1c", "t2w", "t2f"]
                    ],
                )
            )
        return cases

    def test_init_with_default_params(self, tmp_path: Path) -> None:
        """Test datamodule initialization with default params."""
        datamodule = BraTS2023DataModule(data_dir=str(tmp_path))
        assert datamodule.batch_size == 2
        assert datamodule.num_workers == 4
        assert datamodule.config.crop_size == (128, 128, 128)

    def test_init_with_custom_params(self, tmp_path: Path) -> None:
        """Test datamodule initialization with custom params."""
        datamodule = BraTS2023DataModule(
            data_dir=str(tmp_path),
            crop_size=(64, 64, 64),
            batch_size=8,
            num_workers=2,
            train_ratio=0.9,
            subdatasets=["gli"],
        )
        assert datamodule.batch_size == 8
        assert datamodule.num_workers == 2
        assert datamodule.config.crop_size == (64, 64, 64)
        assert datamodule.config.train_ratio == 0.9

    @patch("maskgit3d.data.brats.datamodule._discover_cases")
    def test_setup_fit(self, mock_discover: MagicMock, tmp_path: Path) -> None:
        """Test setup with fit stage."""
        mock_discover.return_value = self.create_mock_cases(10, 10, 10)

        datamodule = BraTS2023DataModule(
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
        )
        datamodule.setup(stage="fit")

        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None

    @patch("maskgit3d.data.brats.datamodule._discover_cases")
    def test_setup_test(self, mock_discover: MagicMock, tmp_path: Path) -> None:
        """Test setup with test stage."""
        mock_discover.return_value = self.create_mock_cases(10, 10, 10)

        datamodule = BraTS2023DataModule(
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
        )
        datamodule.setup(stage="test")

        assert datamodule.test_dataset is not None

    @patch("maskgit3d.data.brats.datamodule._discover_cases")
    def test_val_equals_test(self, mock_discover: MagicMock, tmp_path: Path) -> None:
        """Test that val and test dataloaders return same data (val=test split)."""
        mock_discover.return_value = self.create_mock_cases(10, 10, 10)

        datamodule = BraTS2023DataModule(
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
        )
        datamodule.setup(stage="fit")

        assert datamodule.val_dataset is datamodule.test_dataset

    @patch("maskgit3d.data.brats.datamodule._discover_cases")
    def test_train_dataloader(self, mock_discover: MagicMock, tmp_path: Path) -> None:
        """Test train_dataloader creation."""
        mock_discover.return_value = self.create_mock_cases(10, 10, 10)

        datamodule = BraTS2023DataModule(
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
        )
        datamodule.setup(stage="fit")

        loader = datamodule.train_dataloader()

        assert loader is not None
        assert loader.batch_size == 4

    @patch("maskgit3d.data.brats.datamodule._discover_cases")
    def test_val_dataloader(self, mock_discover: MagicMock, tmp_path: Path) -> None:
        """Test val_dataloader creation."""
        mock_discover.return_value = self.create_mock_cases(10, 10, 10)

        datamodule = BraTS2023DataModule(
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
        )
        datamodule.setup(stage="fit")

        loader = datamodule.val_dataloader()

        assert loader is not None

    @patch("maskgit3d.data.brats.datamodule._discover_cases")
    def test_test_dataloader(self, mock_discover: MagicMock, tmp_path: Path) -> None:
        """Test test_dataloader creation."""
        mock_discover.return_value = self.create_mock_cases(10, 10, 10)

        datamodule = BraTS2023DataModule(
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
        )
        datamodule.setup(stage="test")

        loader = datamodule.test_dataloader()

        assert loader is not None

    def test_train_dataloader_without_setup(self, tmp_path: Path) -> None:
        """Test train_dataloader raises error without setup."""
        datamodule = BraTS2023DataModule(
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
        )

        with pytest.raises(RuntimeError, match="train_dataset is None"):
            datamodule.train_dataloader()

    def test_val_dataloader_without_setup(self, tmp_path: Path) -> None:
        """Test val_dataloader raises error without setup."""
        datamodule = BraTS2023DataModule(
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
        )

        with pytest.raises(RuntimeError, match="val_dataset is None"):
            datamodule.val_dataloader()

    def test_test_dataloader_without_setup(self, tmp_path: Path) -> None:
        """Test test_dataloader raises error without setup."""
        datamodule = BraTS2023DataModule(
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
        )

        with pytest.raises(RuntimeError, match="test_dataset is None"):
            datamodule.test_dataloader()

    @patch("maskgit3d.data.brats.datamodule._discover_cases")
    def test_predict_dataloader(self, mock_discover: MagicMock, tmp_path: Path) -> None:
        """Test predict_dataloader creation."""
        mock_discover.return_value = self.create_mock_cases(10, 10, 10)

        datamodule = BraTS2023DataModule(
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
        )
        datamodule.setup(stage="test")

        loader = datamodule.predict_dataloader()

        assert loader is not None

    @patch("maskgit3d.data.brats.datamodule._discover_cases")
    def test_stratified_split(self, mock_discover: MagicMock, tmp_path: Path) -> None:
        """Test that split is stratified by subdataset."""
        mock_discover.return_value = self.create_mock_cases(10, 10, 10)

        datamodule = BraTS2023DataModule(
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            train_ratio=0.8,
        )
        datamodule.setup(stage="fit")

        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None
        assert len(datamodule.train_dataset) == 24
        assert len(datamodule.val_dataset) == 6

    @patch("maskgit3d.data.brats.datamodule._discover_cases")
    def test_single_subdataset(self, mock_discover: MagicMock, tmp_path: Path) -> None:
        """Test datamodule with single subdataset."""
        mock_discover.return_value = self.create_mock_cases(10, 0, 0)

        datamodule = BraTS2023DataModule(
            data_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            subdatasets=["gli"],
        )
        datamodule.setup(stage="fit")

        assert datamodule.train_dataset is not None
        assert len(datamodule.train_dataset) == 8
