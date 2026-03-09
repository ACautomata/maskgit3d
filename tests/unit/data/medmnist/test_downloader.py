"""Tests for MedMNIST Downloader."""

import sys
from unittest.mock import Mock

import pytest

from src.maskgit3d.data.medmnist.config import MedMNISTConfig, MedMNISTDatasetName
from src.maskgit3d.data.medmnist.downloader import MedMNISTDownloader


class TestMedMNISTDownloader:
    @pytest.fixture
    def config(self, tmp_path):
        """Create test config."""
        return MedMNISTConfig(
            dataset_name=MedMNISTDatasetName.ORGAN,
            data_dir=str(tmp_path),
            download=False,
        )

    @pytest.fixture
    def downloader(self, config):
        """Create downloader instance."""
        return MedMNISTDownloader(config)

    def test_init_creates_data_dir(self, tmp_path):
        """Test that init creates data directory."""
        config = MedMNISTConfig(
            dataset_name=MedMNISTDatasetName.ORGAN,
            data_dir=str(tmp_path / "new_dir"),
            download=False,
        )
        downloader = MedMNISTDownloader(config)
        assert downloader.data_dir.exists()

    def test_get_data_path(self, downloader):
        """Test data path generation."""
        path = downloader._get_data_path("train")
        assert path.name == "organmnist3d.npz"

    def test_check_cached_returns_false_when_not_exists(self, downloader, tmp_path):
        """Test _check_cached returns False when file doesn't exist."""
        result = downloader._check_cached("train")
        assert result is False

    def test_check_cached_returns_true_when_exists(self, downloader, tmp_path):
        """Test _check_cached returns True when file exists."""
        # Create fake file
        data_path = downloader._get_data_path("train")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_bytes(b"fake data")

        result = downloader._check_cached("train")
        assert result is True

    def test_check_cached_with_md5_mismatch(self, downloader, tmp_path):
        """Test _check_cached handles MD5 mismatch."""
        # Set expected MD5
        downloader.MD5_CHECKSUMS = {"organmnist3d": {"train": "wrong_md5"}}

        # Create file
        data_path = downloader._get_data_path("train")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_bytes(b"fake data")

        result = downloader._check_cached("train")
        assert result is False

    def test_compute_md5(self, downloader, tmp_path):
        """Test MD5 computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"hello world")

        md5 = downloader._compute_md5(test_file)
        # MD5 of "hello world"
        assert md5 == "5eb63bbbe01eeed093cb22bb8f5acdc3"

    def test_ensure_data_available_raises_when_not_cached_and_no_download(self, downloader):
        """Test ensure_data_available raises FileNotFoundError when not cached."""
        with pytest.raises(FileNotFoundError) as exc_info:
            downloader.ensure_data_available("train")

        assert "not found" in str(exc_info.value)

    def test_ensure_data_available_returns_cached_path(self, downloader):
        """Test ensure_data_available returns cached path when available."""
        data_path = downloader._get_data_path("train")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_bytes(b"fake data")

        result = downloader.ensure_data_available("train")
        assert result == data_path

    def test_get_data_path_format(self, downloader):
        """Test _get_data_path returns correct path."""
        path = downloader._get_data_path("train")
        assert "organmnist3d.npz" in str(path)

    def test_get_data_path_is_shared_across_splits(self, downloader):
        assert downloader._get_data_path("train") == downloader._get_data_path("val")
        assert downloader._get_data_path("val") == downloader._get_data_path("test")

    def test_ensure_data_available_downloads_single_archive(self, downloader, monkeypatch):
        """Test ensure_data_available downloads when data not cached."""
        downloader.config.download = True

        mock_medmnist = Mock()
        mock_dataset_cls = Mock()
        mock_medmnist.OrganMNIST3D = mock_dataset_cls
        mock_dataset_cls.return_value = Mock()

        monkeypatch.setitem(sys.modules, "medmnist", mock_medmnist)

        result = downloader.ensure_data_available("train")
        assert result == downloader.data_dir / "organmnist3d.npz"
        mock_dataset_cls.assert_called_once_with(
            split="train",
            root=str(downloader.data_dir),
            download=True,
        )

    def test_check_cached_exists(self, downloader):
        data_path = downloader._get_data_path("train")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_bytes(b"fake data")

        assert downloader._check_cached("train") is True

    def test_check_cached_not_exists(self, downloader):
        assert downloader._check_cached("nonexistent") is False

    def test_ensure_data_available_returns_cached(self, downloader, tmp_path, monkeypatch):
        data_path = downloader._get_data_path("train")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_bytes(b"cached data")

        class NoDownloadModule:
            pass

        mock_medmnist = NoDownloadModule()
        monkeypatch.setitem(sys.modules, "medmnist", mock_medmnist)

        result = downloader.ensure_data_available("train")

        assert result == data_path
        assert not hasattr(mock_medmnist, "OrganMNIST3D")
