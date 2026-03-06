"""Tests for MedMnist3DDataProvider crop_size and roi_size parameters."""

from unittest.mock import MagicMock

import pytest

from maskgit3d.infrastructure.data.medmnist_provider import MedMnist3DDataProvider


class _FakeBaseDataset:
    """Fake MedMNIST dataset for testing."""

    num_classes = 7

    def __init__(
        self,
        root: str,
        split: str,
        download: bool,
        size: int = 28,
    ):
        self.root = root
        self.split = split
        self.download = download
        self.size = size

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int):
        import numpy as np

        return (self.size, self.size, self.size), 1


def test_provider_uses_crop_size_for_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that training uses crop_size parameter."""

    def mock_get_dataset_class(self):
        return _FakeBaseDataset

    monkeypatch.setattr(MedMnist3DDataProvider, "_get_dataset_class", mock_get_dataset_class)

    provider = MedMnist3DDataProvider(
        dataset_type="organ",
        crop_size=(32, 32, 32),
        roi_size=(64, 64, 64),
        data_root="./data",
        download=False,
    )

    assert provider.crop_size == (32, 32, 32)
    assert provider.roi_size == (64, 64, 64)


def test_provider_defaults_to_spatial_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that crop_size and roi_size default to spatial_size."""

    def mock_get_dataset_class(self):
        return _FakeBaseDataset

    monkeypatch.setattr(MedMnist3DDataProvider, "_get_dataset_class", mock_get_dataset_class)

    provider = MedMnist3DDataProvider(
        dataset_type="organ",
        spatial_size=(48, 48, 48),
        data_root="./data",
        download=False,
    )

    assert provider.crop_size == (48, 48, 48)
    assert provider.roi_size == (48, 48, 48)


def test_provider_has_separate_transforms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that train and inference transforms are separate."""

    def mock_get_dataset_class(self):
        return _FakeBaseDataset

    monkeypatch.setattr(MedMnist3DDataProvider, "_get_dataset_class", mock_get_dataset_class)

    from monai.transforms.compose import Compose

    provider = MedMnist3DDataProvider(
        dataset_type="organ",
        crop_size=(32, 32, 32),
        roi_size=(64, 64, 64),
        data_root="./data",
        download=False,
    )

    assert isinstance(provider.train_transform, Compose)
    assert isinstance(provider.inference_transform, Compose)
    assert provider.train_transform != provider.inference_transform


def test_train_dataset_uses_train_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that train_dataset uses train_transform."""

    def mock_get_dataset_class(self):
        return _FakeBaseDataset

    monkeypatch.setattr(MedMnist3DDataProvider, "_get_dataset_class", mock_get_dataset_class)

    provider = MedMnist3DDataProvider(
        dataset_type="organ",
        crop_size=(32, 32, 32),
        data_root="./data",
        download=False,
    )

    train_ds = provider.train_dataset
    assert train_ds.transform is provider.train_transform


def test_val_dataset_uses_inference_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that val_dataset uses inference_transform."""

    def mock_get_dataset_class(self):
        return _FakeBaseDataset

    monkeypatch.setattr(MedMnist3DDataProvider, "_get_dataset_class", mock_get_dataset_class)

    provider = MedMnist3DDataProvider(
        dataset_type="organ",
        crop_size=(32, 32, 32),
        data_root="./data",
        download=False,
    )

    val_ds = provider.val_dataset
    assert val_ds.transform is provider.inference_transform


def test_test_dataset_uses_inference_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that test_dataset uses inference_transform."""

    def mock_get_dataset_class(self):
        return _FakeBaseDataset

    monkeypatch.setattr(MedMnist3DDataProvider, "_get_dataset_class", mock_get_dataset_class)

    provider = MedMnist3DDataProvider(
        dataset_type="organ",
        crop_size=(32, 32, 32),
        data_root="./data",
        download=False,
    )

    test_ds = provider.test_dataset
    assert test_ds.transform is provider.inference_transform
