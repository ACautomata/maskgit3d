from typing import Any

import numpy as np
import pytest
import torch

import maskgit3d.infrastructure.data.medmnist_provider as med


class _FakeBaseDataset:
    def __init__(self, root: str, split: str, download: bool):
        self.root = root
        self.split = split
        self.download = download

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        _ = idx
        return np.ones((28, 28, 28), dtype=np.float32), 1


def test_get_dataset_class_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(med, "DATASET_CLASS_MAP", {})

    original_import = __import__

    def fake_import(name: str, *args: Any, **kwargs: Any):
        if name == "medmnist":
            raise ImportError("missing medmnist")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError, match="medmnist package is required"):
        med._get_dataset_class(med.MedMNIST3DDataset.ORGAN)


def test_get_dataset_class_unsupported_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(med, "DATASET_CLASS_MAP", {med.MedMNIST3DDataset.ORGAN: _FakeBaseDataset})

    with pytest.raises(ValueError, match="Unsupported dataset type"):
        med._get_dataset_class(med.MedMNIST3DDataset.NODULE)


def test_wrapper_getitem_numpy_and_transform_path() -> None:
    wrapper = med.MedMNIST3DDatasetWrapper(
        dataset=_FakeBaseDataset("./data", "train", True),
        transform=lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x,
        spatial_size=(64, 64, 64),
    )

    image, target = wrapper[0]
    assert len(wrapper) == 3
    assert image.shape == (1, 28, 28, 28)
    assert torch.equal(image, target)


def test_wrapper_getitem_tensor_label_path() -> None:
    class TensorDataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            _ = idx
            return torch.zeros(1, 28, 28, 28), torch.tensor(0)

    wrapper = med.MedMNIST3DDatasetWrapper(dataset=TensorDataset(), transform=None)
    image, target = wrapper[0]
    assert image.shape == (1, 28, 28, 28)
    assert torch.equal(image, target)


def test_provider_dataset_creation_and_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDatasetClass(_FakeBaseDataset):
        num_classes = 7

    monkeypatch.setattr(med, "_get_dataset_class", lambda _: FakeDatasetClass)

    provider = med.MedMnist3DDataProvider(
        dataset_type="organ",
        spatial_size=(32, 32, 32),
        input_size=28,
        batch_size=2,
        num_workers=0,
        drop_last_train=False,
        pin_memory=False,
    )

    assert provider.train_dataset is provider.train_dataset
    assert provider.val_dataset is provider.val_dataset
    assert provider.test_dataset is provider.test_dataset

    train_loader = provider.train_loader()
    val_loader = provider.val_loader()
    test_loader = provider.test_loader()

    assert train_loader.drop_last is False
    assert val_loader.drop_last is False
    assert test_loader.drop_last is False
    assert provider.get_num_classes() == 7

    info = provider.get_dataset_info()
    assert info["dataset_type"] == "organ"
    assert info["num_train"] == 3
    assert info["num_val"] == 3
    assert info["num_test"] == 3
    assert info["num_classes"] == 7
