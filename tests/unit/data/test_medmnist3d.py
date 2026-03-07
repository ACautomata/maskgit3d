from typing import Any

import numpy as np
import pytest
import torch
from monai.transforms.croppad.array import RandSpatialCrop


class _FakeMedMNIST3D:
    def __init__(
        self,
        root: str,
        split: str,
        download: bool,
        size: int = 28,
    ) -> None:
        self.root = root
        self.split = split
        self.download = download
        self.size = size

    def __len__(self) -> int:
        return 5

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        _ = idx
        return np.ones((self.size, self.size, self.size), dtype=np.float32), 1


def test_medmnist3d_datamodule_fit_setup_and_dataloaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskgit3d.infrastructure.data.medmnist_provider as med_provider
    from maskgit3d.data.medmnist3d import MedMNIST3DDataModule

    monkeypatch.setattr(med_provider, "_get_dataset_class", lambda _: _FakeMedMNIST3D)

    dm = MedMNIST3DDataModule(
        dataset_type="organ",
        crop_size=(16, 16, 16),
        input_size=28,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        drop_last_train=False,
    )

    dm.setup(stage="fit")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    train_inputs, train_targets = next(iter(train_loader))
    val_inputs, val_targets = next(iter(val_loader))

    assert train_inputs.shape == (2, 1, 16, 16, 16)
    assert train_targets.shape == (2, 1, 16, 16, 16)
    assert val_inputs.shape == (2, 1, 28, 28, 28)
    assert val_targets.shape == (2, 1, 28, 28, 28)


def test_medmnist3d_datamodule_test_setup_and_dataloader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskgit3d.infrastructure.data.medmnist_provider as med_provider
    from maskgit3d.data.medmnist3d import MedMNIST3DDataModule

    monkeypatch.setattr(med_provider, "_get_dataset_class", lambda _: _FakeMedMNIST3D)

    dm = MedMNIST3DDataModule(
        dataset_type="organ",
        input_size=28,
        batch_size=3,
        num_workers=0,
        pin_memory=False,
    )

    dm.setup(stage="test")

    test_loader = dm.test_dataloader()
    test_inputs, test_targets = next(iter(test_loader))

    assert test_inputs.shape == (3, 1, 28, 28, 28)
    assert test_targets.shape == (3, 1, 28, 28, 28)


def test_medmnist3d_datamodule_uses_monai_augmentation() -> None:
    from maskgit3d.data.medmnist3d import MedMNIST3DDataModule

    dm = MedMNIST3DDataModule(crop_size=(16, 16, 16))
    transform_types = [type(t) for t in dm.train_transform.transforms]

    assert RandSpatialCrop in transform_types
