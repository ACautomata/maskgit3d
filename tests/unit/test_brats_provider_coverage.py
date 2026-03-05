# pyright: reportArgumentType=false
from pathlib import Path

import numpy as np
import pytest
import torch

from maskgit3d.infrastructure.data.brats_provider import BraTS2021Dataset, BraTS2023Dataset


def _create_empty_nii(path: Path) -> None:
    path.touch()


def test_brats2021_dataset_handles_missing_modality_and_numpy_transform(tmp_path: Path) -> None:
    patient_id = "BraTS2021_00001"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir(parents=True)
    _create_empty_nii(patient_dir / f"{patient_id}_t1.nii.gz")

    dataset = BraTS2021Dataset(
        data_dir=tmp_path,
        patient_ids=[patient_id],
        modalities=["t1", "t2"],
        transform=lambda x: x.numpy(),  # type: ignore[arg-type]
        spatial_size=(4, 4, 4),
    )

    dataset._load_nifti = lambda _: torch.ones(4, 4, 4)  # type: ignore[method-assign]

    image, target = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (2, 4, 4, 4)
    assert torch.equal(image, target)


def test_brats2021_dataset_validate_modalities_raises() -> None:
    with pytest.raises(ValueError, match="Invalid modalities"):
        BraTS2021Dataset(
            data_dir=Path("."),
            patient_ids=[],
            modalities=["bad_modality"],
        )


def test_brats2021_load_nifti_uses_nibabel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyNifti:
        def get_fdata(self) -> np.ndarray:
            return np.ones((2, 2, 2), dtype=np.float32)

    monkeypatch.setattr(
        "maskgit3d.infrastructure.data.brats_provider.nib.load",
        lambda _: DummyNifti(),
    )

    dataset = BraTS2021Dataset(data_dir=tmp_path, patient_ids=[], modalities=["t1"])
    tensor = dataset._load_nifti(tmp_path / "unused.nii.gz")
    assert tensor.shape == (2, 2, 2)


def test_brats2023_dataset_transform_and_segmentation_numpy() -> None:
    image_np = np.ones((4, 3, 3, 3), dtype=np.float32)
    label_np = np.zeros((3, 3, 3, 3), dtype=np.float32)
    ds = BraTS2023Dataset(
        data_dicts=[{"image": image_np, "label": label_np, "tumor_type": 1}],
        transform=lambda d: {
            "image": d["image"],
            "label": d["label"],
            "tumor_type": d["tumor_type"],
        },
        task="segmentation",
    )

    image, target, tumor_type = ds[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert tumor_type.dtype == torch.long


def test_brats_provider_2021_get_patient_info_and_missing_patient(tmp_path: Path) -> None:
    from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

    patient_id = "BraTS2021_00001"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir(parents=True)
    _create_empty_nii(patient_dir / f"{patient_id}_t1.nii.gz")

    provider = BraTSDataProvider(
        data_dir=tmp_path,
        version="2021",
        modalities=["t1"],
        spatial_size=(4, 4, 4),
        num_workers=0,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
    )

    info = provider.get_patient_info(patient_id)
    assert info["patient_id"] == patient_id
    assert "t1" in info["available_modalities"]

    with pytest.raises(ValueError, match="Patient not found"):
        provider.get_patient_info("not_exists")


def test_brats_provider_2021_dataset_properties_and_loader_flags(tmp_path: Path) -> None:
    from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

    patient_id = "BraTS2021_00001"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir(parents=True)
    _create_empty_nii(patient_dir / f"{patient_id}_t1.nii.gz")

    provider = BraTSDataProvider(
        data_dir=tmp_path,
        version="2021",
        modalities=["t1"],
        spatial_size=(4, 4, 4),
        num_workers=0,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
    )

    assert provider.train_dataset is provider.train_dataset
    assert provider.val_dataset is provider.val_dataset
    assert provider.test_dataset is provider.test_dataset

    train_loader = provider.train_loader()
    val_loader = provider.val_loader()
    test_loader = provider.test_loader()

    assert train_loader.drop_last is True
    assert val_loader.drop_last is False
    assert test_loader.drop_last is False


def test_brats_provider_2023_get_patient_info_with_label(tmp_path: Path) -> None:
    from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

    patient_id = "BraTS-GLI-00001-000"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir(parents=True)
    for suffix in ["-t1n.nii.gz", "-t1c.nii.gz", "-t2w.nii.gz", "-t2f.nii.gz", "-seg.nii.gz"]:
        _create_empty_nii(patient_dir / f"{patient_id}{suffix}")

    provider = BraTSDataProvider(
        data_dir=tmp_path,
        version="2023",
        tumor_types=["GLI"],
        task="segmentation",
        num_workers=0,
    )

    info = provider.get_patient_info(patient_id)
    assert info["patient_id"] == patient_id
    assert "label" in info["file_paths"]


def test_brats_provider_validation_edges_and_discovery_branches(tmp_path: Path) -> None:
    from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

    gli = tmp_path / "gli"
    men = tmp_path / "men"
    met = tmp_path / "met"
    gli.mkdir()
    men.mkdir()
    met.mkdir()

    with pytest.raises(ValueError, match="between 0 and 1"):
        BraTSDataProvider(
            data_dir=tmp_path,
            version="2023",
            train_ratio=-0.1,
            val_ratio=0.55,
            test_ratio=0.55,
        )

    with pytest.raises(ValueError, match="spatial_size must be a 3-tuple"):
        BraTSDataProvider(data_dir=tmp_path, version="2023", spatial_size=(4, 4))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="data_dir must be provided when version is '2021'"):
        BraTSDataProvider(
            data_dir=None,
            data_dirs={"GLI": gli, "MEN": men, "MET": met},
            version="2021",
        )

    sample_file = gli / "BraTS-GLI-00001-000"
    sample_file.write_text("not a directory")
    with pytest.raises(FileNotFoundError, match="No valid BraTS 2023 patient folders"):
        BraTSDataProvider(data_dirs={"GLI": gli, "MEN": men, "MET": met})


def test_brats_provider_get_patient_info_raises_when_2021_data_dir_unset(tmp_path: Path) -> None:
    from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

    patient_id = "BraTS2021_00001"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir(parents=True)
    _create_empty_nii(patient_dir / f"{patient_id}_t1.nii.gz")

    provider = BraTSDataProvider(
        data_dir=tmp_path, version="2021", modalities=["t1"], num_workers=0
    )
    provider.data_dir = None

    with pytest.raises(ValueError, match="data_dir is not configured"):
        provider.get_patient_info(patient_id)
