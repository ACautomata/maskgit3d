"""
Tests for data providers: MedMnist3DDataProvider and BraTSDataProvider.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from maskgit3d.infrastructure.data.transforms import (
    create_3d_preprocessing,
    create_brats2023_preprocessing,
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

    def test_create_brats2023_preprocessing_reconstruction(self):
        """Test creating BraTS 2023 preprocessing pipeline for reconstruction."""
        transform = create_brats2023_preprocessing(
            spatial_size=(64, 64, 64),
            normalize_mode="zscore",
            task="reconstruction",
        )
        assert transform is not None

    def test_create_brats2023_preprocessing_segmentation(self):
        """Test creating BraTS 2023 preprocessing pipeline for segmentation."""
        transform = create_brats2023_preprocessing(
            spatial_size=(64, 64, 64),
            normalize_mode="minmax",
            task="segmentation",
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
                    version="2021",
                    modalities=["invalid_modality"],
                )

    def test_invalid_ratios(self):
        """Test that ratios not summing to 1.0 raise ValueError."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            patient_dir = Path(tmpdir) / "BraTS2021_00001"
            patient_dir.mkdir()

            # Create a dummy NIfTI file
            import nibabel
            import numpy as np

            data = np.random.rand(10, 10, 10).astype(np.float32)
            affine = np.eye(4)
            img = nibabel.Nifti1Image(data, affine)  # type: ignore[attr-defined]
            nibabel.save(img, str(patient_dir / "BraTS2021_00001_t1.nii.gz"))  # type: ignore[attr-defined]

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

            import nibabel
            import numpy as np

            data = np.random.rand(10, 10, 10).astype(np.float32)
            affine = np.eye(4)
            img = nibabel.Nifti1Image(data, affine)  # type: ignore[attr-defined]
            nibabel.save(img, str(patient_dir / "BraTS2021_00001_t1.nii.gz"))  # type: ignore[attr-defined]

            provider = BraTSDataProvider(
                data_dir=tmpdir,
                version="2021",
                modalities=["t1"],
                spatial_size=(32, 32, 32),
                batch_size=1,
            )

            assert provider.num_modalities == 1
            assert (
                provider.num_train_samples + provider.num_val_samples + provider.num_test_samples
                == 1
            )


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
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            patient_dir = Path(tmpdir) / "BraTS2021_00001"
            patient_dir.mkdir()

            # Create only T1 modality
            import nibabel
            import numpy as np

            data = np.random.rand(10, 10, 10).astype(np.float32)
            affine = np.eye(4)
            img = nibabel.Nifti1Image(data, affine)  # type: ignore[attr-defined]
            nibabel.save(img, str(patient_dir / "BraTS2021_00001_t1.nii.gz"))  # type: ignore[attr-defined]

            dataset = BraTSDataset(
                data_dir=Path(tmpdir),
                patient_ids=["BraTS2021_00001"],
                modalities=["t1", "t2"],  # T2 is missing
                spatial_size=(32, 32, 32),
            )

            # Should not raise, but log warning
            assert len(dataset) == 1


def _create_brats2023_patient(
    parent_dir: Path,
    patient_id: str,
    include_seg: bool = True,
) -> Path:
    """Create a fake BraTS2023 patient directory with minimal NIfTI files."""
    import nibabel
    import numpy as np

    patient_dir = parent_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    data = np.random.rand(4, 4, 4).astype(np.float32)
    affine = np.eye(4)

    for _suffix in ["-t1n.nii.gz", "-t1c.nii.gz", "-t2w.nii.gz", "-t2f.nii.gz"]:
        nibabel.save(nibabel.Nifti1Image(data, affine), str(patient_dir / f"{patient_id}{_suffix}"))  # type: ignore[attr-defined]

    if include_seg:
        seg_data = np.random.randint(0, 4, size=(4, 4, 4)).astype(np.float32)
        nibabel.save(nibabel.Nifti1Image(seg_data, affine), str(patient_dir / f"{patient_id}-seg.nii.gz"))  # type: ignore[attr-defined]

    return patient_dir


class TestBraTS2023Provider:
    """Tests for BraTSDataProvider with version='2023'."""

    def test_mutual_exclusivity_data_dir_and_data_dirs(self):
        """Test data_dir and data_dirs cannot both be provided."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            gli_dir = base / "gli"
            men_dir = base / "men"
            met_dir = base / "met"
            gli_dir.mkdir()
            men_dir.mkdir()
            met_dir.mkdir()

            with pytest.raises(ValueError, match="mutually exclusive"):
                BraTSDataProvider(
                    data_dir=base,
                    data_dirs={"GLI": gli_dir, "MEN": men_dir, "MET": met_dir},
                )

    def test_neither_data_dir_nor_data_dirs(self):
        """Test neither data_dir nor data_dirs raises ValueError."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with pytest.raises(ValueError, match="At least one of data_dir or data_dirs"):
            BraTSDataProvider(data_dir=None, data_dirs=None)

    def test_invalid_version(self):
        """Test invalid version raises ValueError."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_brats2023_patient(Path(tmpdir), "BraTS-GLI-00001-000")
            with pytest.raises(ValueError, match="version must be '2021' or '2023'"):
                BraTSDataProvider(data_dir=tmpdir, version="2022")

    def test_invalid_task(self):
        """Test invalid task raises ValueError."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_brats2023_patient(Path(tmpdir), "BraTS-GLI-00001-000")
            with pytest.raises(ValueError, match="task must be 'reconstruction' or 'segmentation'"):
                BraTSDataProvider(data_dir=tmpdir, task="invalid")

    def test_invalid_tumor_types(self):
        """Test invalid tumor types raise ValueError."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_brats2023_patient(Path(tmpdir), "BraTS-GLI-00001-000")
            with pytest.raises(ValueError, match="Invalid tumor_types"):
                BraTSDataProvider(data_dir=tmpdir, tumor_types=["INVALID"])

    def test_empty_data_dirs(self):
        """Test empty data_dirs raises ValueError."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with pytest.raises(ValueError, match="data_dirs cannot be empty"):
            BraTSDataProvider(data_dirs={})

    def test_invalid_data_dirs_keys(self):
        """Test invalid data_dirs keys raise ValueError."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(
                ValueError,
                match="Invalid data_dirs keys",
            ),
        ):
            BraTSDataProvider(data_dirs={"INVALID": Path(tmpdir)})

    def test_missing_data_dirs_entries(self):
        """Test missing data_dirs entries for selected tumor types raises ValueError."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            gli_dir = Path(tmpdir) / "gli"
            gli_dir.mkdir()

            with pytest.raises(ValueError, match="Missing data_dirs entries"):
                BraTSDataProvider(tumor_types=["GLI", "MEN"], data_dirs={"GLI": gli_dir})

    def test_v2023_discovery_with_data_dir(self):
        """Test BraTS 2023 patient discovery using data_dir."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_brats2023_patient(root, "BraTS-GLI-00001-000")
            _create_brats2023_patient(root, "BraTS-GLI-00002-000")

            provider = BraTSDataProvider(data_dir=root, tumor_types=["GLI"])

            assert len(provider._all_samples) == 2
            assert all(len(sample["image"]) == 4 for sample in provider._all_samples)  # type: ignore[index]

    def test_v2023_discovery_with_data_dirs(self):
        """Test BraTS 2023 patient discovery using data_dirs mapping."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gli_dir = root / "gli"
            men_dir = root / "men"
            met_dir = root / "met"
            gli_dir.mkdir()
            men_dir.mkdir()
            met_dir.mkdir()

            _create_brats2023_patient(gli_dir, "BraTS-GLI-00001-000")
            _create_brats2023_patient(men_dir, "BraTS-MEN-00001-000")
            _create_brats2023_patient(met_dir, "BraTS-MET-00001-000")

            provider = BraTSDataProvider(
                data_dirs={"GLI": gli_dir, "MEN": men_dir, "MET": met_dir},
            )

            assert len(provider._all_samples) == 3
            assert all(len(sample["image"]) == 4 for sample in provider._all_samples)  # type: ignore[index]

    def test_v2023_no_patients_found(self):
        """Test empty directory raises FileNotFoundError for BraTS 2023."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(
                FileNotFoundError,
                match="No valid BraTS 2023 patient folders",
            ),
        ):
            BraTSDataProvider(data_dir=tmpdir)

    def test_v2023_skips_missing_modality(self):
        """Test patient missing one modality is skipped."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_brats2023_patient(root, "BraTS-GLI-00001-000")
            missing = _create_brats2023_patient(root, "BraTS-GLI-00002-000")
            (missing / "BraTS-GLI-00002-000-t2f.nii.gz").unlink()

            provider = BraTSDataProvider(data_dir=root, tumor_types=["GLI"])

            assert len(provider._all_samples) == 1
            only_sample = provider._all_samples[0]
            assert "BraTS-GLI-00001-000" in only_sample["image"][0]  # type: ignore[index]

    def test_v2023_segmentation_requires_label(self):
        """Test segmentation task skips patient without label."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_brats2023_patient(root, "BraTS-GLI-00001-000", include_seg=True)
            _create_brats2023_patient(root, "BraTS-GLI-00002-000", include_seg=False)

            provider = BraTSDataProvider(
                data_dir=root,
                tumor_types=["GLI"],
                task="segmentation",
            )

            assert len(provider._all_samples) == 1
            assert "label" in provider._all_samples[0]

    def test_default_version_is_2023(self):
        """Test default provider version is 2023."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_brats2023_patient(Path(tmpdir), "BraTS-GLI-00001-000")
            provider = BraTSDataProvider(data_dir=tmpdir, tumor_types=["GLI"])
            assert provider.version == "2023"

    def test_tumor_type_filtering(self):
        """Test tumor type filtering during discovery."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_brats2023_patient(root, "BraTS-GLI-00001-000")
            _create_brats2023_patient(root, "BraTS-MEN-00001-000")

            provider = BraTSDataProvider(data_dir=root, tumor_types=["GLI"])

            assert len(provider._all_samples) == 1
            assert provider._all_samples[0]["tumor_type"] == 0  # type: ignore[index]


class TestBraTS2023StratifiedSplit:
    """Tests for BraTSDataProvider._split_patients_stratified."""

    @staticmethod
    def _make_sample_dicts(per_type: int):
        from maskgit3d.infrastructure.data.brats_provider import TUMOR_TYPE_MAP

        samples = []
        for tumor in ["GLI", "MEN", "MET"]:
            for idx in range(per_type):
                samples.append(
                    {
                        "image": [f"/tmp/BraTS-{tumor}-{idx:05d}-000-t1n.nii.gz"],
                        "tumor_type": TUMOR_TYPE_MAP[tumor],
                    }
                )
        return samples

    @staticmethod
    def _counts_by_type(samples):
        counts = {0: 0, 1: 0, 2: 0}
        for sample in samples:
            counts[sample["tumor_type"]] += 1
        return counts

    def test_stratified_split_proportions(self):
        """Test stratified split follows expected 70/15/15 proportions per tumor type."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_brats2023_patient(Path(tmpdir), "BraTS-GLI-00001-000")
            provider = BraTSDataProvider(data_dir=tmpdir, tumor_types=["GLI"])

            samples = self._make_sample_dicts(per_type=10)
            train, val, test = provider._split_patients_stratified(samples)

            train_counts = self._counts_by_type(train)
            val_counts = self._counts_by_type(val)
            test_counts = self._counts_by_type(test)

            for tumor_idx in [0, 1, 2]:
                assert abs(train_counts[tumor_idx] - 7) <= 1
                assert abs(val_counts[tumor_idx] - 1.5) <= 1
                assert abs(test_counts[tumor_idx] - 1.5) <= 1

    def test_stratified_split_deterministic(self):
        """Test stratified split is deterministic with same seed."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_brats2023_patient(Path(tmpdir), "BraTS-GLI-00001-000")
            samples = self._make_sample_dicts(per_type=10)

            provider_a = BraTSDataProvider(data_dir=tmpdir, tumor_types=["GLI"], random_seed=123)
            provider_b = BraTSDataProvider(data_dir=tmpdir, tumor_types=["GLI"], random_seed=123)

            split_a = provider_a._split_patients_stratified(samples)
            split_b = provider_b._split_patients_stratified(samples)

            assert split_a == split_b

    def test_stratified_split_different_seeds(self):
        """Test stratified split differs with different seeds."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_brats2023_patient(Path(tmpdir), "BraTS-GLI-00001-000")
            samples = self._make_sample_dicts(per_type=10)

            provider_a = BraTSDataProvider(data_dir=tmpdir, tumor_types=["GLI"], random_seed=123)
            provider_b = BraTSDataProvider(data_dir=tmpdir, tumor_types=["GLI"], random_seed=456)

            split_a = provider_a._split_patients_stratified(samples)
            split_b = provider_b._split_patients_stratified(samples)

            assert split_a != split_b

    def test_stratified_split_all_types_represented(self):
        """Test each split has all tumor types represented."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_brats2023_patient(Path(tmpdir), "BraTS-GLI-00001-000")
            provider = BraTSDataProvider(data_dir=tmpdir, tumor_types=["GLI"])

            samples = self._make_sample_dicts(per_type=10)
            train, val, test = provider._split_patients_stratified(samples)

            for split in (train, val, test):
                represented = {sample["tumor_type"] for sample in split}
                assert represented == {0, 1, 2}


class TestBraTS2023Dataset:
    """Tests for BraTS2023Dataset."""

    def test_dataset_length(self):
        """Test dataset __len__ matches number of data dicts."""
        from maskgit3d.infrastructure.data.brats_provider import BraTS2023Dataset

        data_dicts = [
            {"image": torch.zeros(4, 4, 4, 4), "tumor_type": 0},
            {"image": torch.zeros(4, 4, 4, 4), "tumor_type": 1},
        ]
        dataset = BraTS2023Dataset(data_dicts=data_dicts, transform=None)
        assert len(dataset) == len(data_dicts)

    def test_dataset_reconstruction_output(self):
        """Test reconstruction task returns image clone as target."""
        from maskgit3d.infrastructure.data.brats_provider import BraTS2023Dataset

        image = torch.rand(4, 8, 8, 8)
        data_dicts = [{"image": image, "tumor_type": 0}]
        dataset = BraTS2023Dataset(data_dicts=data_dicts, transform=None, task="reconstruction")

        output_image, target, _ = dataset[0]
        assert torch.equal(output_image, target)
        assert output_image is not target

    def test_dataset_segmentation_output(self):
        """Test segmentation task returns label as target."""
        from maskgit3d.infrastructure.data.brats_provider import BraTS2023Dataset

        image = torch.rand(4, 8, 8, 8)
        label = torch.randint(0, 2, (3, 8, 8, 8)).float()
        data_dicts = [{"image": image, "label": label, "tumor_type": 0}]
        dataset = BraTS2023Dataset(data_dicts=data_dicts, transform=None, task="segmentation")

        _, target, _ = dataset[0]
        assert torch.equal(target, label)

    def test_dataset_returns_three_element_tuple(self):
        """Test dataset returns (image, target, tumor_type)."""
        from maskgit3d.infrastructure.data.brats_provider import BraTS2023Dataset

        data_dicts = [{"image": torch.rand(4, 8, 8, 8), "tumor_type": 1}]
        dataset = BraTS2023Dataset(data_dicts=data_dicts, transform=None)

        sample = dataset[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 3

    def test_dataset_tumor_type_tensor(self):
        """Test tumor_type is returned as long tensor."""
        from maskgit3d.infrastructure.data.brats_provider import BraTS2023Dataset

        data_dicts = [{"image": torch.rand(4, 8, 8, 8), "tumor_type": 2}]
        dataset = BraTS2023Dataset(data_dicts=data_dicts, transform=None)

        _, _, tumor_type = dataset[0]
        assert torch.is_tensor(tumor_type)
        assert tumor_type.dtype == torch.long


class TestBraTS2023BackwardCompat:
    """Backward-compatibility tests for BraTS 2021 behavior."""

    def test_brats_dataset_alias(self):
        """Test BraTSDataset alias points to BraTS2021Dataset."""
        from maskgit3d.infrastructure.data.brats_provider import (
            BraTS2021Dataset,
            BraTSDataset,
        )

        assert BraTSDataset is BraTS2021Dataset

    def test_v2021_still_works(self):
        """Test BraTS 2021 provider initialization still works."""
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            patient_id = "BraTS2021_00001"
            patient_dir = Path(tmpdir) / patient_id
            patient_dir.mkdir()

            import nibabel
            import numpy as np

            data = np.random.rand(6, 6, 6).astype(np.float32)
            affine = np.eye(4)
            nibabel.save(nibabel.Nifti1Image(data, affine), str(patient_dir / f"{patient_id}_t1.nii.gz"))  # type: ignore[attr-defined]

            provider = BraTSDataProvider(
                data_dir=tmpdir,
                version="2021",
                modalities=["t1"],
                batch_size=1,
                num_workers=0,
            )

            assert provider.version == "2021"
            assert provider.task == "reconstruction"
            assert provider.num_modalities == 1

    def test_v2021_returns_two_element_tuple(self):
        """Test BraTS 2021 dataset returns (image, target) only."""
        from maskgit3d.infrastructure.data.brats_provider import BraTS2021Dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            patient_id = "BraTS2021_00001"
            patient_dir = Path(tmpdir) / patient_id
            patient_dir.mkdir()

            import nibabel
            import numpy as np

            data = np.random.rand(6, 6, 6).astype(np.float32)
            affine = np.eye(4)
            nibabel.save(nibabel.Nifti1Image(data, affine), str(patient_dir / f"{patient_id}_t1.nii.gz"))  # type: ignore[attr-defined]

            dataset = BraTS2021Dataset(
                data_dir=Path(tmpdir),
                patient_ids=[patient_id],
                modalities=["t1"],
                transform=None,
                spatial_size=(6, 6, 6),
            )

            sample = dataset[0]
            assert isinstance(sample, tuple)
            assert len(sample) == 2


class TestBraTS2023Constants:
    """Tests for BraTS 2023 constants."""

    def test_brats2023_modalities(self):
        """Test BRATS2023_MODALITIES contains expected modality keys."""
        from maskgit3d.infrastructure.data.brats_provider import BRATS2023_MODALITIES

        assert set(BRATS2023_MODALITIES.keys()) == {"t1n", "t1c", "t2w", "t2f"}

    def test_tumor_type_map(self):
        """Test tumor type map values."""
        from maskgit3d.infrastructure.data.brats_provider import TUMOR_TYPE_MAP

        assert TUMOR_TYPE_MAP == {"GLI": 0, "MEN": 1, "MET": 2}

    def test_valid_tumor_types(self):
        """Test valid tumor types list."""
        from maskgit3d.infrastructure.data.brats_provider import VALID_TUMOR_TYPES

        assert VALID_TUMOR_TYPES == ["GLI", "MEN", "MET"]
