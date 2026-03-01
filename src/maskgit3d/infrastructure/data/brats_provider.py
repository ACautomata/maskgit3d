"""
BraTS Data Provider for loading 3D MRI brain tumor segmentation data.

This module provides BraTSDataProvider for loading NIfTI format MRI data
from the BraTS (Brain Tumor Segmentation) dataset.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import nibabel as nib
import torch
from monai.transforms import Compose
from torch.utils.data import DataLoader, Dataset

from maskgit3d.domain.interfaces import DataProvider
from maskgit3d.infrastructure.data.transforms import (
    create_brats2023_preprocessing,
    create_brats_preprocessing,
)

logger = logging.getLogger(__name__)


# BraTS modality suffixes for file naming
BRATS_MODALITIES = {
    "t1": "_t1.nii.gz",
    "t1ce": "_t1ce.nii.gz",
    "t2": "_t2.nii.gz",
    "flair": "_flair.nii.gz",
}

# Default modality selection
DEFAULT_MODALITIES = ["t1", "t1ce", "t2", "flair"]


# BraTS 2023 modality suffixes
BRATS2023_MODALITIES = {
    "t1n": "-t1n.nii.gz",
    "t1c": "-t1c.nii.gz",
    "t2w": "-t2w.nii.gz",
    "t2f": "-t2f.nii.gz",
}

# Tumor type mapping for metadata tensor
TUMOR_TYPE_MAP = {
    "GLI": 0,
    "MEN": 1,
    "MET": 2,
}

# Valid tumor types
VALID_TUMOR_TYPES = list(TUMOR_TYPE_MAP.keys())


class BraTS2021Dataset(Dataset):
    """
    Dataset for loading BraTS MRI data in NIfTI format.

    Each patient folder contains multiple modalities (t1, t1ce, t2, flair)
    stored as NIfTI files with standardized naming conventions.
    """

    def __init__(
        self,
        data_dir: Path,
        patient_ids: List[str],
        modalities: List[str],
        transform: Optional[Compose] = None,
        spatial_size: Tuple[int, int, int] = (64, 64, 64),
    ):
        """
        Initialize BraTS dataset.

        Args:
            data_dir: Root directory containing patient folders
            patient_ids: List of patient identifiers (folder names)
            modalities: List of modalities to load (t1, t1ce, t2, flair)
            transform: Optional MONAI transforms to apply
            spatial_size: Target spatial dimensions (D, H, W)
        """
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.modalities = modalities
        self.transform = transform
        self.spatial_size = spatial_size

        # Validate modalities
        self._validate_modalities()

        # Build file paths for each patient
        self._file_paths = self._build_file_paths()

    def _validate_modalities(self) -> None:
        """Validate that all requested modalities are supported."""
        invalid = [m for m in self.modalities if m not in BRATS_MODALITIES]
        if invalid:
            raise ValueError(
                f"Invalid modalities: {invalid}. "
                f"Supported modalities are: {list(BRATS_MODALITIES.keys())}"
            )

    def _build_file_paths(self) -> Dict[str, Dict[str, Path]]:
        """
        Build dictionary mapping patient IDs to their modality file paths.

        Returns:
            Dictionary mapping patient_id -> modality -> file_path
        """
        file_paths = {}
        for patient_id in self.patient_ids:
            patient_dir = self.data_dir / patient_id
            if not patient_dir.exists():
                logger.warning(f"Patient directory not found: {patient_dir}")
                continue

            file_paths[patient_id] = {}
            for modality in self.modalities:
                suffix = BRATS_MODALITIES[modality]
                # Try different naming patterns
                patterns = [
                    patient_dir / f"{patient_id}{suffix}",
                    patient_dir / f"{patient_id}_{modality}.nii.gz",
                ]

                found = False
                for pattern in patterns:
                    if pattern.exists():
                        file_paths[patient_id][modality] = pattern
                        found = True
                        break

                if not found:
                    logger.warning(
                        f"File not found for patient {patient_id}, "
                        f"modality {modality}. Tried: {patterns}"
                    )

        return file_paths

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self._file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Index of sample to retrieve

        Returns:
            Tuple of (input_tensor, target_tensor) where input_tensor has
            shape [C, D, H, W] and target_tensor is the same for reconstruction
            tasks.
        """
        patient_id = list(self._file_paths.keys())[idx]
        modality_paths = self._file_paths[patient_id]

        # Load all modalities
        volumes = []
        for modality in self.modalities:
            if modality not in modality_paths:
                # Return zeros if file not found
                volume = torch.zeros(self.spatial_size, dtype=torch.float32)
                logger.warning(
                    f"Returning zeros for missing modality {modality} in patient {patient_id}"
                )
            else:
                volume = self._load_nifti(modality_paths[modality])

            volumes.append(volume)

        # Stack modalities as channels: [C, D, H, W]
        input_tensor = torch.stack(volumes, dim=0)

        # Apply transforms if available
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)

        # For reconstruction tasks, target = input
        target_tensor = input_tensor.clone()

        return input_tensor, target_tensor

    def _load_nifti(self, file_path: Path) -> torch.Tensor:
        """
        Load a NIfTI file and convert to tensor.

        Args:
            file_path: Path to the NIfTI file

        Returns:
            Tensor with shape [D, H, W]
        """
        nifti_img = nib.load(str(file_path))
        data = nifti_img.get_fdata()
        return torch.from_numpy(data).float()


BraTSDataset = BraTS2021Dataset


class BraTS2023Dataset(Dataset):
    """Dataset for BraTS 2023 format using MONAI dictionary transforms."""

    def __init__(
        self,
        data_dicts: List[Dict[str, Any]],
        transform: Optional[Compose] = None,
        task: str = "reconstruction",
    ):
        """
        Args:
            data_dicts: List of dicts with keys:
                - "image": list of 4 file paths [t1n, t1c, t2w, t2f]
                - "label": path to seg file (only for segmentation task)
                - "tumor_type": int (0=GLI, 1=MEN, 2=MET)
            transform: MONAI dictionary transform pipeline
            task: "reconstruction" or "segmentation"
        """
        self.data_dicts = data_dicts
        self.transform = transform
        self.task = task

    def __len__(self) -> int:
        return len(self.data_dicts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = dict(self.data_dicts[idx])

        if self.transform is not None:
            data = self.transform(data)

        image = data["image"]
        tumor_type = torch.tensor(data["tumor_type"], dtype=torch.long)

        if self.task == "segmentation":
            target = data["label"]
        else:
            target = image.clone()

        return image, target, tumor_type


class BraTSDataProvider(DataProvider):
    """
    Data provider for BraTS (Brain Tumor Segmentation) dataset.

    Loads NIfTI format MRI data with support for multiple modalities
    (t1, t1ce, t2, flair) and automatic train/val/test splitting.

    The BraTS dataset structure should be:
        data_dir/
        ├── BraTS2021_00001/
        │   ├── BraTS2021_00001_t1.nii.gz
        │   ├── BraTS2021_00001_t1ce.nii.gz
        │   ├── BraTS2021_00001_t2.nii.gz
        │   └── BraTS2021_00001_flair.nii.gz
        ├── BraTS2021_00002/
        │   └── ...
        └── ...

    Example usage:
        >>> provider = BraTSDataProvider(
        ...     data_dir="/path/to/brats/data",
        ...     modalities=["t1", "t1ce", "t2", "flair"],
        ...     spatial_size=(64, 64, 64),
        ...     batch_size=2,
        ... )
        >>> for batch_x, batch_y in provider.train_loader():
        ...     # Process batch
        ...     pass
    """

    # Download information for user guidance
    BRATS_DOWNLOAD_INFO = """
    BraTS Dataset Download Instructions:
    ====================================

    The BraTS (Brain Tumor Segmentation) dataset can be obtained from:

    1. Official Website (requires registration):
       https://www.synapse.org/#!Synapse:syn25829067

       - Create a free account
       - Register for the BraTS challenge
       - Download the training data

    2. Alternative Sources:
       - Kaggle: https://www.kaggle.com/datasets/awsaf49/brats2021-training-data
       - Medical Segmentation Decathlon: http://medicaldecathlon.com/

    3. After downloading, extract and organize:
       tar -xzf BraTS2021_Training_Data.tar.gz

       The expected directory structure is:
       data_dir/
       ├── BraTS2021_00001/
       │   ├── BraTS2021_00001_t1.nii.gz
       │   ├── BraTS2021_00001_t1ce.nii.gz
       │   ├── BraTS2021_00001_t2.nii.gz
       │   └── BraTS2021_00001_flair.nii.gz
       └── ...

    4. Set the data_dir parameter to point to the extracted directory.
    """

    def __init__(
        self,
        data_dir: Union[str, Path, None] = None,
        modalities: Optional[List[str]] = None,
        spatial_size: Tuple[int, int, int] = (64, 64, 64),
        batch_size: int = 1,
        num_workers: int = 4,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        normalize_mode: str = "zscore",
        version: str = "2023",
        task: str = "reconstruction",
        tumor_types: Optional[List[str]] = None,
        data_dirs: Optional[Dict[str, Union[str, Path]]] = None,
    ):
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.modalities = modalities or DEFAULT_MODALITIES
        self.spatial_size = spatial_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.normalize_mode = normalize_mode
        self.version = version
        self.task = task
        self.tumor_types = [tumor_type.upper() for tumor_type in (tumor_types or VALID_TUMOR_TYPES)]
        self.data_dirs = (
            {tumor_type.upper(): Path(path) for tumor_type, path in data_dirs.items()}
            if data_dirs is not None
            else None
        )

        self._validate_inputs()

        if self.version == "2021":
            self.task = "reconstruction"
            self.transform = create_brats_preprocessing(
                spatial_size=spatial_size,
                normalize_mode=normalize_mode,
            )
            self._all_samples: List[Union[str, Dict[str, Any]]] = self._discover_patients_2021()
            self._train_samples, self._val_samples, self._test_samples = self._split_patients(
                [sample for sample in self._all_samples if isinstance(sample, str)]
            )
        else:
            self.transform = create_brats2023_preprocessing(
                spatial_size=spatial_size,
                normalize_mode=normalize_mode,
                task=self.task,
            )
            self._all_samples = self._discover_patients_2023()
            self._train_samples, self._val_samples, self._test_samples = (
                self._split_patients_stratified(
                    [sample for sample in self._all_samples if isinstance(sample, dict)]
                )
            )

        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None

    def _validate_inputs(self) -> None:
        """Validate all input parameters."""
        if self.data_dir is not None and self.data_dirs is not None:
            raise ValueError("data_dir and data_dirs are mutually exclusive")

        if self.data_dir is None and self.data_dirs is None:
            raise ValueError("At least one of data_dir or data_dirs must be provided")

        if self.version not in ("2021", "2023"):
            raise ValueError(f"version must be '2021' or '2023', got '{self.version}'")

        if self.task not in ("reconstruction", "segmentation"):
            raise ValueError(f"task must be 'reconstruction' or 'segmentation', got '{self.task}'")

        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= ratio_sum <= 1.01):
            raise ValueError(
                f"Ratios must sum to 1.0, got: train={self.train_ratio}, "
                f"val={self.val_ratio}, test={self.test_ratio}, sum={ratio_sum}"
            )

        for name, ratio in [
            ("train", self.train_ratio),
            ("val", self.val_ratio),
            ("test", self.test_ratio),
        ]:
            if ratio < 0 or ratio > 1:
                raise ValueError(f"Invalid {name}_ratio: {ratio}. Must be between 0 and 1.")

        if len(self.spatial_size) != 3:
            raise ValueError(f"spatial_size must be a 3-tuple, got: {self.spatial_size}")

        if self.version == "2021":
            if self.data_dir is None:
                raise ValueError("data_dir must be provided when version is '2021'")

            if not self.data_dir.exists():
                raise FileNotFoundError(
                    f"BraTS data directory not found: {self.data_dir}\n{self.BRATS_DOWNLOAD_INFO}"
                )

            invalid = [m for m in self.modalities if m not in BRATS_MODALITIES]
            if invalid:
                raise ValueError(
                    f"Invalid modalities: {invalid}. "
                    f"Supported modalities are: {list(BRATS_MODALITIES.keys())}"
                )
            return

        invalid_tumor_types = [
            tumor_type for tumor_type in self.tumor_types if tumor_type not in VALID_TUMOR_TYPES
        ]
        if invalid_tumor_types:
            raise ValueError(
                f"Invalid tumor_types: {invalid_tumor_types}. "
                f"Supported tumor types are: {VALID_TUMOR_TYPES}"
            )

        if self.data_dir is not None and not self.data_dir.exists():
            raise FileNotFoundError(f"BraTS data directory not found: {self.data_dir}")

        if self.data_dirs is not None:
            if not self.data_dirs:
                raise ValueError("data_dirs cannot be empty")

            invalid_keys = [key for key in self.data_dirs if key not in VALID_TUMOR_TYPES]
            if invalid_keys:
                raise ValueError(
                    f"Invalid data_dirs keys: {invalid_keys}. "
                    f"Supported tumor types are: {VALID_TUMOR_TYPES}"
                )

            missing_keys = [
                tumor_type for tumor_type in self.tumor_types if tumor_type not in self.data_dirs
            ]
            if missing_keys:
                raise ValueError(f"Missing data_dirs entries for tumor types: {missing_keys}")

            for tumor_type in self.tumor_types:
                if not self.data_dirs[tumor_type].exists():
                    raise FileNotFoundError(
                        f"BraTS data directory for {tumor_type} not found: {self.data_dirs[tumor_type]}"
                    )

    def _discover_patients_2021(self) -> List[str]:
        """Discover BraTS 2021 patient folders in the data directory."""
        if self.data_dir is None:
            return []

        patient_ids: List[str] = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and list(item.glob("*.nii.gz")):
                patient_ids.append(item.name)

        if not patient_ids:
            raise FileNotFoundError(
                f"No patient folders with NIfTI files found in: {self.data_dir}\n"
                f"Expected structure:\n"
                f"  {self.data_dir}/\n"
                f"  ├── Patient_001/\n"
                f"  │   ├── Patient_001_t1.nii.gz\n"
                f"  │   ├── Patient_001_t1ce.nii.gz\n"
                f"  │   ├── Patient_001_t2.nii.gz\n"
                f"  │   └── Patient_001_flair.nii.gz\n"
                f"  └── ...\n\n"
                f"{self.BRATS_DOWNLOAD_INFO}"
            )

        logger.info("Discovered %s BraTS 2021 patients", len(patient_ids))
        return sorted(patient_ids)

    def _build_patient_dict_2023(
        self,
        patient_dir: Path,
        tumor_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Build BraTS 2023 sample dictionary for one patient."""
        patient_id = patient_dir.name
        modality_paths: List[str] = []

        for suffix in BRATS2023_MODALITIES.values():
            modality_path = patient_dir / f"{patient_id}{suffix}"
            if not modality_path.exists():
                logger.warning(
                    "Skipping patient %s due to missing modality file: %s",
                    patient_id,
                    modality_path,
                )
                return None
            modality_paths.append(str(modality_path))

        sample: Dict[str, Any] = {
            "image": modality_paths,
            "tumor_type": TUMOR_TYPE_MAP[tumor_type],
        }

        if self.task == "segmentation":
            label_path = patient_dir / f"{patient_id}-seg.nii.gz"
            if not label_path.exists():
                logger.warning(
                    "Skipping patient %s due to missing segmentation label: %s",
                    patient_id,
                    label_path,
                )
                return None
            sample["label"] = str(label_path)

        return sample

    def _discover_patients_2023(self) -> List[Dict[str, Any]]:
        """Discover BraTS 2023 patients and build MONAI dictionary samples."""
        patient_dicts: List[Dict[str, Any]] = []

        if self.data_dir is not None:
            for tumor_type in self.tumor_types:
                for patient_dir in sorted(self.data_dir.glob(f"BraTS-{tumor_type}-*")):
                    if not patient_dir.is_dir():
                        continue
                    sample = self._build_patient_dict_2023(patient_dir, tumor_type)
                    if sample is not None:
                        patient_dicts.append(sample)
        elif self.data_dirs is not None:
            for tumor_type in self.tumor_types:
                tumor_dir = self.data_dirs[tumor_type]
                for patient_dir in sorted(tumor_dir.iterdir()):
                    if not patient_dir.is_dir():
                        continue
                    if not patient_dir.name.startswith(f"BraTS-{tumor_type}-"):
                        continue
                    sample = self._build_patient_dict_2023(patient_dir, tumor_type)
                    if sample is not None:
                        patient_dicts.append(sample)

        if not patient_dicts:
            source = str(self.data_dir) if self.data_dir is not None else str(self.data_dirs)
            raise FileNotFoundError(f"No valid BraTS 2023 patient folders found in: {source}")

        patient_dicts.sort(key=lambda sample: sample["image"][0])
        logger.info("Discovered %s BraTS 2023 patients", len(patient_dicts))
        return patient_dicts

    def _split_patients(self, patient_ids: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Split patient IDs into train, validation, and test sets."""
        rng = random.Random(self.random_seed)
        shuffled_ids = patient_ids.copy()
        rng.shuffle(shuffled_ids)

        n_total = len(shuffled_ids)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_ids = shuffled_ids[:n_train]
        val_ids = shuffled_ids[n_train : n_train + n_val]
        test_ids = shuffled_ids[n_train + n_val :]

        logger.info(
            "Split patients: train=%s, val=%s, test=%s",
            len(train_ids),
            len(val_ids),
            len(test_ids),
        )
        return train_ids, val_ids, test_ids

    def _split_patients_stratified(
        self,
        patient_dicts: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split samples by tumor type with stratified train/val/test ratios."""
        rng = random.Random(self.random_seed)
        grouped: Dict[int, List[Dict[str, Any]]] = {
            TUMOR_TYPE_MAP["GLI"]: [],
            TUMOR_TYPE_MAP["MEN"]: [],
            TUMOR_TYPE_MAP["MET"]: [],
        }

        for sample in patient_dicts:
            grouped[sample["tumor_type"]].append(sample)

        train_samples: List[Dict[str, Any]] = []
        val_samples: List[Dict[str, Any]] = []
        test_samples: List[Dict[str, Any]] = []

        for tumor_type, samples in grouped.items():
            samples_copy = samples.copy()
            rng.shuffle(samples_copy)

            n_total = len(samples_copy)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)

            train_samples.extend(samples_copy[:n_train])
            val_samples.extend(samples_copy[n_train : n_train + n_val])
            test_samples.extend(samples_copy[n_train + n_val :])

            logger.info(
                "Stratified split for tumor_type=%s: train=%s, val=%s, test=%s",
                tumor_type,
                n_train,
                n_val,
                n_total - n_train - n_val,
            )

        rng.shuffle(train_samples)
        rng.shuffle(val_samples)
        rng.shuffle(test_samples)
        return train_samples, val_samples, test_samples

    @property
    def train_dataset(self) -> Dataset:
        """Get or create training dataset."""
        if self._train_dataset is None:
            if self.version == "2021":
                self._train_dataset = BraTS2021Dataset(
                    data_dir=self.data_dir,
                    patient_ids=self._train_samples,
                    modalities=self.modalities,
                    transform=self.transform,
                    spatial_size=self.spatial_size,
                )
            else:
                self._train_dataset = BraTS2023Dataset(
                    data_dicts=self._train_samples,
                    transform=self.transform,
                    task=self.task,
                )
        return self._train_dataset

    @property
    def val_dataset(self) -> Dataset:
        """Get or create validation dataset."""
        if self._val_dataset is None:
            if self.version == "2021":
                self._val_dataset = BraTS2021Dataset(
                    data_dir=self.data_dir,
                    patient_ids=self._val_samples,
                    modalities=self.modalities,
                    transform=self.transform,
                    spatial_size=self.spatial_size,
                )
            else:
                self._val_dataset = BraTS2023Dataset(
                    data_dicts=self._val_samples,
                    transform=self.transform,
                    task=self.task,
                )
        return self._val_dataset

    @property
    def test_dataset(self) -> Dataset:
        """Get or create test dataset."""
        if self._test_dataset is None:
            if self.version == "2021":
                self._test_dataset = BraTS2021Dataset(
                    data_dir=self.data_dir,
                    patient_ids=self._test_samples,
                    modalities=self.modalities,
                    transform=self.transform,
                    spatial_size=self.spatial_size,
                )
            else:
                self._test_dataset = BraTS2023Dataset(
                    data_dicts=self._test_samples,
                    transform=self.transform,
                    task=self.task,
                )
        return self._test_dataset

    def train_loader(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Get training data loader."""
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return iter(loader)

    def val_loader(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Get validation data loader."""
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return iter(loader)

    def test_loader(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Get test data loader."""
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return iter(loader)

    @property
    def num_modalities(self) -> int:
        """Return number of modalities."""
        if self.version == "2023":
            return len(BRATS2023_MODALITIES)
        return len(self.modalities)

    @property
    def num_train_samples(self) -> int:
        """Return number of training samples."""
        return len(self._train_samples)

    @property
    def num_val_samples(self) -> int:
        """Return number of validation samples."""
        return len(self._val_samples)

    @property
    def num_test_samples(self) -> int:
        """Return number of test samples."""
        return len(self._test_samples)

    def get_patient_info(self, patient_id: str) -> Dict[str, Any]:
        """Get information about a specific patient."""
        if self.version == "2021":
            if self.data_dir is None:
                raise ValueError("data_dir is not configured for BraTS 2021")

            patient_dir = self.data_dir / patient_id
            if not patient_dir.exists():
                raise ValueError(f"Patient not found: {patient_id}")

            info = {
                "patient_id": patient_id,
                "available_modalities": [],
                "file_paths": {},
            }

            for modality in self.modalities:
                suffix = BRATS_MODALITIES[modality]
                patterns = [
                    patient_dir / f"{patient_id}{suffix}",
                    patient_dir / f"{patient_id}_{modality}.nii.gz",
                ]

                for pattern in patterns:
                    if pattern.exists():
                        info["available_modalities"].append(modality)
                        info["file_paths"][modality] = str(pattern)
                        break

            return info

        for sample in self._all_samples:
            if isinstance(sample, dict):
                sample_patient_id = Path(sample["image"][0]).parent.name
                if sample_patient_id == patient_id:
                    info: Dict[str, Any] = {
                        "patient_id": patient_id,
                        "tumor_type": sample["tumor_type"],
                        "file_paths": {"image": sample["image"]},
                    }
                    if "label" in sample:
                        info["file_paths"]["label"] = sample["label"]
                    return info

        raise ValueError(f"Patient not found: {patient_id}")
