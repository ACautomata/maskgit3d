"""
BraTS Data Provider for loading 3D MRI brain tumor segmentation data.

This module provides BraTSDataProvider for loading NIfTI format MRI data
from the BraTS (Brain Tumor Segmentation) dataset.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import logging

import nibabel as nib
import torch
from torch.utils.data import DataLoader, Dataset
from monai.transforms import Compose

from maskgit3d.domain.interfaces import DataProvider
from maskgit3d.infrastructure.data.transforms import create_brats_preprocessing


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


class BraTSDataset(Dataset):
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
                    f"Returning zeros for missing modality {modality} "
                    f"in patient {patient_id}"
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
        data_dir: Union[str, Path],
        modalities: Optional[List[str]] = None,
        spatial_size: Tuple[int, int, int] = (64, 64, 64),
        batch_size: int = 1,
        num_workers: int = 4,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        normalize_mode: str = "zscore",
    ):
        """
        Initialize BraTS data provider.

        Args:
            data_dir: Root directory containing patient folders
            modalities: List of modalities to load. If None, uses all
                available modalities: ["t1", "t1ce", "t2", "flair"]
            spatial_size: Target spatial dimensions (D, H, W)
            batch_size: Batch size for data loaders
            num_workers: Number of data loading workers
            train_ratio: Ratio of data for training (default: 0.7)
            val_ratio: Ratio of data for validation (default: 0.15)
            test_ratio: Ratio of data for testing (default: 0.15)
            random_seed: Random seed for reproducible splits
            normalize_mode: Normalization mode ("minmax" or "zscore")

        Raises:
            FileNotFoundError: If data_dir does not exist
            ValueError: If modalities are invalid or ratios don't sum to 1.0
        """
        self.data_dir = Path(data_dir)
        self.modalities = modalities or DEFAULT_MODALITIES
        self.spatial_size = spatial_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.normalize_mode = normalize_mode

        # Validate inputs
        self._validate_inputs()

        # Create preprocessing transform
        self.transform = create_brats_preprocessing(
            spatial_size=spatial_size,
            normalize_mode=normalize_mode,
        )

        # Discover patient folders and create datasets
        self._patient_ids = self._discover_patients()
        self._train_ids, self._val_ids, self._test_ids = self._split_patients()

        # Create datasets
        self._train_dataset: Optional[BraTSDataset] = None
        self._val_dataset: Optional[BraTSDataset] = None
        self._test_dataset: Optional[BraTSDataset] = None

    def _validate_inputs(self) -> None:
        """Validate all input parameters."""
        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"BraTS data directory not found: {self.data_dir}\n"
                f"{self.BRATS_DOWNLOAD_INFO}"
            )

        # Validate modalities
        invalid = [m for m in self.modalities if m not in BRATS_MODALITIES]
        if invalid:
            raise ValueError(
                f"Invalid modalities: {invalid}. "
                f"Supported modalities are: {list(BRATS_MODALITIES.keys())}"
            )

        # Validate ratios
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= ratio_sum <= 1.01):  # Allow small floating point errors
            raise ValueError(
                f"Ratios must sum to 1.0, got: train={self.train_ratio}, "
                f"val={self.val_ratio}, test={self.test_ratio}, sum={ratio_sum}"
            )

        # Validate individual ratios
        for name, ratio in [
            ("train", self.train_ratio),
            ("val", self.val_ratio),
            ("test", self.test_ratio),
        ]:
            if ratio < 0 or ratio > 1:
                raise ValueError(
                    f"Invalid {name}_ratio: {ratio}. Must be between 0 and 1."
                )

        # Validate spatial size
        if len(self.spatial_size) != 3:
            raise ValueError(
                f"spatial_size must be a 3-tuple, got: {self.spatial_size}"
            )

    def _discover_patients(self) -> List[str]:
        """
        Discover patient folders in the data directory.

        Returns:
            List of patient IDs (folder names)

        Raises:
            FileNotFoundError: If no patient folders are found
        """
        patient_ids = []

        for item in self.data_dir.iterdir():
            if item.is_dir():
                # Check if directory contains NIfTI files
                nii_files = list(item.glob("*.nii.gz"))
                if nii_files:
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

        logger.info(f"Discovered {len(patient_ids)} patients in {self.data_dir}")
        return sorted(patient_ids)

    def _split_patients(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Split patient IDs into train, validation, and test sets.

        Returns:
            Tuple of (train_ids, val_ids, test_ids)
        """
        import random

        # Set random seed for reproducibility
        rng = random.Random(self.random_seed)

        # Shuffle patient IDs
        shuffled_ids = self._patient_ids.copy()
        rng.shuffle(shuffled_ids)

        # Calculate split indices
        n_total = len(shuffled_ids)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        # Split
        train_ids = shuffled_ids[:n_train]
        val_ids = shuffled_ids[n_train : n_train + n_val]
        test_ids = shuffled_ids[n_train + n_val :]

        logger.info(
            f"Split patients: train={len(train_ids)}, "
            f"val={len(val_ids)}, test={len(test_ids)}"
        )

        return train_ids, val_ids, test_ids

    @property
    def train_dataset(self) -> BraTSDataset:
        """Get or create training dataset."""
        if self._train_dataset is None:
            self._train_dataset = BraTSDataset(
                data_dir=self.data_dir,
                patient_ids=self._train_ids,
                modalities=self.modalities,
                transform=self.transform,
                spatial_size=self.spatial_size,
            )
        return self._train_dataset

    @property
    def val_dataset(self) -> BraTSDataset:
        """Get or create validation dataset."""
        if self._val_dataset is None:
            self._val_dataset = BraTSDataset(
                data_dir=self.data_dir,
                patient_ids=self._val_ids,
                modalities=self.modalities,
                transform=self.transform,
                spatial_size=self.spatial_size,
            )
        return self._val_dataset

    @property
    def test_dataset(self) -> BraTSDataset:
        """Get or create test dataset."""
        if self._test_dataset is None:
            self._test_dataset = BraTSDataset(
                data_dir=self.data_dir,
                patient_ids=self._test_ids,
                modalities=self.modalities,
                transform=self.transform,
                spatial_size=self.spatial_size,
            )
        return self._test_dataset

    def train_loader(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get training data loader.

        Returns:
            Iterator over training batches of shape [B, C, D, H, W]
            where C is the number of modalities.
        """
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return iter(loader)

    def val_loader(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get validation data loader.

        Returns:
            Iterator over validation batches of shape [B, C, D, H, W]
        """
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return iter(loader)

    def test_loader(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get test data loader.

        Returns:
            Iterator over test batches of shape [B, C, D, H, W]
        """
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
        return len(self.modalities)

    @property
    def num_train_samples(self) -> int:
        """Return number of training samples."""
        return len(self._train_ids)

    @property
    def num_val_samples(self) -> int:
        """Return number of validation samples."""
        return len(self._val_ids)

    @property
    def num_test_samples(self) -> int:
        """Return number of test samples."""
        return len(self._test_ids)

    def get_patient_info(self, patient_id: str) -> Dict[str, Any]:
        """
        Get information about a specific patient.

        Args:
            patient_id: Patient identifier

        Returns:
            Dictionary with patient information
        """
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