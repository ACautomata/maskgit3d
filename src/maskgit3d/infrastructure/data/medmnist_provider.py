"""
MedMNIST 3D Data Provider for medical imaging datasets.

This module provides MedMnist3DDataProvider for loading 3D medical imaging
datasets from the MedMNIST collection, supporting various organ and structure
datasets with automatic download and preprocessing capabilities.
"""

from enum import Enum
from typing import Dict, Iterator, Optional, Tuple, Type

import torch
from torch.utils.data import DataLoader, Dataset
from monai.transforms import Compose

from maskgit3d.domain.interfaces import DataProvider
from maskgit3d.infrastructure.data.transforms import create_medmnist_preprocessing


class MedMNIST3DDataset(str, Enum):
    """Available MedMNIST 3D dataset types."""

    ORGAN = "organ"
    NODULE = "nodule"
    ADRENAL = "adrenal"
    VESSEL = "vessel"
    FRACTURE = "fracture"
    SYNAPSE = "synapse"


# Mapping from dataset type to MedMNIST class
DATASET_CLASS_MAP: Dict[MedMNIST3DDataset, Type] = {}

# Lazy import to avoid errors if medmnist is not installed
def _get_dataset_class(dataset_type: MedMNIST3DDataset) -> Type:
    """
    Get the MedMNIST dataset class for the specified type.

    Uses lazy importing to avoid import errors when medmnist is not available.

    Args:
        dataset_type: The type of MedMNIST 3D dataset

    Returns:
        The MedMNIST dataset class

    Raises:
        ImportError: If medmnist package is not installed
        ValueError: If dataset type is not supported
    """
    global DATASET_CLASS_MAP

    if not DATASET_CLASS_MAP:
        try:
            from medmnist import (
                OrganMNIST3D,
                NoduleMNIST3D,
                AdrenalMNIST3D,
                VesselMNIST3D,
                FractureMNIST3D,
                SynapseMNIST3D,
            )

            DATASET_CLASS_MAP = {
                MedMNIST3DDataset.ORGAN: OrganMNIST3D,
                MedMNIST3DDataset.NODULE: NoduleMNIST3D,
                MedMNIST3DDataset.ADRENAL: AdrenalMNIST3D,
                MedMNIST3DDataset.VESSEL: VesselMNIST3D,
                MedMNIST3DDataset.FRACTURE: FractureMNIST3D,
                MedMNIST3DDataset.SYNAPSE: SynapseMNIST3D,
            }
        except ImportError as e:
            raise ImportError(
                "medmnist package is required for MedMnist3DDataProvider. "
                "Install it with: pip install medmnist"
            ) from e

    if dataset_type not in DATASET_CLASS_MAP:
        raise ValueError(
            f"Unsupported dataset type: {dataset_type}. "
            f"Supported types: {list(MedMNIST3DDataset)}"
        )

    return DATASET_CLASS_MAP[dataset_type]


class MedMNIST3DDatasetWrapper(Dataset):
    """
    Wrapper for MedMNIST 3D datasets with preprocessing.

    Wraps MedMNIST datasets and applies preprocessing transforms to convert
    data from [0, 255] range to [-1, 1] and reshape to target spatial size.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Optional[Compose] = None,
        spatial_size: Tuple[int, int, int] = (64, 64, 64),
    ):
        """
        Initialize the dataset wrapper.

        Args:
            dataset: The underlying MedMNIST dataset
            transform: Optional MONAI transforms to apply
            spatial_size: Target spatial dimensions (D, H, W)
        """
        self.dataset = dataset
        self.transform = transform
        self.spatial_size = spatial_size

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample with preprocessing.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input_tensor, target_tensor) where input_tensor
            has shape [C, D, H, W] normalized to [-1, 1]
        """
        # MedMNIST returns (image, label) tuple
        # image shape: (D, H, W) for 28x28x28 or 64x64x64
        # label: classification label
        image, label = self.dataset[idx]

        # Convert to tensor if numpy array
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)

        # Apply preprocessing transforms
        if self.transform is not None:
            image = self.transform(image)

        # Ensure image has channel dimension [C, D, H, W]
        if image.dim() == 3:
            # Add channel dimension: (D, H, W) -> (1, D, H, W)
            image = image.unsqueeze(0)

        # Convert label to tensor
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        # For reconstruction/generation tasks, target = input
        # Return (input, target) where target is the same as input
        # This matches the expected format for VQGAN/MaskGIT training
        return image, image.clone()


class MedMnist3DDataProvider(DataProvider):
    """
    Data provider for MedMNIST 3D datasets.

    Supports loading various 3D medical imaging datasets from MedMNIST
    collection with automatic download and preprocessing.

    Supported datasets:
        - organ: OrganMNIST3D
        - nodule: NoduleMNIST3D
        - adrenal: AdrenalMNIST3D
        - vessel: VesselMNIST3D
        - fracture: FractureMNIST3D
        - synapse: SynapseMNIST3D

    Features:
        - Automatic dataset download
        - Support for 28x28x28 and 64x64x64 resolutions
        - Data normalization from [0, 255] to [-1, 1]
        - Configurable batch size and data loading workers

    Example:
        >>> provider = MedMnist3DDataProvider(
        ...     dataset_type="organ",
        ...     spatial_size=(64, 64, 64),
        ...     batch_size=16,
        ...     data_root="./data",
        ... )
        >>> for batch in provider.train_loader():
        ...     inputs, targets = batch
        ...     # inputs shape: [16, 1, 64, 64, 64]
    """

    # Supported input sizes from MedMNIST
    SUPPORTED_INPUT_SIZES = (28, 64)

    def __init__(
        self,
        dataset_type: str = "organ",
        spatial_size: Tuple[int, int, int] = (64, 64, 64),
        input_size: int = 28,
        batch_size: int = 16,
        num_workers: int = 4,
        data_root: str = "./data",
        download: bool = True,
        in_channels: int = 1,
        pin_memory: bool = True,
        drop_last_train: bool = True,
    ):
        """
        Initialize the MedMNIST 3D data provider.

        Args:
            dataset_type: Type of MedMNIST 3D dataset. One of:
                "organ", "nodule", "adrenal", "vessel", "fracture", "synapse"
            spatial_size: Target spatial dimensions (D, H, W) after preprocessing.
                Default is (64, 64, 64).
            input_size: Input size of MedMNIST dataset. Either 28 or 64.
                Default is 28 (28x28x28 resolution).
            batch_size: Batch size for data loaders. Default is 16.
            num_workers: Number of data loading workers. Default is 4.
            data_root: Root directory for dataset storage. Default is "./data".
            download: Whether to download dataset if not found. Default is True.
            in_channels: Number of input channels. Default is 1 (grayscale).
            pin_memory: Whether to pin memory for faster GPU transfer. Default True.
            drop_last_train: Whether to drop last batch in training. Default True.

        Raises:
            ValueError: If dataset_type is not supported
            ValueError: If input_size is not 28 or 64
            ImportError: If medmnist package is not installed
        """
        # Validate dataset type
        try:
            self.dataset_type = MedMNIST3DDataset(dataset_type.lower())
        except ValueError as e:
            raise ValueError(
                f"Unsupported dataset type: '{dataset_type}'. "
                f"Supported types: {[d.value for d in MedMNIST3DDataset]}"
            ) from e

        # Validate input size
        if input_size not in self.SUPPORTED_INPUT_SIZES:
            raise ValueError(
                f"input_size must be one of {self.SUPPORTED_INPUT_SIZES}, "
                f"got {input_size}"
            )

        self.spatial_size = spatial_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.download = download
        self.in_channels = in_channels
        self.pin_memory = pin_memory
        self.drop_last_train = drop_last_train

        # Create preprocessing transform
        self.transform = create_medmnist_preprocessing(
            spatial_size=spatial_size,
            input_size=input_size,
        )

        # Load datasets lazily
        self._train_dataset: Optional[MedMNIST3DDatasetWrapper] = None
        self._val_dataset: Optional[MedMNIST3DDatasetWrapper] = None
        self._test_dataset: Optional[MedMNIST3DDatasetWrapper] = None

    def _get_dataset_class(self) -> Type:
        """Get the MedMNIST dataset class for the configured type."""
        return _get_dataset_class(self.dataset_type)

    def _create_dataset(
        self,
        split: str,
    ) -> MedMNIST3DDatasetWrapper:
        """
        Create a MedMNIST dataset for the specified split.

        Args:
            split: Dataset split ("train", "val", or "test")

        Returns:
            Wrapped dataset with preprocessing transforms
        """
        dataset_class = self._get_dataset_class()

        # Create underlying MedMNIST dataset
        # MedMNIST uses 'val' for validation split
        medmnist_split = "val" if split == "val" else split

        base_dataset = dataset_class(
            root=self.data_root,
            split=medmnist_split,
            download=self.download,
        )

        # Wrap with preprocessing
        return MedMNIST3DDatasetWrapper(
            dataset=base_dataset,
            transform=self.transform,
            spatial_size=self.spatial_size,
        )

    @property
    def train_dataset(self) -> MedMNIST3DDatasetWrapper:
        """Get training dataset (lazy loaded)."""
        if self._train_dataset is None:
            self._train_dataset = self._create_dataset("train")
        return self._train_dataset

    @property
    def val_dataset(self) -> MedMNIST3DDatasetWrapper:
        """Get validation dataset (lazy loaded)."""
        if self._val_dataset is None:
            self._val_dataset = self._create_dataset("val")
        return self._val_dataset

    @property
    def test_dataset(self) -> MedMNIST3DDatasetWrapper:
        """Get test dataset (lazy loaded)."""
        if self._test_dataset is None:
            self._test_dataset = self._create_dataset("test")
        return self._test_dataset

    def train_loader(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get training data loader.

        Returns:
            Iterator over training batches. Each batch is a tuple of
            (input_tensor, target_tensor) with shapes [B, C, D, H, W]
            where values are normalized to [-1, 1].
        """
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last_train,
        )
        return iter(loader)

    def val_loader(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get validation data loader.

        Returns:
            Iterator over validation batches. Each batch is a tuple of
            (input_tensor, target_tensor) with shapes [B, C, D, H, W]
            where values are normalized to [-1, 1].
        """
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        return iter(loader)

    def test_loader(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get test data loader.

        Returns:
            Iterator over test batches. Each batch is a tuple of
            (input_tensor, target_tensor) with shapes [B, C, D, H, W]
            where values are normalized to [-1, 1].
        """
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        return iter(loader)

    def get_num_classes(self) -> int:
        """
        Get the number of classes in the dataset.

        Returns:
            Number of classification classes
        """
        # Access underlying MedMNIST dataset to get class info
        dataset_class = self._get_dataset_class()
        return getattr(dataset_class, "num_classes", 1)

    def get_dataset_info(self) -> Dict[str, object]:
        """
        Get information about the dataset.

        Returns:
            Dictionary containing dataset metadata including:
            - dataset_type: Type of MedMNIST dataset
            - spatial_size: Target spatial dimensions
            - input_size: Original input resolution
            - num_train: Number of training samples
            - num_val: Number of validation samples
            - num_test: Number of test samples
            - num_classes: Number of classification classes
        """
        dataset_class = self._get_dataset_class()

        return {
            "dataset_type": self.dataset_type.value,
            "spatial_size": self.spatial_size,
            "input_size": self.input_size,
            "num_train": len(self.train_dataset),
            "num_val": len(self.val_dataset),
            "num_test": len(self.test_dataset),
            "num_classes": getattr(dataset_class, "num_classes", 1),
            "in_channels": self.in_channels,
        }