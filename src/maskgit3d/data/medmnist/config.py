"""MedMNIST-3D configuration classes and enums."""

from dataclasses import dataclass
from enum import Enum


class MedMNISTDatasetName(str, Enum):
    """Supported MedMNIST-3D dataset types."""

    ORGAN = "organmnist3d"
    NODULE = "nodulemnist3d"
    ADRENAL = "adrenalmnist3d"
    VESSEL = "vesselmnist3d"
    FRACTURE = "fracturemnist3d"
    SYNAPSE = "synapsemnist3d"


class TaskType(str, Enum):
    """Task type for data loading."""

    RECONSTRUCTION = "reconstruction"
    CLASSIFICATION = "classification"


@dataclass
class MedMNISTConfig:
    """Configuration for MedMNIST-3D dataset.

    Attributes:
        dataset_name: Which MedMNIST-3D dataset to use
        task_type: Task type (reconstruction or classification)
        data_dir: Root directory for data storage
        download: Whether to download if data not present
        crop_size: Spatial crop size for training (D, H, W)
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory in DataLoader
        drop_last_train: Whether to drop last incomplete batch in training
    """

    # Required
    dataset_name: MedMNISTDatasetName

    # With defaults
    task_type: TaskType = TaskType.RECONSTRUCTION
    data_dir: str = "./data"
    download: bool = True
    crop_size: tuple[int, int, int] = (32, 32, 32)
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    drop_last_train: bool = True

    @property
    def num_classes(self) -> int:
        """Return number of classes for the dataset."""
        class_counts = {
            MedMNISTDatasetName.ORGAN: 11,
            MedMNISTDatasetName.NODULE: 2,
            MedMNISTDatasetName.ADRENAL: 2,
            MedMNISTDatasetName.VESSEL: 2,
            MedMNISTDatasetName.FRACTURE: 3,
            MedMNISTDatasetName.SYNAPSE: 1,  # Multi-label
        }
        return class_counts[self.dataset_name]

    @property
    def input_size(self) -> int:
        """Original input size for all MedMNIST-3D datasets."""
        return 28
