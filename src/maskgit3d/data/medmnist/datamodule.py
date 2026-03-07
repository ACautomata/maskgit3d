"""MedMNIST-3D DataModule for PyTorch Lightning."""

import logging
from typing import Optional, Tuple, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .config import MedMNISTConfig, MedMNISTDatasetName, TaskType
from .dataset import MedMNIST3DDataset

logger = logging.getLogger(__name__)


class MedMNIST3DDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for MedMNIST-3D datasets.

    Handles data loading, preprocessing, and DataLoader creation
    for training, validation, and testing.
    """

    def __init__(
        self,
        dataset_name: Union[str, MedMNISTDatasetName] = MedMNISTDatasetName.ORGAN,
        task_type: Union[str, TaskType] = TaskType.RECONSTRUCTION,
        data_dir: str = "./data",
        download: bool = True,
        crop_size: Tuple[int, int, int] = (32, 32, 32),
        roi_size: Optional[Tuple[int, int, int]] = None,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last_train: bool = True,
    ):
        """Initialize MedMNIST-3D DataModule.

        Args:
            dataset_name: Which MedMNIST-3D dataset to use
            task_type: Task type (reconstruction or classification)
            data_dir: Root directory for data storage
            download: Whether to download if data not present
            crop_size: Spatial crop size for training (D, H, W)
            roi_size: ROI size for inference, None means no crop
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory in DataLoader
            drop_last_train: Whether to drop last incomplete batch in training
        """
        super().__init__()

        # Convert string to enum if needed
        if isinstance(dataset_name, str):
            dataset_name = MedMNISTDatasetName(dataset_name)
        if isinstance(task_type, str):
            task_type = TaskType(task_type)

        self.config = MedMNISTConfig(
            dataset_name=dataset_name,
            task_type=task_type,
            data_dir=data_dir,
            download=download,
            crop_size=crop_size,
            roi_size=roi_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last_train=drop_last_train,
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset: Optional[MedMNIST3DDataset] = None
        self.val_dataset: Optional[MedMNIST3DDataset] = None
        self.test_dataset: Optional[MedMNIST3DDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for each stage.

        Args:
            stage: Stage name (fit/test/validate)
        """
        if stage == "fit" or stage is None:
            self.train_dataset = MedMNIST3DDataset(
                config=self.config,
                split="train",
            )
            self.val_dataset = MedMNIST3DDataset(
                config=self.config,
                split="val",
            )
            logger.info(
                f"Setup datasets: train={len(self.train_dataset)}, val={len(self.val_dataset)}"
            )

        if stage == "test" or stage is None:
            self.test_dataset = MedMNIST3DDataset(
                config=self.config,
                split="test",
            )
            logger.info(f"Setup dataset: test={len(self.test_dataset)}")

        if stage == "validate" or stage is None:
            if self.val_dataset is None:
                self.val_dataset = MedMNIST3DDataset(
                    config=self.config,
                    split="val",
                )

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader.

        Returns:
            Training DataLoader
        """
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.config.drop_last_train,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader.

        Returns:
            Validation DataLoader
        """
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None. Call setup() first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader.

        Returns:
            Test DataLoader
        """
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction DataLoader.

        Returns:
            Prediction DataLoader (uses test dataset)
        """
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
