"""MedMNIST-3D Dataset implementation."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import MedMNISTConfig, TaskType
from .downloader import MedMNISTDownloader

logger = logging.getLogger(__name__)


class MedMNIST3DDataset(Dataset):
    """MedMNIST-3D Dataset wrapper.

    Loads data from MedMNIST npz files and provides PyTorch Dataset interface.
    """

    SPLIT_KEYS = {
        "train": "train_images",
        "val": "val_images",
        "test": "test_images",
    }

    LABEL_KEYS = {
        "train": "train_labels",
        "val": "val_labels",
        "test": "test_labels",
    }

    def __init__(
        self,
        config: MedMNISTConfig,
        split: str,
        downloader: Optional[MedMNISTDownloader] = None,
    ):
        """Initialize MedMNIST-3D dataset.

        Args:
            config: MedMNIST configuration
            split: Data split (train/val/test)
            downloader: Optional downloader instance
        """
        if split not in self.SPLIT_KEYS:
            raise ValueError(
                f"Invalid split: {split}. Must be one of {list(self.SPLIT_KEYS.keys())}"
            )

        self.config = config
        self.split = split
        self.task_type = config.task_type
        self.downloader = downloader or MedMNISTDownloader(config)

        self.data_path = self.downloader.ensure_data_available(split)
        self.images, self.labels = self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load images and labels from npz file."""
        data = np.load(self.data_path)

        image_key = self.SPLIT_KEYS[self.split]
        label_key = self.LABEL_KEYS[self.split]

        images = data[image_key]
        labels = data[label_key]

        logger.debug(
            f"Loaded {len(images)} samples from {self.split} split, "
            f"images shape: {images.shape}, labels shape: {labels.shape}"
        )

        return images, labels

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, target) where:
                - image: Tensor of shape (D, H, W) or (1, D, H, W)
                - target: None for reconstruction, tensor of label for classification
        """
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to tensor and add channel dimension
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        # Normalize to [0, 1]
        image = image / 255.0

        # Add channel dimension at front: (D, H, W) -> (1, D, H, W)
        image = image.unsqueeze(0)

        # Return target based on task type
        if self.task_type == TaskType.RECONSTRUCTION:
            target = None
        else:
            target = torch.tensor(label.item(), dtype=torch.long)

        return image, target

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return self.config.num_classes

    @property
    def input_size(self) -> int:
        """Return input size (28 for all MedMNIST-3D datasets)."""
        return self.config.input_size
