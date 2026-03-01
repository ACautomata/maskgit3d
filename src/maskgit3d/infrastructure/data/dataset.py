"""
Simple data providers for synthetic data generation.

This module provides SimpleDataProvider for generating synthetic 3D volumes
for testing and development purposes.
"""
from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from maskgit3d.domain.interfaces import DataProvider


class SyntheticDataset(Dataset):
    """Dataset that generates synthetic 3D volumes on the fly."""

    def __init__(
        self,
        num_samples: int,
        in_channels: int = 1,
        out_channels: int = 1,
        spatial_size: tuple[int, int, int] = (64, 64, 64),
        mode: str = "train",
    ):
        """
        Initialize synthetic dataset.

        Args:
            num_samples: Number of samples in the dataset
            in_channels: Number of input channels
            out_channels: Number of output channels
            spatial_size: Spatial size of volumes (D, H, W)
            mode: Dataset mode ("train", "val", or "test")
        """
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size
        self.mode = mode

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        # Generate random input volume
        x = torch.randn(
            self.in_channels,
            self.spatial_size[0],
            self.spatial_size[1],
            self.spatial_size[2],
        )

        # Generate target (for reconstruction, target = input)
        y = torch.randn(
            self.out_channels,
            self.spatial_size[0],
            self.spatial_size[1],
            self.spatial_size[2],
        )

        return x, y


class SimpleDataProvider(DataProvider):
    """
    Simple data provider using synthetic data.

    Generates synthetic 3D volumes for training, validation, and testing.
    """

    def __init__(
        self,
        num_train: int = 100,
        num_val: int = 20,
        num_test: int = 20,
        batch_size: int = 1,
        in_channels: int = 1,
        out_channels: int = 1,
        spatial_size: tuple[int, int, int] = (64, 64, 64),
        num_workers: int = 0,
    ):
        """
        Initialize simple data provider.

        Args:
            num_train: Number of training samples
            num_val: Number of validation samples
            num_test: Number of test samples
            batch_size: Batch size
            in_channels: Number of input channels
            out_channels: Number of output channels
            spatial_size: Spatial size of volumes (D, H, W)
            num_workers: Number of data loading workers
        """
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size
        self.num_workers = num_workers

        # Create datasets
        self.train_dataset = SyntheticDataset(
            num_samples=num_train,
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_size=spatial_size,
            mode="train",
        )
        self.val_dataset = SyntheticDataset(
            num_samples=num_val,
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_size=spatial_size,
            mode="val",
        )
        self.test_dataset = SyntheticDataset(
            num_samples=num_test,
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_size=spatial_size,
            mode="test",
        )

    def train_loader(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Get training data loader."""
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return iter(loader)

    def val_loader(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Get validation data loader."""
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
        return iter(loader)

    def test_loader(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Get test data loader."""
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
        return iter(loader)
