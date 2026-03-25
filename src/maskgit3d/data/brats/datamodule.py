"""BraTS2023 DataModule for PyTorch Lightning."""

import logging
from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .config import BraTS2023Config, BraTSSubDataset
from .dataset import BraTS2023Dataset, _discover_cases, _generate_stratified_split
from .transforms import create_brats2023_training_transforms, create_brats2023_validation_transforms

logger = logging.getLogger(__name__)


def brats_collate_fn(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate BraTS samples into (images, labels) tuple.

    Args:
        batch: List of sample dicts with 'image' and 'modality_label' keys

    Returns:
        Tuple of (images_tensor, labels_tensor)
    """
    images = torch.stack([sample["image"] for sample in batch])
    labels = torch.tensor([sample["modality_label"] for sample in batch], dtype=torch.long)
    return images, labels


class BraTS2023DataModule(LightningDataModule):
    """PyTorch Lightning DataModule for BraTS 2023 dataset.

    Handles data loading, preprocessing, and DataLoader creation
    for training, validation, and testing.

    Key features:
    - Loads GLI, MEN, MET sub-datasets
    - Stratified 80/20 train/val split
    - Val and test use the same held-out split
    - Training: Stochastic single-modality sampling (random from 4 modalities)
    - Validation/Test: Deterministic single-modality sampling
    - Training: 128³ random crop with nnUNet-style augmentations
    - Inference: Full volume with sliding window support
    - Returns samples with: image (1-channel), modality_label (0-3), modality name

    Attributes:
        config: BraTS2023Config instance
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory in DataLoader
    """

    def __init__(
        self,
        data_dir: str = "/data/dataset/",
        crop_size: tuple[int, int, int] = (128, 128, 128),
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_ratio: float = 0.8,
        seed: int = 42,
        subdatasets: list[str] | None = None,
        stratify: bool = True,
        drop_last_train: bool = True,
        normalize_mode: str = "zscore",
    ):
        """Initialize BraTS2023 DataModule.

        Args:
            data_dir: Root directory containing BraTS case folders
            crop_size: Spatial crop size for training (D, H, W)
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory in DataLoader
            train_ratio: Ratio of data for training (0-1)
            seed: Random seed for deterministic splits
            subdatasets: List of sub-datasets to include ("gli", "men", "met")
            stratify: Whether to stratify splits by sub-dataset type
            drop_last_train: Whether to drop last incomplete batch in training
            normalize_mode: Normalization mode ("zscore" or "minmax")
        """
        super().__init__()

        # Convert string subdatasets to enum
        if subdatasets is None:
            subdataset_enums = [
                BraTSSubDataset.GLI,
                BraTSSubDataset.MEN,
                BraTSSubDataset.MET,
            ]
        else:
            subdataset_enums = [BraTSSubDataset(sd.lower()) for sd in subdatasets]

        self.config = BraTS2023Config(
            data_dir=data_dir,
            crop_size=crop_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_ratio=train_ratio,
            seed=seed,
            subdatasets=subdataset_enums,
            stratify=stratify,
            drop_last_train=drop_last_train,
        )
        self.normalize_mode = normalize_mode

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Datasets will be created in setup()
        self.train_dataset: BraTS2023Dataset | None = None
        self.val_dataset: BraTS2023Dataset | None = None
        self.test_dataset: BraTS2023Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for each stage.

        Args:
            stage: Stage name (fit/test/validate)
        """
        data_dir = Path(self.config.data_dir)

        # Discover all cases
        all_cases = _discover_cases(data_dir, self.config.subdatasets)

        if not all_cases:
            logger.warning(
                f"No complete BraTS cases found in {data_dir}. "
                f"Expected directories like BraTS-GLI-XXXXX-XXX, BraTS-MEN-XXXXX-XXX, BraTS-MET-XXXXX-XXX"
            )
            return

        # Generate stratified split
        train_cases, held_out_cases = _generate_stratified_split(
            all_cases,
            self.config.train_ratio,
            self.config.seed,
        )

        logger.info(
            f"Discovered {len(all_cases)} cases: "
            f"{len(train_cases)} train, {len(held_out_cases)} val/test"
        )

        if stage == "fit" or stage is None:
            train_transform = create_brats2023_training_transforms(
                crop_size=self.config.crop_size,
                normalize_mode=self.normalize_mode,
            )
            self.train_dataset = BraTS2023Dataset(
                cases=train_cases,
                transform=train_transform,
                deterministic=False,
                seed=self.config.seed,
            )
            logger.info(f"Setup train dataset: {len(self.train_dataset)} samples")

        if stage in ("fit", "validate", "test") or stage is None:
            val_transform = create_brats2023_validation_transforms(
                normalize_mode=self.normalize_mode,
            )
            self.val_dataset = BraTS2023Dataset(
                cases=held_out_cases,
                transform=val_transform,
                deterministic=True,
                seed=self.config.seed,
            )
            self.test_dataset = self.val_dataset
            logger.info(f"Setup val/test dataset: {len(self.val_dataset)} samples")

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
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=brats_collate_fn,
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
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=brats_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader.

        Note: Returns the same data as val_dataloader (val = test split).

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
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=brats_collate_fn,
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
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=brats_collate_fn,
        )
