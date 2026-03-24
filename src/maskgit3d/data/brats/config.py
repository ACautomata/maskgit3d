"""BraTS2023 configuration classes and enums."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class BraTSSubDataset(str, Enum):
    """Supported BraTS 2023 sub-dataset types."""

    GLI = "gli"
    MEN = "men"
    MET = "met"


# Fixed modality order: T1, T1 contrast-enhanced, T2, T2-FLAIR
MODALITY_ORDER: Tuple[str, str, str, str] = ("t1n", "t1c", "t2w", "t2f")


def _validate_crop_size(crop_size: tuple[int, int, int]) -> None:
    """Validate crop size configuration."""
    if len(crop_size) != 3:
        raise ValueError(f"crop_size must be a 3-element tuple, got {len(crop_size)} elements")
    if any(s % 16 != 0 for s in crop_size):
        raise ValueError(
            f"crop_size must be divisible by 16 for VQVAE compatibility, got {crop_size}"
        )


def _validate_train_ratio(train_ratio: float) -> None:
    """Validate train ratio is within valid range."""
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")


@dataclass
class BraTS2023Config:
    """Configuration for BraTS 2023 dataset.

    Attributes:
        data_dir: Root directory for BraTS data (default: /data/dataset/)
        crop_size: Spatial crop size for training (D, H, W), must be divisible by 16
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory in DataLoader
        train_ratio: Ratio of data for training (0-1), rest is held-out for val/test
        seed: Random seed for deterministic splits
        subdatasets: List of sub-datasets to include (GLI, MEN, MET)
        stratify: Whether to stratify splits by sub-dataset type
        drop_last_train: Whether to drop last incomplete batch in training
    """

    data_dir: str = "/data/dataset/"
    crop_size: tuple[int, int, int] = (128, 128, 128)
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    train_ratio: float = 0.8
    seed: int = 42
    subdatasets: list[BraTSSubDataset] = field(
        default_factory=lambda: [BraTSSubDataset.GLI, BraTSSubDataset.MEN, BraTSSubDataset.MET]
    )
    stratify: bool = True
    drop_last_train: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        _validate_crop_size(self.crop_size)
        _validate_train_ratio(self.train_ratio)

    @property
    def modality_order(self) -> Tuple[str, str, str, str]:
        """Return the fixed modality order for BraTS 2023."""
        return MODALITY_ORDER

    @property
    def num_modalities(self) -> int:
        """Return the number of modalities (always 4 for BraTS 2023)."""
        return len(MODALITY_ORDER)
