"""MedMNIST dataset downloader with caching and verification."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from .config import MedMNISTConfig, MedMNISTDatasetName

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Dataset metadata."""

    name: MedMNISTDatasetName
    split: str  # train/val/test
    file_path: Path
    md5: str | None = None
    num_samples: int = 0


class MedMNISTDownloader:
    """Downloader for MedMNIST-3D datasets.

    Handles data downloading, caching, and MD5 verification.
    Uses the medmnist library for actual downloads.
    """

    # MD5 checksums from medmnist official
    # TODO: Add actual checksums when available
    MD5_CHECKSUMS: dict[str, dict[str, str]] = {
        "organmnist3d": {"train": "", "val": "", "test": ""},
        "nodulemnist3d": {"train": "", "val": "", "test": ""},
        "adrenalmnist3d": {"train": "", "val": "", "test": ""},
        "vesselmnist3d": {"train": "", "val": "", "test": ""},
        "fracturemnist3d": {"train": "", "val": "", "test": ""},
        "synapsemnist3d": {"train": "", "val": "", "test": ""},
    }

    DATASET_CLASS_NAMES: dict[str, str] = {
        "organmnist3d": "OrganMNIST3D",
        "nodulemnist3d": "NoduleMNIST3D",
        "adrenalmnist3d": "AdrenalMNIST3D",
        "vesselmnist3d": "VesselMNIST3D",
        "fracturemnist3d": "FractureMNIST3D",
        "synapsemnist3d": "SynapseMNIST3D",
    }

    def __init__(self, config: MedMNISTConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def ensure_data_available(self, split: str) -> Path:
        """Ensure data is available, downloading if necessary.

        Args:
            split: Data split (train/val/test)

        Returns:
            Path to data file

        Raises:
            FileNotFoundError: If data not found and download=False
        """
        if self._check_cached(split):
            logger.debug(f"Using cached data for {split}")
            return self._get_data_path(split)

        if self.config.download:
            logger.info(f"Downloading {self.config.dataset_name.value} {split}...")
            return self._download(split)

        raise FileNotFoundError(
            f"Dataset {self.config.dataset_name.value} ({split}) not found at "
            f"{self._get_data_path(split)}. Set download=True or download manually."
        )

    def _check_cached(self, split: str) -> bool:
        """Check if data is cached and valid.

        Args:
            split: Data split

        Returns:
            True if cached and valid
        """
        path = self._get_data_path(split)
        if not path.exists():
            return False

        # Check MD5 if available
        expected_md5 = self.MD5_CHECKSUMS.get(self.config.dataset_name.value, {}).get(split)

        if expected_md5:
            actual_md5 = self._compute_md5(path)
            if actual_md5 != expected_md5:
                logger.warning(
                    f"MD5 mismatch for {path.name}. Expected {expected_md5}, "
                    f"got {actual_md5}. Will re-download."
                )
                return False

        return True

    def _get_data_path(self, split: str) -> Path:
        """Get path for data file."""
        filename = f"{self.config.dataset_name.value}.npz"
        return self.data_dir / filename

    def _download(self, split: str) -> Path:
        """Download dataset using medmnist library.

        Args:
            split: Data split

        Returns:
            Path to downloaded file
        """
        # Lazy import medmnist
        try:
            import medmnist
        except ImportError as e:
            raise ImportError(
                "medmnist package is required. Install with: pip install medmnist"
            ) from e

        # Download will be triggered by creating dataset
        dataset_cls_name = self.DATASET_CLASS_NAMES[self.config.dataset_name.value]
        dataset_cls = getattr(medmnist, dataset_cls_name)
        _ = dataset_cls(
            split=split,
            root=str(self.data_dir),
            download=True,
            size=self.config.image_size,
        )

        return self._get_data_path(split)

    def _compute_md5(self, filepath: Path) -> str:
        """Compute MD5 hash of file.

        Args:
            filepath: Path to file

        Returns:
            MD5 hex digest
        """
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
