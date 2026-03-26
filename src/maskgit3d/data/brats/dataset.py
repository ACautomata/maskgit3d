"""BraTS2023 dataset implementation with case discovery and split generation."""

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from maskgit3d.data.brats.config import MODALITY_ORDER, BraTSSubDataset

MODALITY_TO_LABEL: dict[str, int] = {modality: idx for idx, modality in enumerate(MODALITY_ORDER)}


@dataclass
class BraTS2023CaseRecord:
    """Record for a single BraTS 2023 case.

    Attributes:
        case_id: Unique case identifier (e.g., "BraTS-GLI-00001-000")
        subdataset: Which sub-dataset this case belongs to (GLI, MEN, MET)
        image_paths: List of 4 paths to modality files in fixed order (t1n, t1c, t2w, t2f)
    """

    case_id: str
    subdataset: BraTSSubDataset
    image_paths: list[Path]


def _is_complete_case(case_dir: Path, case_id: str) -> bool:
    """Check if a case directory contains all 4 required modalities.

    Args:
        case_dir: Path to case directory
        case_id: Case identifier

    Returns:
        True if all 4 modalities exist, False otherwise
    """
    for modality in MODALITY_ORDER:
        modality_file = case_dir / f"{case_id}-{modality}.nii.gz"
        if not modality_file.exists():
            return False
    return True


def _discover_cases(
    data_dir: Path, subdatasets: list[BraTSSubDataset]
) -> list[BraTS2023CaseRecord]:
    """Discover all complete BraTS 2023 cases from the data directory.

    Recursively searches subdirectories for case folders (BraTS-XXX-YYYYY-ZZZ).

    Args:
        data_dir: Root directory containing BraTS data (may have nested structure)
        subdatasets: List of sub-datasets to include

    Returns:
        List of BraTS2023CaseRecord for complete cases
    """
    cases: list[BraTS2023CaseRecord] = []
    subdataset_prefixes = [f"BraTS-{sd.value.upper()}" for sd in subdatasets]

    if not data_dir.exists():
        return cases

    for item in data_dir.rglob("*"):
        if not item.is_dir():
            continue

        case_id = item.name

        if not any(case_id.startswith(prefix) for prefix in subdataset_prefixes):
            continue

        subdataset: BraTSSubDataset | None = None
        for sd in subdatasets:
            if case_id.startswith(f"BraTS-{sd.value.upper()}"):
                subdataset = sd
                break

        if subdataset is None:
            continue

        if not _is_complete_case(item, case_id):
            continue

        image_paths = [item / f"{case_id}-{modality}.nii.gz" for modality in MODALITY_ORDER]

        cases.append(
            BraTS2023CaseRecord(
                case_id=case_id,
                subdataset=subdataset,
                image_paths=image_paths,
            )
        )

    return cases


def _generate_stratified_split(
    cases: list[BraTS2023CaseRecord],
    train_ratio: float,
    seed: int,
) -> tuple[list[BraTS2023CaseRecord], list[BraTS2023CaseRecord]]:
    """Generate a stratified train/held-out split of cases.

    Args:
        cases: List of all cases to split
        train_ratio: Ratio of cases for training (0-1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_cases, held_out_cases)
    """
    random.seed(seed)

    by_subdataset: dict[BraTSSubDataset, list[BraTS2023CaseRecord]] = {
        BraTSSubDataset.GLI: [],
        BraTSSubDataset.MEN: [],
        BraTSSubDataset.MET: [],
    }

    for case in cases:
        by_subdataset[case.subdataset].append(case)

    train_cases: list[BraTS2023CaseRecord] = []
    held_out_cases: list[BraTS2023CaseRecord] = []

    for _subdataset, subdataset_cases in by_subdataset.items():
        if not subdataset_cases:
            continue

        shuffled = subdataset_cases.copy()
        random.shuffle(shuffled)

        n_train = int(len(shuffled) * train_ratio)

        train_cases.extend(shuffled[:n_train])
        held_out_cases.extend(shuffled[n_train:])

    return train_cases, held_out_cases


class BraTS2023Dataset(Dataset):
    """PyTorch Dataset for BraTS 2023 with mixed-modality sampling.

    Each sample randomly selects one modality from the 4 available
    (t1n, t1c, t2w, t2f) and returns it as a single-channel volume
    along with its modality label.

    Attributes:
        cases: List of BraTS2023CaseRecord for this split
        transform: Optional MONAI transform to apply to samples
        deterministic: If True, use deterministic sampling (for validation/test)
        seed: Base seed for deterministic sampling
    """

    def __init__(
        self,
        cases: list[BraTS2023CaseRecord],
        transform=None,
        deterministic: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.cases = cases
        self.transform = transform
        self.deterministic = deterministic
        self.seed = seed

    def __len__(self) -> int:
        return len(self.cases)

    def _sample_modality(self, case_id: str) -> int:
        """Sample a modality index for the given case.

        Uses torch.randint for proper seeding with DataLoader workers.
        For deterministic mode, derives modality from stable hash(case_id + seed).

        Args:
            case_id: Case identifier for deterministic hashing

        Returns:
            Modality index (0-3)
        """
        if self.deterministic:
            key = f"{case_id}_{self.seed}".encode()
            combined = int(hashlib.md5(key).hexdigest(), 16)
            return combined % len(MODALITY_ORDER)
        else:
            return int(torch.randint(0, len(MODALITY_ORDER), (1,)).item())

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= len(self.cases):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.cases)}")

        case = self.cases[idx]

        modality_idx = self._sample_modality(case.case_id)
        selected_path = case.image_paths[modality_idx]
        modality_name = MODALITY_ORDER[modality_idx]
        modality_label = modality_idx

        sample = {
            "image": selected_path,
            "modality_label": modality_label,
            "modality": modality_name,
            "case_id": case.case_id,
            "subdataset": case.subdataset,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
