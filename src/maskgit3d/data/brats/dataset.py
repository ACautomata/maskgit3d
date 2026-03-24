"""BraTS2023 dataset implementation with case discovery and split generation."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from maskgit3d.data.brats.config import MODALITY_ORDER, BraTSSubDataset


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

    Args:
        data_dir: Root directory containing BraTS case folders
        subdatasets: List of sub-datasets to include

    Returns:
        List of BraTS2023CaseRecord for complete cases
    """
    cases: list[BraTS2023CaseRecord] = []
    subdataset_prefixes = [f"BraTS-{sd.value.upper()}" for sd in subdatasets]

    if not data_dir.exists():
        return cases

    for item in data_dir.iterdir():
        if not item.is_dir():
            continue

        case_id = item.name

        # Check if this is a BraTS directory for one of the requested subdatasets
        if not any(case_id.startswith(prefix) for prefix in subdataset_prefixes):
            continue

        # Determine subdataset from case_id
        subdataset: BraTSSubDataset | None = None
        for sd in subdatasets:
            if case_id.startswith(f"BraTS-{sd.value.upper()}"):
                subdataset = sd
                break

        if subdataset is None:
            continue

        # Check if case is complete (all modalities present)
        if not _is_complete_case(item, case_id):
            continue

        # Build image paths in fixed order
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
) -> Tuple[list[BraTS2023CaseRecord], list[BraTS2023CaseRecord]]:
    """Generate a stratified train/held-out split of cases.

    Args:
        cases: List of all cases to split
        train_ratio: Ratio of cases for training (0-1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_cases, held_out_cases)
    """
    random.seed(seed)

    # Group cases by subdataset for stratification
    by_subdataset: dict[BraTSSubDataset, list[BraTS2023CaseRecord]] = {
        BraTSSubDataset.GLI: [],
        BraTSSubDataset.MEN: [],
        BraTSSubDataset.MET: [],
    }

    for case in cases:
        by_subdataset[case.subdataset].append(case)

    train_cases: list[BraTS2023CaseRecord] = []
    held_out_cases: list[BraTS2023CaseRecord] = []

    for subdataset, subdataset_cases in by_subdataset.items():
        if not subdataset_cases:
            continue

        # Shuffle within subdataset for reproducibility
        shuffled = subdataset_cases.copy()
        random.shuffle(shuffled)

        # Calculate split point
        n_train = int(len(shuffled) * train_ratio)

        # Add to respective lists
        train_cases.extend(shuffled[:n_train])
        held_out_cases.extend(shuffled[n_train:])

    return train_cases, held_out_cases
