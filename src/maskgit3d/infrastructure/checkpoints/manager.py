from pathlib import Path
from typing import Any

import torch


def load_checkpoint(
    path: str | Path,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """
    Load a checkpoint file with security best practices.

    Args:
        path: Path to the checkpoint file.
        map_location: Device to map tensors to.

    Returns:
        Dictionary containing checkpoint data.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        ValueError: If path is a directory instead of a file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Checkpoint path is not a file: {path}")

    # Use weights_only=True to prevent arbitrary code execution via pickle
    return torch.load(path, map_location=map_location, weights_only=True)


def save_checkpoint(
    payload: dict[str, Any],
    path: str | Path,
) -> None:
    """
    Save a checkpoint to disk.

    Args:
        payload: Dictionary containing checkpoint data to save.
        path: Destination path for the checkpoint file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
