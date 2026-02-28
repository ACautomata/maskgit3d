from pathlib import Path
from unittest.mock import patch

import pytest
import torch


def test_checkpoint_loader_uses_weights_only(tmp_path: Path) -> None:
    """Test that load_checkpoint uses weights_only=True for security."""
    # Create a fake checkpoint file
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": {}}, ckpt_path)

    captured: dict = {"weights_only": None}

    def fake_load(path: Path, map_location: str, weights_only: bool = False) -> dict:
        captured["weights_only"] = weights_only
        return {"model_state_dict": {}}

    with patch("torch.load", fake_load):
        # Import after patching to avoid caching issues
        from maskgit3d.infrastructure.checkpoints.manager import load_checkpoint

        load_checkpoint(ckpt_path, map_location="cpu")

    assert captured["weights_only"] is True, (
        "load_checkpoint must use weights_only=True to prevent arbitrary code execution"
    )


def test_load_checkpoint_raises_on_missing_file(tmp_path: Path) -> None:
    """Test that load_checkpoint raises FileNotFoundError for missing files."""
    from maskgit3d.infrastructure.checkpoints.manager import load_checkpoint

    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nonexistent.pt", map_location="cpu")


def test_load_checkpoint_raises_on_directory(tmp_path: Path) -> None:
    """Test that load_checkpoint raises ValueError when path is a directory."""
    from maskgit3d.infrastructure.checkpoints.manager import load_checkpoint

    with pytest.raises(ValueError):
        load_checkpoint(tmp_path, map_location="cpu")


def test_save_checkpoint_creates_directory(tmp_path: Path) -> None:
    """Test that save_checkpoint creates parent directories."""
    from maskgit3d.infrastructure.checkpoints.manager import save_checkpoint

    ckpt_path = tmp_path / "subdir" / "checkpoint.pt"
    save_checkpoint({"model_state_dict": {}}, ckpt_path)

    assert ckpt_path.exists()
    loaded = torch.load(ckpt_path, weights_only=True)
    assert "model_state_dict" in loaded
