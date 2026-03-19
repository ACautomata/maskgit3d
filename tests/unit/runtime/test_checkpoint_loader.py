from pathlib import Path
from unittest.mock import Mock

import torch

from src.maskgit3d.models.vqvae import VQVAE
from src.maskgit3d.runtime.checkpoints import VQVAECheckpointLoader, load_vqvae_from_checkpoint


def test_load_vqvae_from_checkpoint_extracts_state_dict(tmp_path: Path) -> None:
    source_model = VQVAE()
    checkpoint_path = tmp_path / "vqvae.ckpt"
    torch.save({"state_dict": source_model.state_dict()}, checkpoint_path)

    loaded_model = load_vqvae_from_checkpoint(str(checkpoint_path))

    assert isinstance(loaded_model, VQVAE)
    source_state = source_model.state_dict()
    loaded_state = loaded_model.state_dict()
    assert loaded_state.keys() == source_state.keys()
    assert torch.equal(loaded_state["quant_conv.weight"], source_state["quant_conv.weight"])


def test_vqvae_checkpoint_loader_uses_weights_only_when_loading() -> None:
    source_model = VQVAE()
    torch_load = Mock(return_value=source_model.state_dict())

    loader = VQVAECheckpointLoader(torch_load=torch_load)

    loader.load("/tmp/vqvae.ckpt")

    torch_load.assert_called_once_with(
        "/tmp/vqvae.ckpt",
        map_location="cpu",
        weights_only=True,
    )
