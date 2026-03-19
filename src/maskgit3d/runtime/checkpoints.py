from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import torch

from ..models.vqvae import VQVAE


class CheckpointLoaderProtocol(Protocol):
    def load(self, ckpt_path: str) -> VQVAE: ...


class VQVAECheckpointLoader:
    def __init__(
        self,
        model_factory: Callable[[], VQVAE] | None = None,
        torch_load: Callable[..., Any] = torch.load,
    ) -> None:
        self._model_factory = model_factory or VQVAE
        self._torch_load = torch_load

    def load(self, ckpt_path: str) -> VQVAE:
        checkpoint = self._torch_load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = _extract_state_dict(checkpoint)
        model = self._model_factory()
        model.load_state_dict(state_dict)
        return model


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        nested_state = checkpoint["state_dict"]
        if isinstance(nested_state, Mapping):
            return nested_state
    if isinstance(checkpoint, Mapping):
        return checkpoint
    raise TypeError(
        "Expected checkpoint to be a state_dict mapping or contain a 'state_dict' mapping."
    )


def load_vqvae_from_checkpoint(ckpt_path: str) -> VQVAE:
    return VQVAECheckpointLoader().load(ckpt_path)
