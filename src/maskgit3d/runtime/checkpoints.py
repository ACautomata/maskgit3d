from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import torch
from omegaconf import DictConfig, OmegaConf

from ..models.vqvae import VQVAE


class CheckpointLoaderProtocol(Protocol):
    def load(self, ckpt_path: str) -> VQVAE: ...


class VQVAECheckpointLoader:
    def __init__(
        self,
        model_factory: Callable[[DictConfig | None], VQVAE] | None = None,
        torch_load: Callable[..., Any] = torch.load,
    ) -> None:
        self._model_factory = model_factory or self._default_model_factory
        self._torch_load = torch_load

    @staticmethod
    def _default_model_factory(model_config: DictConfig | None = None) -> VQVAE:
        if model_config is not None:
            from .model_factory import create_vqvae_model

            return create_vqvae_model(model_config)
        return VQVAE()

    def load(self, ckpt_path: str) -> VQVAE:
        checkpoint = self._torch_load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = _extract_state_dict(checkpoint)
        model_config = _extract_model_config(checkpoint)
        filtered_state_dict = _filter_vqvae_state_dict(state_dict)
        model = self._model_factory(model_config)
        model.load_state_dict(filtered_state_dict, strict=False)
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


def _extract_model_config(checkpoint: Mapping[str, Any]) -> DictConfig | None:
    """Extract model configuration from checkpoint hyper_parameters if available."""
    hyper_params = checkpoint.get("hyper_parameters")
    if isinstance(hyper_params, dict):
        model_config = hyper_params.get("model_config")
        if model_config is not None:
            result: DictConfig | None = OmegaConf.create(model_config)  # type: ignore[assignment]
            return result
    return None


def _filter_vqvae_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Filter state dict to only include VQVAE model keys.

    - Strips 'vqvae.' prefix from keys (Lightning module format)
    - Removes non-VQVAE keys (loss_fn.*, etc.)
    """
    result: dict[str, Any] = {}
    for key, value in state_dict.items():
        # Skip non-VQVAE keys (loss_fn, etc.)
        if key.startswith("loss_fn."):
            continue
        if key.startswith("vqvae."):
            result[key[len("vqvae.") :]] = value
        elif not _is_non_vqvae_key(key):
            result[key] = value
    return result


def _is_non_vqvae_key(key: str) -> bool:
    """Check if a key belongs to non-VQVAE components."""
    non_vqvae_prefixes = (
        "loss_fn.",
        "maskgit.",
        "gan_strategy.",
        "reconstructor.",
        "training_steps.",
        "optimizer_factory.",
    )
    return any(key.startswith(prefix) for prefix in non_vqvae_prefixes)


def load_vqvae_from_checkpoint(ckpt_path: str) -> VQVAE:
    return VQVAECheckpointLoader().load(ckpt_path)
