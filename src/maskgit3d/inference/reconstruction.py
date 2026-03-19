from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from monai.inferers.inferer import SlidingWindowInferer

from ..utils.sliding_window import create_sliding_window_inferer, pad_to_divisible


class VQVAEReconstructor:
    def __init__(
        self,
        sliding_window: dict[str, Any] | None,
        downsampling_factor: int,
        inferer: Any | None = None,
    ) -> None:
        self.sliding_window_cfg = sliding_window or {}
        self.downsampling_factor = downsampling_factor
        self._sliding_window_inferer = inferer

    def extract_input_tensor(self, batch: torch.Tensor | Sequence[Any]) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            return batch

        if isinstance(batch, Sequence) and batch:
            image = batch[0]
            if isinstance(image, torch.Tensor):
                return image

        raise TypeError("Expected batch to be a tensor or a sequence whose first item is a tensor.")

    def get_sliding_window_inferer(self) -> Any | None:
        if self._sliding_window_inferer is None:
            self._sliding_window_inferer = create_sliding_window_inferer(self.sliding_window_cfg)
        return self._sliding_window_inferer

    def pad_to_divisible(self, x: torch.Tensor) -> torch.Tensor:
        return pad_to_divisible(x, self.downsampling_factor)

    def reconstruct(
        self,
        vqvae: Any,
        batch: torch.Tensor | Sequence[Any],
    ) -> torch.Tensor:
        x_real = self.extract_input_tensor(batch)
        inferer = self.get_sliding_window_inferer()

        if inferer is None:
            recon, _ = vqvae(x_real)
            return recon

        original_shape = x_real.shape[2:]
        x_real_padded = self.pad_to_divisible(x_real)

        def encode_fn(patch: torch.Tensor) -> torch.Tensor:
            _, _, indices = vqvae.encode(patch)
            return indices.float().unsqueeze(1)

        indices_padded = cast(torch.Tensor, inferer(x_real_padded, encode_fn))
        latent_shape = tuple(size // self.downsampling_factor for size in original_shape)
        batch_size = x_real.shape[0]
        indices = indices_padded[:batch_size, 0]
        indices = indices[:, : latent_shape[0], : latent_shape[1], : latent_shape[2]].long()

        z_q = vqvae.quantizer.decode_from_indices(indices)
        latent_roi_size = tuple(
            size // self.downsampling_factor
            for size in self.sliding_window_cfg.get("roi_size", [32, 32, 32])
        )
        latent_needs_sliding_window = any(
            size > roi for size, roi in zip(z_q.shape[2:], latent_roi_size, strict=True)
        )

        if not latent_needs_sliding_window:
            return vqvae.decode(z_q)

        latent_inferer = SlidingWindowInferer(
            roi_size=latent_roi_size,
            sw_batch_size=self.sliding_window_cfg.get("sw_batch_size", 1),
            overlap=self.sliding_window_cfg.get("overlap", 0.25),
            mode=self.sliding_window_cfg.get("mode", "gaussian"),
            sigma_scale=self.sliding_window_cfg.get("sigma_scale", 0.125),
            padding_mode="constant",
            cval=0.0,
        )

        def decode_fn(latent_patch: torch.Tensor) -> torch.Tensor:
            return vqvae.decode(latent_patch)

        decoded = latent_inferer(z_q, decode_fn)
        return cast(torch.Tensor, decoded)
