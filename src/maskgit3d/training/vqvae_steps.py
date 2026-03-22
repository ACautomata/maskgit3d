from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import torch

from ..inference import VQVAEReconstructor
from ..tasks.gan_training_strategy import GANTrainingStrategy


class VQVAETrainingSteps:
    def __init__(
        self,
        vqvae: Any,
        loss_fn: Any,
        gan_strategy: GANTrainingStrategy | Any,
        reconstructor: VQVAEReconstructor | Any,
        log_fn: Callable[..., None] | None = None,
        manual_backward_fn: Callable[[torch.Tensor], None] | None = None,
    ) -> None:
        self.vqvae = vqvae
        self.loss_fn = loss_fn
        self.gan_strategy = gan_strategy
        self.reconstructor = reconstructor
        self.log_fn = log_fn
        self.manual_backward_fn = manual_backward_fn

    def extract_input_tensor(self, batch: torch.Tensor | Sequence[Any]) -> torch.Tensor:
        return self.reconstructor.extract_input_tensor(batch)

    def get_sliding_window_inferer(self) -> Any | None:
        return self.reconstructor.get_sliding_window_inferer()

    def pad_to_divisible(self, x: torch.Tensor) -> torch.Tensor:
        return self.reconstructor.pad_to_divisible(x)

    def get_decoder_last_layer(self) -> torch.nn.Parameter | None:
        try:
            decoder = self.vqvae.decoder.decoder
            params = [param for param in decoder.parameters() if param.ndim >= 2]
            if params:
                return params[-1]
        except (AttributeError, IndexError):
            return None
        return None

    def shared_step_generator(
        self,
        x_real: torch.Tensor,
        batch_idx: int,
        split: str,
        global_step: int,
        last_layer: torch.nn.Parameter | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del batch_idx
        if last_layer is None:
            last_layer = self.get_decoder_last_layer()

        x_recon, vq_loss = self.vqvae(x_real)
        return self.loss_fn(
            inputs=x_real,
            reconstructions=x_recon,
            vq_loss=vq_loss,
            optimizer_idx=0,
            global_step=global_step,
            last_layer=last_layer,
            split=split,
        )

    def shared_step_discriminator(
        self,
        x_real: torch.Tensor,
        x_recon: torch.Tensor,
        vq_loss: torch.Tensor,
        split: str,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.loss_fn(
            inputs=x_real,
            reconstructions=x_recon,
            vq_loss=vq_loss,
            optimizer_idx=1,
            global_step=global_step,
            split=split,
        )

    def training_step(
        self,
        batch: torch.Tensor | Sequence[Any],
        batch_idx: int,
        optimizers: Sequence[Any],
        global_step: int,
        manual_backward_fn: Callable[[torch.Tensor], None] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | torch.nn.Parameter | None]]:
        backward = manual_backward_fn or self.manual_backward_fn
        if backward is None:
            raise ValueError("training_step requires manual_backward_fn.")

        opt_g, opt_d = optimizers
        x_real = self.extract_input_tensor(batch)
        last_layer = self.get_decoder_last_layer()
        x_recon, vq_loss = self.vqvae(x_real)
        loss_g, log_g = self.loss_fn(
            inputs=x_real,
            reconstructions=x_recon,
            vq_loss=vq_loss,
            optimizer_idx=0,
            global_step=global_step,
            last_layer=last_layer,
            split="train",
        )

        opt_g.zero_grad()
        backward(loss_g)
        self.gan_strategy.step_generator(opt_g, loss_g, self.vqvae)

        callback_payload: dict[str, torch.Tensor | torch.nn.Parameter | None] = {
            "x_real": x_real.detach(),
            "x_recon": x_recon.detach(),
            "vq_loss": vq_loss.detach(),
            "last_layer": last_layer,
        }

        x_recon_detached = x_recon.detach()
        returned_loss = loss_g.detach()
        del log_g, x_recon

        loss_d, log_d = self.shared_step_discriminator(
            x_real=x_real,
            x_recon=x_recon_detached,
            vq_loss=vq_loss,
            split="train",
            global_step=global_step,
        )

        opt_d.zero_grad()
        backward(loss_d)
        self.gan_strategy.step_discriminator(opt_d, loss_d)

        del x_real, x_recon_detached, loss_d, log_d, vq_loss, batch_idx, loss_g
        return returned_loss, callback_payload

    def reconstruction_step(
        self,
        batch: torch.Tensor | Sequence[Any],
        split: str,
    ) -> dict[str, Any]:
        x_real = self.extract_input_tensor(batch)
        inferer = self.get_sliding_window_inferer()

        if split == "test" and inferer is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        x_recon, vq_loss = self.reconstructor.reconstruct(self.vqvae, x_real)

        if split == "val":
            return {
                "x_real": x_real.detach(),
                "x_recon": x_recon.detach(),
                "vq_loss": vq_loss.detach(),
            }

        return {
            "x_real": x_real.detach(),
            "x_recon": x_recon.detach(),
            "vq_loss": vq_loss.detach(),
            "inference_time": 0.0,
            "use_sliding_window": inferer is not None,
        }

    def predict_step(self, batch: torch.Tensor | Sequence[Any]) -> dict[str, torch.Tensor]:
        x_real = self.extract_input_tensor(batch)
        x_recon, vq_loss = self.reconstructor.reconstruct(self.vqvae, x_real)
        return {
            "x_real": x_real.detach(),
            "x_recon": x_recon.detach(),
            "vq_loss": vq_loss.detach(),
        }

    def _extract_vq_loss(self, x_real: torch.Tensor) -> Any:
        encode = getattr(self.vqvae, "encode", None)
        if callable(encode):
            with torch.no_grad():
                encoded = encode(x_real)
            if isinstance(encoded, tuple) and len(encoded) >= 2:
                return encoded[1]

        with torch.no_grad():
            forward_output = self.vqvae(x_real)
        if isinstance(forward_output, tuple) and len(forward_output) >= 2:
            return forward_output[1]

        raise ValueError("Unable to extract vq_loss from VQVAE model output.")

    def _ensure_tensor(self, value: Any, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        return torch.tensor(float(value), device=device)
