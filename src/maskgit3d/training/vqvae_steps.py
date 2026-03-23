from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from ..inference import VQVAEReconstructor
from ..tasks.gan_training_strategy import GANTrainingStrategy
from ..tasks.output_contracts import VQVAEEvalStepOutput, VQVAETrainingStepOutput


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
            params: list[torch.nn.Parameter] = [
                param for param in decoder.parameters() if param.ndim >= 2
            ]
            if params:
                result: torch.nn.Parameter | None = params[-1]
                return result
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
        result: tuple[torch.Tensor, dict[str, torch.Tensor]] = self.loss_fn(
            inputs=x_real,
            reconstructions=x_recon,
            vq_loss=vq_loss,
            optimizer_idx=0,
            global_step=global_step,
            last_layer=last_layer,
            split=split,
        )
        return result

    def shared_step_discriminator(
        self,
        x_real: torch.Tensor,
        x_recon: torch.Tensor,
        vq_loss: torch.Tensor,
        split: str,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        disc_result: tuple[torch.Tensor, dict[str, torch.Tensor]] = self.loss_fn(
            inputs=x_real,
            reconstructions=x_recon,
            vq_loss=vq_loss,
            optimizer_idx=1,
            global_step=global_step,
            split=split,
        )
        return disc_result

    def training_step(
        self,
        batch: torch.Tensor | Sequence[Any],
        batch_idx: int,
        optimizers: Sequence[Any],
        global_step: int,
        manual_backward_fn: Callable[[torch.Tensor], None] | None = None,
    ) -> VQVAETrainingStepOutput:
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

        x_recon_detached = x_recon.detach()
        returned_loss = loss_g.detach()
        zero = torch.zeros((), device=returned_loss.device, dtype=returned_loss.dtype)
        del x_recon, last_layer, batch_idx

        loss_d, log_d = self.shared_step_discriminator(
            x_real=x_real,
            x_recon=x_recon_detached,
            vq_loss=vq_loss,
            split="train",
            global_step=global_step,
        )

        opt_d.zero_grad()
        backward(loss_d)
        self.gan_strategy.step_discriminator(opt_d, loss_d, self.loss_fn.discriminator)

        output: VQVAETrainingStepOutput = {
            "loss": returned_loss,
            "loss_g": returned_loss,
            "loss_d": loss_d.detach(),
            "nll_loss": log_g.get("train/nll_loss", zero).detach(),
            "rec_loss": log_g.get("train/rec_loss", zero).detach(),
            "p_loss": log_g.get("train/p_loss", zero).detach(),
            "g_loss": log_g.get("train/g_loss", zero).detach(),
            "vq_loss": log_g.get("train/vq_loss", zero).detach(),
            "disc_loss": log_d.get("train/disc_loss", zero).detach(),
        }
        del x_real, x_recon_detached, loss_d, log_d, vq_loss, loss_g, log_g, zero
        return output

    def reconstruction_step(
        self,
        batch: torch.Tensor | Sequence[Any],
    ) -> VQVAEEvalStepOutput:
        x_real = self.extract_input_tensor(batch)
        x_recon, _ = self.reconstructor.reconstruct(self.vqvae, x_real)
        return {
            "x_real": x_real.detach().cpu(),
            "x_recon": x_recon.detach().cpu(),
        }

    def predict_step(self, batch: torch.Tensor | Sequence[Any]) -> VQVAEEvalStepOutput:
        return self.reconstruction_step(batch)

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
