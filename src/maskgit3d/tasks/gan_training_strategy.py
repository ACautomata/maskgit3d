"""GAN training strategy for manual optimization in Lightning."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from maskgit3d.models.vqvae.vqvae import VQVAE


class GANTrainingStrategy:
    """Handles manual optimization steps for GAN training."""

    def __init__(
        self,
        gradient_clip_val: float = 1.0,
        gradient_clip_enabled: bool = True,
    ):
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_enabled = gradient_clip_enabled

    def step_generator(
        self,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        vqvae: "VQVAE",
    ) -> None:
        """Execute generator optimization step (clip + step only; zero_grad is caller's responsibility)."""
        if self.gradient_clip_enabled and self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self._get_generator_params(vqvae), self.gradient_clip_val
            )
        optimizer.step()

    def step_discriminator(
        self,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        discriminator: torch.nn.Module,
    ) -> None:
        """Execute discriminator optimization step with gradient clipping."""
        if self.gradient_clip_enabled and self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), self.gradient_clip_val)
        optimizer.step()

    def _get_generator_params(self, vqvae: "VQVAE") -> list[torch.nn.Parameter]:
        """Get generator parameters for gradient clipping.

        VQ: Excludes quantizer (uses EMA-based codebook updates).
        FSQ: Includes quantizer (no EMA, needs gradient optimization).
        """
        params = (
            list(vqvae.encoder.parameters())
            + list(vqvae.quant_conv.parameters())
            + list(vqvae.post_quant_conv.parameters())
            + list(vqvae.decoder.parameters())
        )
        if getattr(vqvae, "quantizer_type", "vq") == "fsq":
            params.extend(list(vqvae.quantizer.parameters()))
        return params
