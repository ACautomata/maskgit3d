"""Unified VQVAE loss module with adaptive GAN weighting.

Based on Taming Transformers (CompVis/taming-transformers) implementation:
https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.discriminator.patch_discriminator import PatchDiscriminator3D
from .perceptual_loss import PerceptualLoss

if TYPE_CHECKING:
    from .perceptual_loss import PerceptualLoss


def adopt_weight(weight: float, global_step: int, threshold: int = 0, value: float = 0.0) -> float:
    """Adopt weight with warmup threshold.

    If global_step < threshold, return value (default 0.0).
    Otherwise, return the original weight.

    Args:
        weight: The weight value to potentially adopt.
        global_step: Current training step.
        threshold: Step threshold before using the weight.
        value: Value to use before threshold.

    Returns:
        Weight value based on warmup schedule.
    """
    if global_step < threshold:
        return value
    return weight


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator.

    Args:
        logits_real: Discriminator output for real images.
        logits_fake: Discriminator output for fake images.

    Returns:
        Discriminator hinge loss.
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Vanilla (LSGAN-style) loss for discriminator.

    Args:
        logits_real: Discriminator output for real images.
        logits_fake: Discriminator output for fake images.

    Returns:
        Discriminator vanilla loss.
    """
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    return 0.5 * (loss_real + loss_fake)


def lsgan_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """LSGAN loss for generator.

    Args:
        logits_fake: Discriminator output for generated images.

    Returns:
        Generator LSGAN loss.
    """
    return 0.5 * torch.mean((logits_fake - 1.0) ** 2)


class VQPerceptualLoss(nn.Module):
    """Unified VQVAE loss module with adaptive GAN weighting.

    Attributes:
        perceptual_loss: Perceptual loss module or None if disabled.


    This module combines:
    - L1 reconstruction loss
    - Perceptual loss (LPIPS via MONAI)
    - VQ codebook loss
    - GAN adversarial loss with adaptive weight balancing

    The adaptive weight calculation balances the gradient magnitudes of
    the reconstruction loss and GAN loss, as described in Taming Transformers.

    Args:
        # Discriminator parameters
        disc_in_channels: Number of input channels for discriminator.
        disc_num_layers: Number of layers in discriminator.
        disc_ndf: Base number of discriminator features.
        disc_norm: Normalization layer type ("instance" or "batch").
        disc_loss: Discriminator loss type ("hinge" or "vanilla").

        # Loss weights
        lambda_l1: Weight for L1 reconstruction loss.
        lambda_vq: Weight for VQ codebook loss.
        lambda_perceptual: Weight for perceptual loss.
        discriminator_weight: Base weight for GAN loss (multiplied by adaptive weight).

        # Adaptive weight parameters
        disc_start: Step to start using discriminator (warmup).
        disc_factor: Multiplier for discriminator loss.
        use_adaptive_weight: Whether to use adaptive weight calculation.

        # Perceptual loss parameters
        perceptual_network: Network for perceptual loss ("alex", "vgg", etc.).
        use_perceptual: Whether to use perceptual loss.

    Example:
        >>> loss_fn = VQPerceptualLoss(
        ...     disc_in_channels=1,
        ...     disc_start=50001,
        ...     use_adaptive_weight=True,
        ... )
        >>> # Generator update
        >>> loss_g, log_g = loss_fn(inputs, reconstructions, vq_loss,
        ...                          optimizer_idx=0, global_step=step,
        ...                          last_layer=decoder_last_layer)
        >>> # Discriminator update
        >>> loss_d, log_d = loss_fn(inputs, reconstructions, vq_loss,
        ...                          optimizer_idx=1, global_step=step)
    """

    def __init__(
        self,
        disc_in_channels: int = 1,
        disc_num_layers: int = 3,
        disc_ndf: int = 64,
        disc_norm: str = "instance",
        disc_loss: str = "hinge",
        lambda_l1: float = 1.0,
        lambda_vq: float = 1.0,
        lambda_perceptual: float = 0.1,
        discriminator_weight: float = 1.0,
        disc_start: int = 0,
        disc_factor: float = 1.0,
        use_adaptive_weight: bool = True,
        perceptual_network: str = "alex",
        use_perceptual: bool = True,
    ) -> None:
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_vq = lambda_vq
        self.lambda_perceptual = lambda_perceptual
        self.discriminator_weight = discriminator_weight

        self.disc_start = disc_start
        self.disc_factor = disc_factor
        self.use_adaptive_weight = use_adaptive_weight

        self.discriminator = PatchDiscriminator3D(
            in_channels=disc_in_channels,
            ndf=disc_ndf,
            n_layers=disc_num_layers,
            norm_layer=disc_norm,
        )

        valid_disc_losses = {"hinge", "vanilla"}
        if disc_loss not in valid_disc_losses:
            raise ValueError(
                f"Unsupported disc_loss '{disc_loss}'. Expected one of {sorted(valid_disc_losses)}"
            )
        self.disc_loss_type = disc_loss
        if disc_loss == "hinge":
            self._disc_loss = hinge_d_loss
            self._gen_loss = lambda logits: -torch.mean(logits)
        else:
            self._disc_loss = vanilla_d_loss
            self._gen_loss = lsgan_g_loss

        self.use_perceptual = use_perceptual
        self.perceptual_loss: PerceptualLoss | None = None
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss(network=perceptual_network)
            for param in self.perceptual_loss.parameters():
                param.requires_grad = False

    def calculate_adaptive_weight(
        self,
        nll_loss: torch.Tensor,
        g_loss: torch.Tensor,
        last_layer: nn.Parameter | None = None,
    ) -> torch.Tensor:
        """Calculate adaptive weight for GAN loss.

        The adaptive weight is computed as the ratio of gradient norms:
            d_weight = ||grad_nll|| / (||grad_gan|| + eps)

        This balances the contribution of reconstruction loss and GAN loss
        by scaling the GAN loss to have similar gradient magnitude.

        Args:
            nll_loss: Negative log-likelihood (reconstruction) loss.
            g_loss: Generator (GAN) loss.
            last_layer: Last layer of decoder for gradient computation.

        Returns:
            Adaptive weight for GAN loss.
        """
        if last_layer is None:
            return torch.tensor(self.discriminator_weight, device=nll_loss.device)

        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

        return d_weight * self.discriminator_weight

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        vq_loss: torch.Tensor,
        optimizer_idx: int,
        global_step: int,
        last_layer: nn.Parameter | None = None,
        split: str = "train",
    ) -> tuple[torch.Tensor, dict]:
        """Compute loss for generator or discriminator update.

        Args:
            inputs: Real input images (B, C, D, H, W).
            reconstructions: Reconstructed images (B, C, D, H, W).
            vq_loss: VQ codebook loss (scalar).
            optimizer_idx: 0 for generator update, 1 for discriminator update.
            global_step: Current training step.
            last_layer: Last layer of decoder for adaptive weight calculation.
            split: Split name for logging (e.g., "train", "val").

        Returns:
            Tuple of (loss, log_dict) where:
                - loss: Scalar loss tensor for backpropagation.
                - log_dict: Dictionary of metrics for logging.
        """
        # Reconstruction loss (L1)
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        # Perceptual loss
        p_loss = torch.tensor(0.0, device=inputs.device)
        if self.use_perceptual and self.perceptual_loss is not None:
            p_loss = self.perceptual_loss(reconstructions.contiguous(), inputs.contiguous())
            rec_loss = rec_loss + self.lambda_perceptual * p_loss

        # Total reconstruction loss (NLL)
        nll_loss = torch.mean(rec_loss)

        # Discriminator factor (warmup)
        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.disc_start)

        if optimizer_idx == 0:
            # =====================
            # Generator update
            # =====================

            # Get discriminator logits for fake images
            logits_fake = self.discriminator(reconstructions.contiguous())[0][0]

            # Generator GAN loss
            g_loss = self._gen_loss(logits_fake)

            # Calculate adaptive weight
            if self.use_adaptive_weight and disc_factor > 0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
                except RuntimeError:
                    # Edge case: training=False or no grad
                    d_weight = torch.tensor(self.discriminator_weight, device=inputs.device)
            else:
                d_weight = torch.tensor(self.discriminator_weight, device=inputs.device)

            # Total generator loss
            loss = (
                self.lambda_l1 * nll_loss
                + d_weight * disc_factor * g_loss
                + self.lambda_vq * vq_loss
            )

            # Logging
            log = {
                f"{split}/total_loss": loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": torch.mean(rec_loss).detach(),
                f"{split}/p_loss": p_loss.detach().mean()
                if isinstance(p_loss, torch.Tensor)
                else p_loss,
                f"{split}/g_loss": g_loss.detach().mean(),
                f"{split}/vq_loss": vq_loss.detach().mean()
                if isinstance(vq_loss, torch.Tensor)
                else vq_loss,
                f"{split}/d_weight": d_weight.detach().mean()
                if isinstance(d_weight, torch.Tensor)
                else d_weight,
                f"{split}/disc_factor": torch.tensor(disc_factor),
            }

            return loss, log

        if optimizer_idx == 1:
            # =====================
            # Discriminator update
            # =====================

            # Get discriminator logits
            logits_real = self.discriminator(inputs.contiguous().detach())[0][0]
            logits_fake = self.discriminator(reconstructions.contiguous().detach())[0][0]

            # Discriminator loss
            d_loss = disc_factor * self._disc_loss(logits_real, logits_fake)

            # Logging
            log = {
                f"{split}/disc_loss": d_loss.detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean(),
                f"{split}/disc_factor": torch.tensor(disc_factor),
            }

            return d_loss, log

        raise ValueError(f"Invalid optimizer_idx: {optimizer_idx}. Expected 0 or 1.")
