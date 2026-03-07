"""VQVAE training task with GAN-based manual optimization."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses.gan_loss import GANLoss
from ..losses.vq_loss import VQLoss
from ..models.discriminator.patch_discriminator import PatchDiscriminator3D
from ..models.vqvae import VQVAE
from .base_task import BaseTask


class VQVAETask(BaseTask):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 256,
        num_embeddings: int = 8192,
        embedding_dim: int = 256,
        lr_g: float = 4.5e-06,
        lr_d: float = 1e-04,
        lambda_l1: float = 1.0,
        lambda_vq: float = 1.0,
        lambda_gan: float = 0.1,
    ):
        super().__init__()
        self.automatic_optimization = False

        self.vqvae = VQVAE(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

        self.discriminator = PatchDiscriminator3D(
            in_channels=out_channels,
            ndf=64,
            n_layers=3,
            norm_layer="instance",
        )

        self.gan_loss = GANLoss(gan_mode="lsgan")
        self.vq_loss = VQLoss()

        self.lr_g = lr_g
        self.lr_d = lr_d
        self.lambda_l1 = lambda_l1
        self.lambda_vq = lambda_vq
        self.lambda_gan = lambda_gan

    def forward(self, x: torch.Tensor):
        return self.vqvae(x)

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        optimizers: List[torch.optim.Optimizer] | None = None,
    ):
        if optimizers is None:
            optimizers = self.optimizers()  # type: ignore[assignment]

        opt_g, opt_d = optimizers

        x_real = batch
        x_recon, vq_loss = self.vqvae(x_real)

        loss_l1 = F.l1_loss(x_recon, x_real)

        logits_fake = self.discriminator(x_recon)
        logits_fake = logits_fake[0][0]  # Extract tensor from list
        loss_gan_g = self.gan_loss.generator_loss(logits_fake)

        loss_g = self.lambda_l1 * loss_l1 + self.lambda_vq * vq_loss + self.lambda_gan * loss_gan_g

        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()

        logits_real = self.discriminator(x_real.detach())[0][0]
        logits_fake = self.discriminator(x_recon.detach())[0][0]

        loss_d = self.gan_loss.discriminator_loss(logits_real, logits_fake)

        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()

        self.log("train/loss_l1", loss_l1, prog_bar=True)
        self.log("train/loss_vq", vq_loss, prog_bar=True)
        self.log("train/loss_gan_g", loss_gan_g, prog_bar=True)
        self.log("train/loss_d", loss_d, prog_bar=True)
        self.log("train/loss_g", loss_g, prog_bar=True)

        return loss_g

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x_real = batch
        x_recon, vq_loss = self.vqvae(x_real)

        loss_l1 = F.l1_loss(x_recon, x_real)

        self.log("val/loss_l1", loss_l1, prog_bar=True)
        self.log("val/loss_vq", vq_loss, prog_bar=True)

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        opt_g = torch.optim.Adam(
            list(self.vqvae.encoder.parameters())
            + list(self.vqvae.quant_conv.parameters())
            + list(self.vqvae.post_quant_conv.parameters())
            + list(self.vqvae.quantizer.parameters())
            + list(self.vqvae.decoder.parameters()),
            lr=self.lr_g,
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_d,
        )
        return [opt_g, opt_d]
