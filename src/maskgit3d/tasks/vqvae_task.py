"""VQVAE training task with GAN-based manual optimization."""

import time
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms.croppad.array import DivisiblePad

from ..losses.gan_loss import GANLoss
from ..losses.perceptual_loss import PerceptualLoss
from ..losses.vq_loss import VQLoss
from ..models.discriminator.patch_discriminator import PatchDiscriminator3D
from ..models.vqvae import VQVAE
from .base_task import BaseTask


def compute_downsampling_factor(
    channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4),
) -> int:
    """Compute the downsampling factor from VQVAE channel multipliers.

    The downsampling factor is 2^n where n is the number of downsampling
    layers (len(channel_multipliers) - 1).

    Args:
        channel_multipliers: Channel multipliers for encoder/decoder.
            Default is (1, 1, 2, 2, 4) which gives 4 downsampling layers.

    Returns:
        Downsampling factor (e.g., 16 for 4 downsampling layers)
    """
    num_down_layers = len(channel_multipliers) - 1
    return 2**num_down_layers


def validate_crop_size(
    crop_size: tuple[int, int, int],
    downsampling_factor: int = 16,
) -> tuple[int, int, int]:
    """Validate and adjust crop size to be divisible by downsampling factor.

    Args:
        crop_size: Original crop size (D, H, W)
        downsampling_factor: VQVAE downsampling factor (default 16)

    Returns:
        Validated crop size divisible by downsampling factor

    Raises:
        ValueError: If crop size would need significant adjustment
    """
    for dim, val in zip(("D", "H", "W"), crop_size, strict=True):
        if val % downsampling_factor != 0:
            raise ValueError(
                f"Crop size {crop_size} has {dim}={val} not divisible by {downsampling_factor}. "
                f"Use a size divisible by {downsampling_factor}."
            )

    return crop_size


def compute_padded_size(
    input_size: tuple[int, int, int],
    downsampling_factor: int = 16,
) -> tuple[int, int, int]:
    """Compute the minimum padded size divisible by downsampling factor.

    Args:
        input_size: Input spatial size (D, H, W)
        downsampling_factor: VQVAE downsampling factor (default 16)

    Returns:
        Padded size where each dimension is divisible by downsampling factor
    """
    d, h, w = input_size

    def ceil_div(x: int, divisor: int) -> int:
        return (x + divisor - 1) // divisor * divisor

    return (
        ceil_div(d, downsampling_factor),
        ceil_div(h, downsampling_factor),
        ceil_div(w, downsampling_factor),
    )


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
        use_perceptual: bool = True,
        lambda_perceptual: float = 0.1,
        perceptual_network: str = "alex",
        sliding_window: dict[str, Any] | None = None,
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

        self.use_perceptual = use_perceptual
        self.lambda_perceptual = lambda_perceptual
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss(network=perceptual_network)

        self.sliding_window_cfg = sliding_window or {}
        self._sliding_window_inferer: SlidingWindowInferer | None = None
        self._divisible_pad: DivisiblePad | None = None
        self._downsampling_factor = 16

    def _get_sliding_window_inferer(self) -> SlidingWindowInferer | None:
        if not self.sliding_window_cfg.get("enabled", False):
            return None
        if self._sliding_window_inferer is None:
            roi_size = tuple(self.sliding_window_cfg.get("roi_size", [64, 64, 64]))
            overlap = self.sliding_window_cfg.get("overlap", 0.25)
            mode = self.sliding_window_cfg.get("mode", "gaussian")
            sigma_scale = self.sliding_window_cfg.get("sigma_scale", 0.125)
            sw_batch_size = self.sliding_window_cfg.get("sw_batch_size", 1)
            self._sliding_window_inferer = SlidingWindowInferer(
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
                mode=mode,
                sigma_scale=sigma_scale,
                padding_mode="constant",
                cval=0.0,
            )
        return self._sliding_window_inferer

    def _get_divisible_pad(self) -> DivisiblePad:
        if self._divisible_pad is None:
            self._divisible_pad = DivisiblePad(k=self._downsampling_factor, mode="constant")
        return self._divisible_pad

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

        opt_g, opt_d = optimizers  # type: ignore[misc]  # type: ignore[misc]

        x_real = batch
        x_recon, vq_loss = self.vqvae(x_real)

        loss_l1 = F.l1_loss(x_recon, x_real)

        logits_fake = self.discriminator(x_recon)
        logits_fake = logits_fake[0][0]  # Extract tensor from list
        loss_gan_g = self.gan_loss.generator_loss(logits_fake)

        loss_g = self.lambda_l1 * loss_l1 + self.lambda_vq * vq_loss + self.lambda_gan * loss_gan_g

        if self.use_perceptual:
            loss_perceptual = self.perceptual_loss(x_recon, x_real)
            loss_g = loss_g + self.lambda_perceptual * loss_perceptual
        else:
            loss_perceptual = None

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
        if loss_perceptual is not None:
            self.log("train/loss_perceptual", loss_perceptual, prog_bar=True)

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

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        x_real = batch
        original_shape = x_real.shape[2:]
        inferer = self._get_sliding_window_inferer()

        if inferer is not None:
            x_real_padded = self._get_divisible_pad()(x_real)
            padded_shape = x_real_padded.shape[2:]

            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            def model_forward(x: torch.Tensor) -> torch.Tensor:
                recon, _ = self.vqvae(x)
                return recon

            x_recon_padded = inferer(x_real_padded, model_forward)
            inference_time = time.time() - start_time

            pad_d = padded_shape[0] - original_shape[0]
            pad_h = padded_shape[1] - original_shape[1]
            pad_w = padded_shape[2] - original_shape[2]
            x_recon = x_recon_padded[  # type: ignore[index]
                :,
                :,
                pad_d // 2 : pad_d // 2 + original_shape[0],
                pad_h // 2 : pad_h // 2 + original_shape[1],
                pad_w // 2 : pad_w // 2 + original_shape[2],
            ]
            x_real_for_loss = x_real
        else:
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            x_recon, vq_loss = self.vqvae(x_real)
            x_real_for_loss = x_real
            inference_time = time.time() - start_time

        loss_l1 = F.l1_loss(x_recon, x_real_for_loss)
        with torch.no_grad():
            _, vq_loss, _ = self.vqvae.encode(x_real_for_loss)
            vq_loss = vq_loss if isinstance(vq_loss, torch.Tensor) else torch.tensor(0.0)

        self.log("test/loss_l1", loss_l1, prog_bar=True)
        self.log("test/loss_vq", vq_loss, prog_bar=True)
        self.log("test/inference_time", inference_time, prog_bar=True)

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            self.log("test/peak_memory_mb", peak_memory, prog_bar=True)

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_real = batch
        original_shape = x_real.shape[2:]
        inferer = self._get_sliding_window_inferer()

        if inferer is not None:
            x_real_padded = self._get_divisible_pad()(x_real)
            padded_shape = x_real_padded.shape[2:]

            def model_forward(x: torch.Tensor) -> torch.Tensor:
                recon, _ = self.vqvae(x)
                return recon

            x_recon_padded = inferer(x_real_padded, model_forward)

            pad_d = padded_shape[0] - original_shape[0]
            pad_h = padded_shape[1] - original_shape[1]
            pad_w = padded_shape[2] - original_shape[2]
            x_recon = x_recon_padded[  # type: ignore[index]
                :,
                :,
                pad_d // 2 : pad_d // 2 + original_shape[0],
                pad_h // 2 : pad_h // 2 + original_shape[1],
                pad_w // 2 : pad_w // 2 + original_shape[2],
            ]
        else:
            x_recon, _ = self.vqvae(x_real)

        return x_recon
