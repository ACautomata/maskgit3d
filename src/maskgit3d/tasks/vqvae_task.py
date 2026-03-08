"""VQVAE training task with GAN-based manual optimization and adaptive weighting."""

from typing import Any

import torch
import torch.nn.functional as F
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms.croppad.array import DivisiblePad

from ..losses.vq_perceptual_loss import VQPerceptualLoss
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
    result: int = 2**num_down_layers
    return result


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
        disc_start: int = 0,
        disc_factor: float = 1.0,
        use_adaptive_weight: bool = True,
        disc_loss: str = "hinge",
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

        self.loss_fn = VQPerceptualLoss(
            disc_in_channels=out_channels,
            disc_num_layers=3,
            disc_ndf=64,
            disc_norm="instance",
            disc_loss=disc_loss,
            lambda_l1=lambda_l1,
            lambda_vq=lambda_vq,
            lambda_perceptual=lambda_perceptual,
            discriminator_weight=lambda_gan,
            disc_start=disc_start,
            disc_factor=disc_factor,
            use_adaptive_weight=use_adaptive_weight,
            perceptual_network=perceptual_network,
            use_perceptual=use_perceptual,
        )

        self.lr_g = lr_g
        self.lr_d = lr_d

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

    def _get_decoder_last_layer(self) -> torch.nn.Parameter | None:
        """Get the last layer weight of the decoder for adaptive weight calculation.

        Returns the weight parameter of the final convolution layer in the decoder.
        Used for computing gradient norms in adaptive GAN loss weighting.

        Returns:
            The last layer parameter, or None if not found.
        """
        try:
            return list(self.vqvae.decoder.decoder.parameters())[-1]
        except (AttributeError, IndexError):
            return None

    def _shared_step_generator(
        self,
        x_real: torch.Tensor,
        batch_idx: int,
        split: str,
        last_layer: torch.nn.Parameter | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute generator loss for a batch.

        Args:
            x_real: Input tensor.
            batch_idx: Batch index.
            split: Split name (train/val/test).
            last_layer: Last layer parameter for adaptive weight calculation.
                If None, will be obtained from decoder.

        Returns:
            Tuple of (loss_g, log_g) where loss_g is the generator loss
            and log_g is a dict of logging values.
        """
        if last_layer is None:
            last_layer = self._get_decoder_last_layer()

        x_recon, vq_loss = self.vqvae(x_real)

        loss_g, log_g = self.loss_fn(
            inputs=x_real,
            reconstructions=x_recon,
            vq_loss=vq_loss,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=last_layer,
            split=split,
        )
        return loss_g, log_g

    def _shared_step_discriminator(
        self,
        x_real: torch.Tensor,
        x_recon: torch.Tensor,
        vq_loss: torch.Tensor,
        split: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute discriminator loss for a batch.

        Args:
            x_real: Input tensor.
            x_recon: Reconstructed tensor from VQVAE.
            vq_loss: VQ loss from the quantization step.
            split: Split name (train/val/test).

        Returns:
            Tuple of (loss_d, log_d) where loss_d is the discriminator loss
            and log_d is a dict of logging values.
        """
        loss_d, log_d = self.loss_fn(
            inputs=x_real,
            reconstructions=x_recon,
            vq_loss=vq_loss,
            optimizer_idx=1,
            global_step=self.global_step,
            split=split,
        )
        return loss_d, log_d

    def forward(self, x: torch.Tensor):
        return self.vqvae(x)

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        optimizers: list[torch.optim.Optimizer] | None = None,
    ) -> dict[str, Any]:
        if optimizers is None:
            optimizers = self.optimizers()  # type: ignore[assignment]

        opt_g, opt_d = optimizers  # type: ignore[misc]

        x_real = batch
        last_layer = self._get_decoder_last_layer()

        x_recon, vq_loss = self.vqvae(x_real)

        loss_g, _ = self.loss_fn(
            inputs=x_real,
            reconstructions=x_recon,
            vq_loss=vq_loss,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=last_layer,
            split="train",
        )
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()

        loss_d, _ = self.loss_fn(
            inputs=x_real,
            reconstructions=x_recon,
            vq_loss=vq_loss,
            optimizer_idx=1,
            global_step=self.global_step,
            split="train",
        )
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()

        return {
            "loss": loss_g,
            "x_real": x_real,
            "x_recon": x_recon,
            "vq_loss": vq_loss,
            "last_layer": last_layer,
        }

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, Any]:
        x_real = batch
        x_recon, vq_loss = self.vqvae(x_real)

        return {
            "loss": vq_loss,
            "x_real": x_real,
            "x_recon": x_recon,
            "vq_loss": vq_loss,
        }

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        opt_g = torch.optim.Adam(
            list(self.vqvae.encoder.parameters())
            + list(self.vqvae.quant_conv.parameters())
            + list(self.vqvae.post_quant_conv.parameters())
            + list(self.vqvae.quantizer.parameters())
            + list(self.vqvae.decoder.parameters()),
            lr=self.lr_g,
        )
        opt_d = torch.optim.Adam(
            self.loss_fn.discriminator.parameters(),
            lr=self.lr_d,
        )
        return [opt_g, opt_d]

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, Any]:
        x_real = batch
        original_shape = x_real.shape[2:]
        inferer = self._get_sliding_window_inferer()

        if inferer is not None:
            x_real_padded = self._get_divisible_pad()(x_real)
            padded_shape = x_real_padded.shape[2:]

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            def model_forward(x: torch.Tensor) -> torch.Tensor:
                recon: torch.Tensor
                recon, _ = self.vqvae(x)  # type: ignore[misc]
                return recon

            x_recon_padded = inferer(x_real_padded, model_forward)

            pad_d = padded_shape[0] - original_shape[0]
            pad_h = padded_shape[1] - original_shape[1]
            pad_w = padded_shape[2] - original_shape[2]
            x_recon = x_recon_padded[  # type: ignore[call-overload]
                :,
                :,
                pad_d // 2 : pad_d // 2 + original_shape[0],
                pad_h // 2 : pad_h // 2 + original_shape[1],
                pad_w // 2 : pad_w // 2 + original_shape[2],
            ]
        else:
            x_recon, _ = self.vqvae(x_real)  # type: ignore[misc]

        loss = F.l1_loss(x_recon, x_real)

        return {
            "loss": loss,
            "x_real": x_real,
            "x_recon": x_recon,
            "inference_time": 0.0,
            "use_sliding_window": inferer is not None,
        }

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_real = batch
        x_recon: torch.Tensor
        x_recon, _ = self.vqvae(x_real)  # type: ignore[misc]
        return x_recon
