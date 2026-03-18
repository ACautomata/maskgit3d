"""VQVAE training task with GAN-based manual optimization and adaptive weighting."""

from collections.abc import Sequence
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from monai.inferers.inferer import SlidingWindowInferer
from omegaconf import DictConfig

from ..losses.vq_perceptual_loss import VQPerceptualLoss
from ..models.vqvae import VQVAE
from ..utils.sliding_window import create_sliding_window_inferer, pad_to_divisible
from .base_task import BaseTask
from .gan_training_strategy import GANTrainingStrategy


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
        model_config: Optional[DictConfig] = None,
        optimizer_config: Optional[DictConfig] = None,
        disc_optimizer_config: Optional[DictConfig] = None,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 256,
        num_embeddings: int = 8192,
        embedding_dim: int = 256,
        num_channels: Sequence[int] = (64, 128, 256),
        num_res_blocks: Sequence[int] = (2, 2, 2),
        attention_levels: Sequence[bool] = (False, False, False),
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
        adaptive_weight_max: float = 100.0,
        disc_loss: str = "hinge",
        sliding_window: dict[str, Any] | None = None,
        commitment_cost: float = 0.25,
        gradient_clip_val: float = 1.0,
        gradient_clip_enabled: bool = True,
        quantizer_type: Literal["vq", "fsq"] = "vq",
        fsq_levels: Sequence[int] = (8, 8, 8, 5, 5, 5),
    ):
        super().__init__()
        self.automatic_optimization = False

        if model_config is not None:
            self.vqvae = instantiate(model_config)
        else:
            self.vqvae = VQVAE(
                in_channels=in_channels,
                out_channels=out_channels,
                latent_channels=latent_channels,
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                num_channels=num_channels,
                num_res_blocks=num_res_blocks,
                attention_levels=attention_levels,
                commitment_cost=commitment_cost,
                quantizer_type=quantizer_type,
                fsq_levels=fsq_levels,
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
            adaptive_weight_max=adaptive_weight_max,
            perceptual_network=perceptual_network,
            use_perceptual=use_perceptual,
        )

        self.lr_g = lr_g
        self.lr_d = lr_d
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_enabled = gradient_clip_enabled
        self.gan_strategy = GANTrainingStrategy(gradient_clip_val, gradient_clip_enabled)

        self.sliding_window_cfg = sliding_window or {}
        self._sliding_window_inferer: SlidingWindowInferer | None = None
        self._downsampling_factor = 16

        self.vqvae.enable_gradient_checkpointing()

        self.save_hyperparameters()

    def _extract_input_tensor(self, batch: torch.Tensor | Sequence[Any]) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            return batch

        if isinstance(batch, Sequence) and batch:
            image = batch[0]
            if isinstance(image, torch.Tensor):
                return image

        raise TypeError("Expected batch to be a tensor or a sequence whose first item is a tensor.")

    def _get_sliding_window_inferer(self) -> SlidingWindowInferer | None:
        if self._sliding_window_inferer is None:
            self._sliding_window_inferer = create_sliding_window_inferer(self.sliding_window_cfg)
        return self._sliding_window_inferer

    def _pad_to_divisible(self, x: torch.Tensor) -> torch.Tensor:
        return pad_to_divisible(x, self._downsampling_factor)

    def _get_decoder_last_layer(self) -> torch.nn.Parameter | None:
        """Get the last layer weight of the decoder for adaptive weight calculation.

        Returns the weight parameter of the final convolution layer in the decoder.
        Used for computing gradient norms in adaptive GAN loss weighting.

        Returns:
            The last layer parameter (weight, dim >= 2), or None if not found.
        """
        try:
            decoder = self.vqvae.decoder.decoder
            params = [p for p in decoder.parameters() if p.ndim >= 2]
            if params:
                return params[-1]
            return None
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return self.vqvae(x)  # type: ignore[no-any-return]

    def training_step(
        self,
        batch: torch.Tensor | Sequence[Any],
        batch_idx: int,
        optimizers: list[torch.optim.Optimizer] | None = None,
    ) -> None:
        if optimizers is None:
            optimizers = self.optimizers()  # type: ignore[assignment]

        opt_g, opt_d = optimizers  # type: ignore[misc]

        x_real = self._extract_input_tensor(batch)
        last_layer = self._get_decoder_last_layer()

        x_recon, vq_loss = self.vqvae(x_real)

        loss_g, log_g = self.loss_fn(
            inputs=x_real,
            reconstructions=x_recon,
            vq_loss=vq_loss,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=last_layer,
            split="train",
        )

        prog_bar_keys = {"total_loss", "vq_loss", "g_loss"}
        for key, value in log_g.items():
            if isinstance(value, torch.Tensor):
                metric_name = key.split("/")[-1]
                self.log(
                    key,
                    value.item(),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=metric_name in prog_bar_keys,
                    batch_size=x_real.shape[0],
                )

        self.log(
            "train/x_recon_min", x_recon.min().item(), on_step=True, on_epoch=False, prog_bar=False
        )
        self.log(
            "train/x_recon_max", x_recon.max().item(), on_step=True, on_epoch=False, prog_bar=False
        )
        self.log(
            "train/x_recon_abs_mean",
            x_recon.abs().mean().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        self.log(
            "train/x_real_min", x_real.min().item(), on_step=True, on_epoch=False, prog_bar=False
        )
        self.log(
            "train/x_real_max", x_real.max().item(), on_step=True, on_epoch=False, prog_bar=False
        )

        opt_g.zero_grad()
        self.manual_backward(loss_g)
        self.gan_strategy.step_generator(opt_g, loss_g, self.vqvae)

        # Detach reconstructions before discriminator step to free the generator
        # computation graph. The disc loss_fn already detaches internally for its
        # logits, but keeping x_recon attached here prevents the gen graph from
        # being freed until after disc backward — a major memory leak on 3D volumes.
        x_recon_detached = x_recon.detach()

        # Explicitly delete generator graph tensors now that backward is done
        del loss_g, log_g, x_recon

        loss_d, log_d = self.loss_fn(
            inputs=x_real,
            reconstructions=x_recon_detached,
            vq_loss=vq_loss,
            optimizer_idx=1,
            global_step=self.global_step,
            split="train",
        )

        for key, value in log_d.items():
            if isinstance(value, torch.Tensor):
                metric_name = key.split("/")[-1]
                self.log(
                    key,
                    value.item(),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=metric_name == "disc_loss",
                    batch_size=x_real.shape[0],
                )

        opt_d.zero_grad()
        self.manual_backward(loss_d)
        self.gan_strategy.step_discriminator(opt_d, loss_d)

        # Explicitly delete all remaining large tensors to help CUDA free memory
        del x_real, x_recon_detached, loss_d, log_d, vq_loss

        # Return None — manual optimization doesn't need return values.
        # Returning a dict causes Lightning to accumulate results across the
        # epoch, holding references to CUDA tensors and preventing memory reuse.

    def validation_step(
        self, batch: torch.Tensor | Sequence[Any], batch_idx: int
    ) -> dict[str, Any]:
        x_real = self._extract_input_tensor(batch)
        original_shape = x_real.shape[2:]
        inferer = self._get_sliding_window_inferer()
        x_recon: torch.Tensor

        if inferer is not None:
            # Stage 2 style: sliding window on encoder, then on decoder
            # Step 1: Sliding window on encoder to get latent indices
            x_real_padded = self._pad_to_divisible(x_real)

            def encode_fn(patch: torch.Tensor) -> torch.Tensor:
                z_q, _, indices = self.vqvae.encode(patch)
                return indices.float().unsqueeze(1)

            indices_padded = inferer(x_real_padded, encode_fn)

            # Crop to latent dimensions
            latent_d = original_shape[0] // self._downsampling_factor
            latent_h = original_shape[1] // self._downsampling_factor
            latent_w = original_shape[2] // self._downsampling_factor

            B = x_real.shape[0]
            indices = indices_padded[:B, 0, :latent_d, :latent_h, :latent_w].long()  # type: ignore[call-overload]

            # Step 2: Decode indices to z_q
            z_q = self.vqvae.quantizer.decode_from_indices(indices)

            # Step 3: Sliding window on decoder
            def decode_fn(patch: torch.Tensor) -> torch.Tensor:
                return self.vqvae.decode(patch)

            x_recon_raw = inferer(z_q, decode_fn)
            x_recon = x_recon_raw  # type: ignore[assignment]
        else:
            recon, _ = self.vqvae(x_real)  # type: ignore[misc]
            x_recon = recon

        perceptual_loss = torch.tensor(0.0, device=x_real.device)
        if self.loss_fn.use_perceptual and self.loss_fn.perceptual_loss is not None:
            perceptual_loss = self.loss_fn.perceptual_loss(x_recon, x_real)

        self.log(
            "val_perceptual_loss",
            perceptual_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x_real.shape[0],
        )

        rec_loss = F.l1_loss(x_recon, x_real)
        self.log(
            "val_rec_loss",
            rec_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x_real.shape[0],
        )

        return {
            "x_real": x_real.detach().cpu(),
            "x_recon": x_recon.detach().cpu(),
        }

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        if self.hparams.get("optimizer_config") is not None:
            # Override lr from optimizer config with lr_g for generator
            param_groups = [
                {"params": self.vqvae.encoder.parameters(), "lr": self.lr_g},
                {"params": self.vqvae.quant_conv.parameters(), "lr": self.lr_g},
                {"params": self.vqvae.post_quant_conv.parameters(), "lr": self.lr_g},
                {"params": self.vqvae.decoder.parameters(), "lr": self.lr_g},
            ]
            opt_g = instantiate(self.hparams["optimizer_config"], _partial_=True)(
                param_groups, lr=self.lr_g
            )
        else:
            opt_g = torch.optim.Adam(
                list(self.vqvae.encoder.parameters())
                + list(self.vqvae.quant_conv.parameters())
                + list(self.vqvae.post_quant_conv.parameters())
                + list(self.vqvae.decoder.parameters()),
                lr=self.lr_g,
            )

        if self.hparams.get("disc_optimizer_config") is not None:
            # Override lr from optimizer config with lr_d for discriminator
            opt_d = instantiate(self.hparams["disc_optimizer_config"], _partial_=True)(
                self.loss_fn.discriminator.parameters(), lr=self.lr_d
            )
        else:
            opt_d = torch.optim.Adam(
                self.loss_fn.discriminator.parameters(),
                lr=self.lr_d,
            )

        return [opt_g, opt_d]

    @torch.no_grad()
    def test_step(self, batch: torch.Tensor | Sequence[Any], batch_idx: int) -> dict[str, Any]:
        x_real = self._extract_input_tensor(batch)
        original_shape = x_real.shape[2:]
        inferer = self._get_sliding_window_inferer()
        x_recon: torch.Tensor

        if inferer is not None:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Step 1: Sliding window on encoder to get latent indices
            x_real_padded = self._pad_to_divisible(x_real)

            def encode_fn(patch: torch.Tensor) -> torch.Tensor:
                z_q, _, indices = self.vqvae.encode(patch)
                return indices.float().unsqueeze(1)

            indices_padded = inferer(x_real_padded, encode_fn)

            # Crop to latent dimensions
            latent_d = original_shape[0] // self._downsampling_factor
            latent_h = original_shape[1] // self._downsampling_factor
            latent_w = original_shape[2] // self._downsampling_factor

            B = x_real.shape[0]
            indices = indices_padded[:B, 0, :latent_d, :latent_h, :latent_w].long()  # type: ignore[call-overload]

            # Step 2: Decode indices to z_q
            z_q = self.vqvae.quantizer.decode_from_indices(indices)

            # Step 3: Sliding window on decoder
            def decode_fn(patch: torch.Tensor) -> torch.Tensor:
                return self.vqvae.decode(patch)

            x_recon_raw = inferer(z_q, decode_fn)
            x_recon = x_recon_raw  # type: ignore[assignment]
        else:
            recon, _ = self.vqvae(x_real)  # type: ignore[misc]
            x_recon = recon

        loss = F.l1_loss(x_recon, x_real)

        return {
            "loss": loss.item(),
            "inference_time": 0.0,
            "use_sliding_window": inferer is not None,
            "x_real": x_real.detach().cpu(),
            "x_recon": x_recon.detach().cpu(),
        }

    def predict_step(self, batch: torch.Tensor | Sequence[Any], batch_idx: int) -> torch.Tensor:
        x_real = self._extract_input_tensor(batch)
        x_recon: torch.Tensor
        x_recon, _ = self.vqvae(x_real)  # type: ignore[misc]
        return x_recon
