"""VQVAE training task with GAN-based manual optimization and adaptive weighting."""

from collections.abc import Sequence
from typing import Any, Literal, cast

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from ..inference import VQVAEReconstructor
from ..losses.vq_perceptual_loss import VQPerceptualLoss
from ..models.vqvae import VQVAE
from ..models.vqvae.splitting import compute_downsampling_factor, resolve_num_splits
from ..training import VQVAETrainingSteps
from .base_task import BaseTask
from .gan_training_strategy import GANTrainingStrategy


class VQVAETask(BaseTask):
    def __init__(
        self,
        model_config: DictConfig | None = None,
        optimizer_config: DictConfig | None = None,
        disc_optimizer_config: DictConfig | None = None,
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
        num_splits: int | None = None,
        dim_split: int = 1,
        data_config: DictConfig | None = None,
    ):
        super().__init__()
        self.automatic_optimization = False

        crop_size: tuple[int, int, int] | None = None
        roi_size: tuple[int, int, int] | None = None

        if (
            data_config is not None
            and hasattr(data_config, "crop_size")
            and data_config.crop_size is not None
        ):
            crop_size = tuple(data_config.crop_size)
        if sliding_window is not None and sliding_window.get("roi_size") is not None:
            roi_size = tuple(sliding_window["roi_size"])

        effective_num_channels = num_channels
        if model_config is not None and hasattr(model_config, "num_channels"):
            effective_num_channels = tuple(model_config.num_channels)

        resolved_num_splits, split_reason = resolve_num_splits(
            crop_size=crop_size,
            roi_size=roi_size,
            num_channels=effective_num_channels,
            dim_split=dim_split,
            requested_num_splits=num_splits,
        )

        if model_config is not None:
            config_dict = dict(model_config)
            config_dict["num_splits"] = resolved_num_splits
            config_dict["dim_split"] = dim_split
            self.vqvae = instantiate(config_dict)
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
                num_splits=resolved_num_splits,
                dim_split=dim_split,
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
        self._downsampling_factor = compute_downsampling_factor(list(effective_num_channels))
        self.reconstructor = VQVAEReconstructor(
            sliding_window=self.sliding_window_cfg,
            downsampling_factor=self._downsampling_factor,
        )
        self.training_steps = VQVAETrainingSteps(
            vqvae=self.vqvae,
            loss_fn=self.loss_fn,
            gan_strategy=self.gan_strategy,
            reconstructor=self.reconstructor,
        )

        self.vqvae.enable_gradient_checkpointing()

        self.save_hyperparameters()

    def _extract_input_tensor(self, batch: torch.Tensor | Sequence[Any]) -> torch.Tensor:
        return self.training_steps.extract_input_tensor(batch)

    def _get_sliding_window_inferer(self):
        return self.training_steps.get_sliding_window_inferer()

    def _pad_to_divisible(self, x: torch.Tensor) -> torch.Tensor:
        return self.training_steps.pad_to_divisible(x)

    def _get_decoder_last_layer(self) -> torch.nn.Parameter | None:
        return self.training_steps.get_decoder_last_layer()

    def _shared_step_generator(
        self,
        x_real: torch.Tensor,
        batch_idx: int,
        split: str,
        last_layer: torch.nn.Parameter | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.training_steps.shared_step_generator(
            x_real=x_real,
            batch_idx=batch_idx,
            split=split,
            global_step=self.global_step,
            last_layer=last_layer,
        )

    def _shared_step_discriminator(
        self,
        x_real: torch.Tensor,
        x_recon: torch.Tensor,
        vq_loss: torch.Tensor,
        split: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.training_steps.shared_step_discriminator(
            x_real=x_real,
            x_recon=x_recon,
            vq_loss=vq_loss,
            split=split,
            global_step=self.global_step,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return self.vqvae(x)  # type: ignore[no-any-return]

    def training_step(
        self,
        batch: torch.Tensor | Sequence[Any],
        batch_idx: int,
        optimizers: list[torch.optim.Optimizer] | None = None,
    ) -> None:
        resolved_optimizers = (
            cast(list[torch.optim.Optimizer], self.optimizers())
            if optimizers is None
            else optimizers
        )
        return self.training_steps.training_step(
            batch=batch,
            batch_idx=batch_idx,
            optimizers=resolved_optimizers,
            global_step=self.global_step,
            log_fn=self.log,
            manual_backward_fn=self.manual_backward,
        )

    def validation_step(
        self, batch: torch.Tensor | Sequence[Any], batch_idx: int
    ) -> dict[str, Any]:
        del batch_idx
        return self.training_steps.reconstruction_step(batch=batch, split="val", log_fn=self.log)

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        return self.training_steps.create_optimizers(
            lr_g=self.lr_g,
            lr_d=self.lr_d,
            optimizer_config=self.hparams.get("optimizer_config"),
            disc_optimizer_config=self.hparams.get("disc_optimizer_config"),
        )

    @torch.no_grad()
    def test_step(self, batch: torch.Tensor | Sequence[Any], batch_idx: int) -> dict[str, Any]:
        del batch_idx
        return self.training_steps.reconstruction_step(batch=batch, split="test", log_fn=self.log)

    def predict_step(self, batch: torch.Tensor | Sequence[Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self.training_steps.predict_step(batch)
