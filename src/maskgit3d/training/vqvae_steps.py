from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig

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
        log_fn: Callable[..., None] | None = None,
        manual_backward_fn: Callable[[torch.Tensor], None] | None = None,
    ) -> None:
        logger = log_fn or self.log_fn
        backward = manual_backward_fn or self.manual_backward_fn
        if logger is None or backward is None:
            raise ValueError("training_step requires log_fn and manual_backward_fn.")

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
        self._log_generator_metrics(logger, log_g, x_real, x_recon)

        opt_g.zero_grad()
        backward(loss_g)
        self.gan_strategy.step_generator(opt_g, loss_g, self.vqvae)

        x_recon_detached = x_recon.detach()
        del loss_g, log_g, x_recon

        loss_d, log_d = self.shared_step_discriminator(
            x_real=x_real,
            x_recon=x_recon_detached,
            vq_loss=vq_loss,
            split="train",
            global_step=global_step,
        )
        self._log_discriminator_metrics(logger, log_d, x_real.shape[0])

        opt_d.zero_grad()
        backward(loss_d)
        self.gan_strategy.step_discriminator(opt_d, loss_d)

        del x_real, x_recon_detached, loss_d, log_d, vq_loss, batch_idx
        return None

    def reconstruction_step(
        self,
        batch: torch.Tensor | Sequence[Any],
        split: str,
        log_fn: Callable[..., None] | None = None,
    ) -> dict[str, Any]:
        logger = log_fn or self.log_fn
        if logger is None:
            raise ValueError("reconstruction_step requires log_fn.")

        x_real = self.extract_input_tensor(batch)
        inferer = self.get_sliding_window_inferer()

        if split == "test" and inferer is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        x_recon = self.reconstructor.reconstruct(self.vqvae, x_real)
        batch_size = x_real.shape[0]

        if split == "val":
            perceptual_loss = torch.tensor(0.0, device=x_real.device)
            if self.loss_fn.use_perceptual and self.loss_fn.perceptual_loss is not None:
                perceptual_loss = self.loss_fn.perceptual_loss(x_recon, x_real)
            logger(
                "val_perceptual_loss",
                perceptual_loss.item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
            rec_loss = F.l1_loss(x_recon, x_real)
            logger(
                "val_rec_loss",
                rec_loss.item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
            return {
                "x_real": x_real.detach().cpu(),
                "x_recon": x_recon.detach().cpu(),
            }

        loss = F.l1_loss(x_recon, x_real)
        return {
            "loss": loss.item(),
            "inference_time": 0.0,
            "use_sliding_window": inferer is not None,
            "x_real": x_real.detach().cpu(),
            "x_recon": x_recon.detach().cpu(),
        }

    def predict_step(self, batch: torch.Tensor | Sequence[Any]) -> torch.Tensor:
        x_real = self.extract_input_tensor(batch)
        x_recon, _ = self.vqvae(x_real)
        return x_recon

    def _log_generator_metrics(
        self,
        logger: Callable[..., None],
        metrics: dict[str, torch.Tensor],
        x_real: torch.Tensor,
        x_recon: torch.Tensor,
    ) -> None:
        prog_bar_keys = {"total_loss", "vq_loss", "g_loss"}
        batch_size = x_real.shape[0]
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                metric_name = key.split("/")[-1]
                logger(
                    key,
                    value.item(),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=metric_name in prog_bar_keys,
                    batch_size=batch_size,
                )

        for name, value in {
            "train/x_recon_min": x_recon.min().item(),
            "train/x_recon_max": x_recon.max().item(),
            "train/x_recon_abs_mean": x_recon.abs().mean().item(),
            "train/x_real_min": x_real.min().item(),
            "train/x_real_max": x_real.max().item(),
        }.items():
            logger(name, value, on_step=True, on_epoch=False, prog_bar=False, batch_size=batch_size)

    def _log_discriminator_metrics(
        self,
        logger: Callable[..., None],
        metrics: dict[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                logger(
                    key,
                    value.item(),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=key.split("/")[-1] == "disc_loss",
                    batch_size=batch_size,
                )

    def create_optimizers(
        self,
        lr_g: float,
        lr_d: float,
        optimizer_config: DictConfig | None = None,
        disc_optimizer_config: DictConfig | None = None,
    ) -> list[torch.optim.Optimizer]:
        if optimizer_config is not None:
            param_groups = [
                {"params": self.vqvae.encoder.parameters(), "lr": lr_g},
                {"params": self.vqvae.quant_conv.parameters(), "lr": lr_g},
                {"params": self.vqvae.post_quant_conv.parameters(), "lr": lr_g},
                {"params": self.vqvae.decoder.parameters(), "lr": lr_g},
            ]
            opt_g = instantiate(optimizer_config, _partial_=True)(param_groups, lr=lr_g)
        else:
            opt_g = torch.optim.Adam(
                list(self.vqvae.encoder.parameters())
                + list(self.vqvae.quant_conv.parameters())
                + list(self.vqvae.post_quant_conv.parameters())
                + list(self.vqvae.decoder.parameters()),
                lr=lr_g,
            )

        if disc_optimizer_config is not None:
            opt_d = instantiate(disc_optimizer_config, _partial_=True)(
                self.loss_fn.discriminator.parameters(), lr=lr_d
            )
        else:
            opt_d = torch.optim.Adam(
                self.loss_fn.discriminator.parameters(),
                lr=lr_d,
            )

        return [opt_g, opt_d]
