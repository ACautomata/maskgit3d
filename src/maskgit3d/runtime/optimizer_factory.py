from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

import torch
from hydra.utils import instantiate
from hydra.utils import get_class
from omegaconf import DictConfig
from omegaconf import OmegaConf


class OptimizerFactoryProtocol(Protocol):
    def create(
        self,
        params: Iterable[Any],
        optimizer_config: DictConfig,
    ) -> torch.optim.Optimizer: ...


def _optimizer_kwargs(optimizer_config: DictConfig) -> dict[str, Any]:
    optimizer_kwargs: dict[str, Any] = {}
    for key, value in optimizer_config.items():
        if key != "_target_":
            optimizer_kwargs[str(key)] = value
    return optimizer_kwargs


class AdamOptimizerFactory:
    def create(
        self,
        params: Iterable[Any],
        optimizer_config: DictConfig,
    ) -> torch.optim.Adam:
        return torch.optim.Adam(params, **_optimizer_kwargs(optimizer_config))


class AdamWOptimizerFactory:
    def create(
        self,
        params: Iterable[Any],
        optimizer_config: DictConfig,
    ) -> torch.optim.AdamW:
        return torch.optim.AdamW(params, **_optimizer_kwargs(optimizer_config))


class GANOptimizerFactory:
    def __init__(
        self,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        weight_decay_g: float = 1e-4,
        weight_decay_d: float = 0.0,
        optimizer_config: DictConfig | None = None,
        disc_optimizer_config: DictConfig | None = None,
        warmup_steps: int = 20,
        max_epochs: int = 100,
        steps_per_epoch: int = 1000,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.weight_decay_g = weight_decay_g
        self.weight_decay_d = weight_decay_d
        self.optimizer_config = optimizer_config
        self.disc_optimizer_config = disc_optimizer_config
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.min_lr_ratio = min_lr_ratio

    def create_optimizers(
        self,
        generator: Any,
        discriminator: torch.nn.Module,
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        param_groups = [
            {"params": generator.encoder.parameters(), "lr": self.lr_g},
            {"params": generator.quant_conv.parameters(), "lr": self.lr_g},
            {"params": generator.post_quant_conv.parameters(), "lr": self.lr_g},
            {"params": generator.decoder.parameters(), "lr": self.lr_g},
        ]

        if self.optimizer_config is not None:
            opt_g = instantiate(self.optimizer_config, _partial_=True)(param_groups, lr=self.lr_g)
        else:
            # Default to AdamW with betas=(0.9, 0.9) and weight_decay for stable GAN training
            opt_g = torch.optim.AdamW(
                param_groups, lr=self.lr_g, betas=(0.9, 0.9), weight_decay=self.weight_decay_g
            )

        if self.disc_optimizer_config is not None:
            opt_d = instantiate(self.disc_optimizer_config, _partial_=True)(
                discriminator.parameters(), lr=self.lr_d
            )
        else:
            # Default to AdamW with betas=(0.9, 0.9) for stable GAN training
            # weight_decay=0.0 for discriminator as per GAN best practices
            opt_d = torch.optim.AdamW(
                discriminator.parameters(),
                lr=self.lr_d,
                betas=(0.9, 0.9),
                weight_decay=self.weight_decay_d,
            )

        return opt_g, opt_d

    def create_schedulers(
        self,
        opt_g: torch.optim.Optimizer,
        opt_d: torch.optim.Optimizer,
        total_steps: int | None = None,
    ) -> tuple[torch.optim.lr_scheduler.LambdaLR, torch.optim.lr_scheduler.LambdaLR]:
        """Create schedulers for generator and discriminator optimizers.

        Args:
            opt_g: Generator optimizer
            opt_d: Discriminator optimizer
            total_steps: Optional total training steps. If None, calculated from
                         max_epochs * steps_per_epoch. Prefer using Lightning's
                         trainer.estimated_stepping_batches when available.
        """
        from .scheduler_factory import create_scheduler

        effective_total_steps = (
            total_steps if total_steps is not None else self.max_epochs * self.steps_per_epoch
        )
        scheduler_config = OmegaConf.create(
            {
                "warmup_steps": self.warmup_steps,
                "min_lr_ratio": self.min_lr_ratio,
                "total_steps": effective_total_steps,
            }
        )

        scheduler_g = create_scheduler(opt_g, scheduler_config)
        scheduler_d = create_scheduler(opt_d, scheduler_config)

        return scheduler_g, scheduler_d


class TransformerOptimizerFactory:
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        optimizer_config: DictConfig | None = None,
        max_epochs: int = 100,
        steps_per_epoch: int = 1000,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.optimizer_config = optimizer_config
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.min_lr_ratio = min_lr_ratio

    def create_optimizer_and_scheduler(
        self, model: Any, total_steps: int | None = None
    ) -> dict[str, Any]:
        from .scheduler_factory import create_scheduler

        if self.optimizer_config is not None:
            optimizer = create_optimizer(model.parameters(), self.optimizer_config)
        else:
            optimizer = torch.optim.AdamW(
                model.transformer.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        effective_total_steps = (
            total_steps if total_steps is not None else self.max_epochs * self.steps_per_epoch
        )
        scheduler_config = OmegaConf.create(
            {
                "warmup_steps": self.warmup_steps,
                "min_lr_ratio": self.min_lr_ratio,
                "total_steps": effective_total_steps,
            }
        )
        scheduler = create_scheduler(optimizer, scheduler_config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def create_optimizer(
    params: Iterable[Any],
    optimizer_config: DictConfig,
) -> torch.optim.Optimizer:
    optimizer_target = optimizer_config.get("_target_")
    if optimizer_target is None:
        raise ValueError("optimizer_config._target_ must be set.")

    optimizer_class = get_class(str(optimizer_target))
    if optimizer_class is torch.optim.Adam:
        factory: OptimizerFactoryProtocol = AdamOptimizerFactory()
        return factory.create(params, optimizer_config)
    if optimizer_class is torch.optim.AdamW:
        factory = AdamWOptimizerFactory()
        return factory.create(params, optimizer_config)

    return optimizer_class(params, **_optimizer_kwargs(optimizer_config))
