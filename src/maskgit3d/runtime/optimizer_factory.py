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
        optimizer_config: DictConfig | None = None,
        disc_optimizer_config: DictConfig | None = None,
    ) -> None:
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.optimizer_config = optimizer_config
        self.disc_optimizer_config = disc_optimizer_config

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
            opt_g = torch.optim.Adam(param_groups, lr=self.lr_g)

        if self.disc_optimizer_config is not None:
            opt_d = instantiate(self.disc_optimizer_config, _partial_=True)(
                discriminator.parameters(), lr=self.lr_d
            )
        else:
            opt_d = torch.optim.Adam(
                discriminator.parameters(),
                lr=self.lr_d,
            )

        return opt_g, opt_d


class TransformerOptimizerFactory:
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        optimizer_config: DictConfig | None = None,
    ) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.optimizer_config = optimizer_config

    def create_optimizer_and_scheduler(self, model: Any) -> dict[str, Any]:
        from .scheduler_factory import create_scheduler

        if self.optimizer_config is not None:
            optimizer = create_optimizer(model.parameters(), self.optimizer_config)
        else:
            optimizer = torch.optim.AdamW(
                model.transformer.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        scheduler_config = OmegaConf.create({"warmup_steps": self.warmup_steps})
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
