from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

import torch
from hydra.utils import get_class
from omegaconf import DictConfig


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
