from __future__ import annotations

from typing import Protocol

import torch
from omegaconf import DictConfig


class SchedulerFactoryProtocol(Protocol):
    def create(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_config: DictConfig,
    ) -> torch.optim.lr_scheduler.LambdaLR: ...


class CosineWarmupSchedulerFactory:
    def create(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_config: DictConfig,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        warmup_steps = int(scheduler_config.get("warmup_steps", 0))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: DictConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    factory: SchedulerFactoryProtocol = CosineWarmupSchedulerFactory()
    return factory.create(optimizer, scheduler_config)
