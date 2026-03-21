from __future__ import annotations

import math
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
        total_steps = int(scheduler_config.get("total_steps", 0))
        min_lr_ratio = float(scheduler_config.get("min_lr_ratio", 0.0))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            if total_steps <= warmup_steps:
                return 1.0

            progress = (step - warmup_steps) / float(total_steps - warmup_steps)
            progress = min(progress, 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: DictConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    factory: SchedulerFactoryProtocol = CosineWarmupSchedulerFactory()
    return factory.create(optimizer, scheduler_config)
