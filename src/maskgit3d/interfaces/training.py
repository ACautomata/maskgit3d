from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@runtime_checkable
class OptimizerFactoryProtocol(Protocol):
    def create_optimizer(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
    ) -> Optimizer: ...

    def create_generator_optimizer(self, model: torch.nn.Module, lr: float) -> Optimizer: ...

    def create_discriminator_optimizer(
        self,
        model: torch.nn.Module,
        lr: float,
    ) -> Optimizer: ...


@runtime_checkable
class SchedulerFactoryProtocol(Protocol):
    def create_scheduler(self, optimizer: Optimizer, warmup_steps: int) -> LRScheduler: ...
