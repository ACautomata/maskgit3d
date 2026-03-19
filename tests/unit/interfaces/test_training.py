from collections.abc import Iterable

import torch

from src.maskgit3d.interfaces.training import OptimizerFactoryProtocol, SchedulerFactoryProtocol


class DummyOptimizerFactory:
    def create_optimizer(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
    ) -> torch.optim.Optimizer:
        return torch.optim.SGD(params, lr=lr)

    def create_generator_optimizer(
        self,
        model: torch.nn.Module,
        lr: float,
    ) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=lr)

    def create_discriminator_optimizer(
        self,
        model: torch.nn.Module,
        lr: float,
    ) -> torch.optim.Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=lr)


class DummySchedulerFactory:
    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        del warmup_steps
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)


def test_optimizer_factory_protocol() -> None:
    factory = DummyOptimizerFactory()

    assert isinstance(factory, OptimizerFactoryProtocol)


def test_scheduler_factory_protocol() -> None:
    factory = DummySchedulerFactory()

    assert isinstance(factory, SchedulerFactoryProtocol)
