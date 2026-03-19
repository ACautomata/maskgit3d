import torch
from omegaconf import OmegaConf

from src.maskgit3d.runtime.scheduler_factory import create_scheduler


def test_create_scheduler_builds_linear_warmup_lambda() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    config = OmegaConf.create({"warmup_steps": 10})

    scheduler = create_scheduler(optimizer, config)
    lr_lambda = scheduler.lr_lambdas[0]

    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert lr_lambda(0) == 0.0
    assert lr_lambda(5) == 0.5
    assert lr_lambda(10) == 1.0
