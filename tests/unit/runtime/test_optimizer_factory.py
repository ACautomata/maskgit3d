import torch
from omegaconf import OmegaConf

from src.maskgit3d.runtime.optimizer_factory import create_optimizer


def test_create_optimizer_keeps_param_group_learning_rates() -> None:
    layer = torch.nn.Linear(4, 2)
    config = OmegaConf.create(
        {
            "_target_": "torch.optim.Adam",
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0,
        }
    )

    optimizer = create_optimizer(
        [
            {"params": layer.weight, "lr": 2e-4},
            {"params": layer.bias, "lr": 5e-4},
        ],
        config,
    )

    assert isinstance(optimizer, torch.optim.Adam)
    assert [group["lr"] for group in optimizer.param_groups] == [2e-4, 5e-4]
