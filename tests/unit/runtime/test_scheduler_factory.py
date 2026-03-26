import math

import torch
from omegaconf import OmegaConf

from maskgit3d.runtime.scheduler_factory import create_scheduler


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


class TestCosineWarmupScheduler:
    """Tests for cosine decay after linear warmup."""

    def test_warmup_phase_linear_increase(self) -> None:
        """During warmup, LR should increase linearly from 0 to 1."""
        optimizer = torch.optim.Adam([torch.randn(2, 2)], lr=1e-4)
        config = OmegaConf.create({"warmup_steps": 10, "min_lr_ratio": 0.1, "total_steps": 100})
        scheduler = create_scheduler(optimizer, config)

        lrs = []
        for step_idx in range(11):
            lrs.append(scheduler.get_last_lr()[0] / 1e-4)
            if step_idx < 10:
                scheduler.step()

        assert abs(lrs[0] - 0.0) < 0.01
        assert abs(lrs[1] - 0.1) < 0.01
        assert abs(lrs[10] - 1.0) < 0.01

    def test_decay_phase_cosine(self) -> None:
        """After warmup, LR should decay with cosine schedule."""
        optimizer = torch.optim.Adam([torch.randn(2, 2)], lr=1e-4)
        config = OmegaConf.create({"warmup_steps": 10, "min_lr_ratio": 0.1, "total_steps": 110})
        scheduler = create_scheduler(optimizer, config)

        for _ in range(10):
            scheduler.step()

        lr_at_decay_start = scheduler.get_last_lr()[0] / 1e-4
        assert abs(lr_at_decay_start - 1.0) < 0.01

        for _ in range(50):
            scheduler.step()

        lr_mid = scheduler.get_last_lr()[0] / 1e-4
        assert abs(lr_mid - 0.55) < 0.05

        for _ in range(50):
            scheduler.step()

        lr_end = scheduler.get_last_lr()[0] / 1e-4
        assert abs(lr_end - 0.1) < 0.02

    def test_min_lr_ratio_respected(self) -> None:
        """LR should decay to min_lr_ratio, not to 0."""
        optimizer = torch.optim.Adam([torch.randn(2, 2)], lr=1e-4)
        config = OmegaConf.create({"warmup_steps": 5, "min_lr_ratio": 0.05, "total_steps": 105})
        scheduler = create_scheduler(optimizer, config)

        for _ in range(105):
            scheduler.step()

        lr_final = scheduler.get_last_lr()[0] / 1e-4
        assert abs(lr_final - 0.05) < 0.02

    def test_no_warmup_cosine_decay(self) -> None:
        """With no warmup, should start cosine decay from 1.0."""
        optimizer = torch.optim.Adam([torch.randn(2, 2)], lr=1e-4)
        config = OmegaConf.create({"warmup_steps": 0, "min_lr_ratio": 0.1, "total_steps": 100})
        scheduler = create_scheduler(optimizer, config)

        lr_start = scheduler.get_last_lr()[0] / 1e-4
        assert abs(lr_start - 1.0) < 0.01

        for _ in range(100):
            scheduler.step()

        lr_end = scheduler.get_last_lr()[0] / 1e-4
        assert abs(lr_end - 0.1) < 0.02

    def test_decay_formula_correctness(self) -> None:
        """Verify the cosine decay formula matches expected math."""
        optimizer = torch.optim.Adam([torch.randn(2, 2)], lr=1e-4)
        warmup_steps = 10
        total_steps = 110
        min_lr_ratio = 0.1

        config = OmegaConf.create(
            {
                "warmup_steps": warmup_steps,
                "min_lr_ratio": min_lr_ratio,
                "total_steps": total_steps,
            }
        )
        scheduler = create_scheduler(optimizer, config)
        lr_lambda = scheduler.lr_lambdas[0]

        test_cases = [
            (10, 1.0),
            (60, None),
            (109, None),
        ]

        for step, expected in test_cases:
            ratio = lr_lambda(step)
            if expected is not None:
                assert abs(ratio - expected) < 0.01, f"Step {step}: {ratio} != {expected}"
            else:
                if step == 60:
                    progress = (60 - warmup_steps) / (total_steps - warmup_steps)
                    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                    expected_ratio = min_lr_ratio + (1 - min_lr_ratio) * cosine_factor
                    assert abs(ratio - expected_ratio) < 0.01, (
                        f"Step 60: {ratio} != {expected_ratio}"
                    )
