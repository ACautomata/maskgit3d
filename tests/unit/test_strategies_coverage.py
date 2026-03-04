import csv
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import maskgit3d.infrastructure.training.strategies as strategies


def _fake_monai_modules(perceptual_cls: type[nn.Module] | None = None) -> dict[str, object]:
    class _PSNRMetric:
        def __init__(self, max_val: float, reduction: str):
            self.value = torch.tensor([30.0])

        def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> None:
            self.value = torch.tensor([25.0])

        def aggregate(self) -> torch.Tensor:
            return self.value

        def reset(self) -> None:
            self.value = torch.tensor([0.0])

    class _SSIMMetric:
        def __init__(self, spatial_dims: int, data_range: float, reduction: str):
            self.value = torch.tensor([0.8])

        def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> None:
            self.value = torch.tensor([0.7])

        def aggregate(self) -> torch.Tensor:
            return self.value

        def reset(self) -> None:
            self.value = torch.tensor([0.0])

    monai_mod = types.ModuleType("monai")
    monai_metrics_mod = types.ModuleType("monai.metrics")
    monai_metrics_reg_mod = types.ModuleType("monai.metrics.regression")
    monai_losses_mod = types.ModuleType("monai.losses")
    monai_losses_perceptual_mod = types.ModuleType("monai.losses.perceptual")

    monai_metrics_reg_mod.PSNRMetric = _PSNRMetric
    monai_metrics_reg_mod.SSIMMetric = _SSIMMetric
    if perceptual_cls is not None:
        monai_losses_perceptual_mod.PerceptualLoss = perceptual_cls

    return {
        "monai": monai_mod,
        "monai.metrics": monai_metrics_mod,
        "monai.metrics.regression": monai_metrics_reg_mod,
        "monai.losses": monai_losses_mod,
        "monai.losses.perceptual": monai_losses_perceptual_mod,
    }


class _RecordingLPIPS:
    def __init__(self) -> None:
        self.called = False
        self.inputs: tuple[torch.Tensor, torch.Tensor] | None = None

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.called = True
        self.inputs = (x, y)
        return torch.tensor([0.25])


class _TinyDisc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1).mean(dim=1) * self.scale


class _TinyVQModel(nn.Module):
    def forward_with_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x * 0.5, torch.tensor(0.1)


def test_mixed_precision_init_enabled_float16_creates_scaler() -> None:
    fake_scaler = MagicMock()
    with (
        patch(
            "maskgit3d.infrastructure.training.strategies.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "maskgit3d.infrastructure.training.strategies.torch.cuda.amp.GradScaler",
            return_value=fake_scaler,
        ),
    ):
        trainer = strategies.MixedPrecisionTrainer(enabled=True, dtype="float16")

    assert trainer.enabled is True
    assert trainer.scaler is fake_scaler


def test_mixed_precision_init_enabled_bfloat16_uses_no_scaler() -> None:
    with patch(
        "maskgit3d.infrastructure.training.strategies.torch.cuda.is_available", return_value=True
    ):
        trainer = strategies.MixedPrecisionTrainer(enabled=True, dtype="bfloat16")

    assert trainer.enabled is True
    assert trainer.scaler is None


def test_mixed_precision_autocast_context_enabled_dtype() -> None:
    sentinel = object()
    with (
        patch(
            "maskgit3d.infrastructure.training.strategies.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "maskgit3d.infrastructure.training.strategies.torch.cuda.amp.GradScaler",
            return_value=MagicMock(),
        ),
        patch(
            "maskgit3d.infrastructure.training.strategies.torch.autocast", return_value=sentinel
        ) as mocked_autocast,
    ):
        trainer = strategies.MixedPrecisionTrainer(enabled=True, dtype="float16")
        ctx = trainer.autocast_context()

    assert ctx is sentinel
    mocked_autocast.assert_called_once_with(device_type="cuda", dtype=torch.float16, enabled=True)


def test_mixed_precision_scale_loss_enabled_fp16_uses_scaler_scale() -> None:
    fake_scaler = MagicMock()
    fake_scaler.scale.return_value = torch.tensor(42.0)
    with (
        patch(
            "maskgit3d.infrastructure.training.strategies.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "maskgit3d.infrastructure.training.strategies.torch.cuda.amp.GradScaler",
            return_value=fake_scaler,
        ),
    ):
        trainer = strategies.MixedPrecisionTrainer(enabled=True, dtype="float16")

    loss = torch.tensor(1.0)
    scaled = trainer.scale_loss(loss)
    fake_scaler.scale.assert_called_once_with(loss)
    assert scaled.item() == 42.0


def test_mixed_precision_step_optimizer_fp16_path() -> None:
    fake_scaler = MagicMock()
    with (
        patch(
            "maskgit3d.infrastructure.training.strategies.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "maskgit3d.infrastructure.training.strategies.torch.cuda.amp.GradScaler",
            return_value=fake_scaler,
        ),
    ):
        trainer = strategies.MixedPrecisionTrainer(enabled=True, dtype="float16", grad_clip=0.5)

    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with patch(
        "maskgit3d.infrastructure.training.strategies.torch.nn.utils.clip_grad_norm_"
    ) as clip_grad:
        trainer.step_optimizer(optimizer, torch.tensor(1.0))

    fake_scaler.unscale_.assert_called_once_with(optimizer)
    clip_grad.assert_called_once()
    fake_scaler.step.assert_called_once_with(optimizer)
    fake_scaler.update.assert_called_once()


def test_mixed_precision_step_optimizer_bf16_path() -> None:
    with patch(
        "maskgit3d.infrastructure.training.strategies.torch.cuda.is_available", return_value=True
    ):
        trainer = strategies.MixedPrecisionTrainer(enabled=True, dtype="bfloat16", grad_clip=1.0)

    model = nn.Linear(3, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with (
        patch(
            "maskgit3d.infrastructure.training.strategies.torch.nn.utils.clip_grad_norm_"
        ) as clip_grad,
        patch.object(optimizer, "step") as step_spy,
    ):
        trainer.step_optimizer(optimizer, torch.tensor(1.0))

    clip_grad.assert_called_once()
    step_spy.assert_called_once()


def test_mixed_precision_step_optimizer_fp16_without_clip_skips_clip_grad() -> None:
    fake_scaler = MagicMock()
    with (
        patch(
            "maskgit3d.infrastructure.training.strategies.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "maskgit3d.infrastructure.training.strategies.torch.cuda.amp.GradScaler",
            return_value=fake_scaler,
        ),
    ):
        trainer = strategies.MixedPrecisionTrainer(enabled=True, dtype="float16", grad_clip=None)

    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with patch(
        "maskgit3d.infrastructure.training.strategies.torch.nn.utils.clip_grad_norm_"
    ) as clip_grad:
        trainer.step_optimizer(optimizer, torch.tensor(1.0))

    clip_grad.assert_not_called()


def test_mixed_precision_state_dict_and_load_state_dict_with_scaler() -> None:
    fake_scaler = MagicMock()
    fake_scaler.state_dict.return_value = {"scale": 1024.0}
    with (
        patch(
            "maskgit3d.infrastructure.training.strategies.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "maskgit3d.infrastructure.training.strategies.torch.cuda.amp.GradScaler",
            return_value=fake_scaler,
        ),
    ):
        trainer = strategies.MixedPrecisionTrainer(enabled=True, dtype="float16")

    state = trainer.state_dict()
    assert state["scaler"] == {"scale": 1024.0}

    trainer.load_state_dict({"enabled": True, "dtype": "float16", "scaler": {"scale": 512.0}})
    fake_scaler.load_state_dict.assert_called_once_with({"scale": 512.0})


def test_vqgan_strategy_initializes_lpips_when_enabled() -> None:
    class _FakeLPIPS(nn.Module):
        def __init__(self, net: str, verbose: bool):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.net = net
            self.verbose = verbose

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.tensor([0.1])

    lpips_mod = types.ModuleType("lpips")
    lpips_ctor = MagicMock(side_effect=_FakeLPIPS)
    lpips_mod.LPIPS = lpips_ctor

    with patch.dict(sys.modules, {"lpips": lpips_mod}):
        strategy = strategies.VQGANTrainingStrategy(perceptual_weight=1.0)

    lpips_ctor.assert_called_once_with(net="vgg", verbose=False)
    assert strategy.lpips_fn is not None
    assert all(not p.requires_grad for p in strategy.lpips_fn.parameters())


def test_compute_perceptual_loss_5d_path() -> None:
    strategy = strategies.VQGANTrainingStrategy(perceptual_weight=0.0)
    recorder = _RecordingLPIPS()
    strategy.lpips_fn = recorder
    strategy.perceptual_weight = 1.0

    x = torch.rand(2, 1, 5, 32, 32)
    xrec = torch.rand(2, 1, 5, 32, 32)
    loss = strategy._compute_perceptual_loss(x, xrec)

    assert recorder.called is True
    assert recorder.inputs is not None
    assert recorder.inputs[0].shape == (2, 3, 32, 32)
    assert loss.item() == 0.25


def test_compute_perceptual_loss_5d_small_spatial_returns_zero() -> None:
    strategy = strategies.VQGANTrainingStrategy(perceptual_weight=0.0)
    recorder = _RecordingLPIPS()
    strategy.lpips_fn = recorder
    strategy.perceptual_weight = 1.0

    x = torch.rand(2, 1, 5, 16, 16)
    xrec = torch.rand(2, 1, 5, 16, 16)
    loss = strategy._compute_perceptual_loss(x, xrec)

    assert loss.item() == 0.0
    assert recorder.called is False


def test_compute_perceptual_loss_2d_path() -> None:
    strategy = strategies.VQGANTrainingStrategy(perceptual_weight=0.0)
    recorder = _RecordingLPIPS()
    strategy.lpips_fn = recorder
    strategy.perceptual_weight = 1.0

    x = torch.rand(2, 1, 32, 32)
    xrec = torch.rand(2, 1, 32, 32)
    loss = strategy._compute_perceptual_loss(x, xrec)

    assert recorder.called is True
    assert recorder.inputs is not None
    assert recorder.inputs[0].shape == (2, 3, 32, 32)
    assert loss.item() == 0.25


def test_compute_perceptual_loss_2d_small_spatial_returns_zero() -> None:
    strategy = strategies.VQGANTrainingStrategy(perceptual_weight=0.0)
    recorder = _RecordingLPIPS()
    strategy.lpips_fn = recorder
    strategy.perceptual_weight = 1.0

    x = torch.rand(2, 1, 16, 16)
    xrec = torch.rand(2, 1, 16, 16)
    loss = strategy._compute_perceptual_loss(x, xrec)

    assert loss.item() == 0.0
    assert recorder.called is False


def test_compute_adversarial_loss_generator_path() -> None:
    disc = _TinyDisc()
    strategy = strategies.VQGANTrainingStrategy(
        discriminator=disc, disc_weight=1.0, perceptual_weight=0.0
    )
    x = torch.ones(2, 1, 4, 4)
    xrec = torch.full((2, 1, 4, 4), 0.5)

    loss, metrics = strategy._compute_adversarial_loss(x, xrec, optimizer_idx=0)

    assert "g_loss" in metrics
    assert torch.is_tensor(loss)
    assert loss.item() == metrics["g_loss"]


def test_compute_adversarial_loss_discriminator_path() -> None:
    disc = _TinyDisc()
    strategy = strategies.VQGANTrainingStrategy(
        discriminator=disc, disc_weight=1.0, perceptual_weight=0.0
    )
    x = torch.ones(2, 1, 4, 4)
    xrec = torch.full((2, 1, 4, 4), -0.5)

    loss, metrics = strategy._compute_adversarial_loss(x, xrec, optimizer_idx=1)

    assert "d_loss" in metrics
    assert "d_loss_real" in metrics
    assert "d_loss_fake" in metrics
    assert loss.item() == metrics["d_loss"]


def test_train_discriminator_step_runs_full_path() -> None:
    disc = _TinyDisc()
    strategy = strategies.VQGANTrainingStrategy(
        discriminator=disc,
        disc_weight=1.0,
        disc_start=0,
        perceptual_weight=0.0,
    )
    model = _TinyVQModel()
    optimizer = torch.optim.SGD(disc.parameters(), lr=0.1)
    batch = (torch.randn(2, 1, 8, 8),)

    metrics = strategy.train_discriminator_step(model, batch, optimizer)

    assert "d_loss" in metrics
    assert "d_loss_real" in metrics
    assert "d_loss_fake" in metrics


def test_train_discriminator_step_early_return_before_disc_start() -> None:
    disc = _TinyDisc()
    strategy = strategies.VQGANTrainingStrategy(
        discriminator=disc,
        disc_weight=1.0,
        disc_start=5,
        perceptual_weight=0.0,
    )
    model = _TinyVQModel()
    optimizer = torch.optim.SGD(disc.parameters(), lr=0.1)
    batch = (torch.randn(2, 1, 8, 8),)

    metrics = strategy.train_discriminator_step(model, batch, optimizer)
    assert metrics == {}


def test_vqgan_metrics_init_enable_lpips_with_monai_perceptual_loss() -> None:
    class _FakePerceptualLoss(nn.Module):
        def __init__(
            self, spatial_dims: int, network_type: str, is_fake_3d: bool, fake_3d_ratio: float
        ):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.tensor([0.2])

    with patch.dict(sys.modules, _fake_monai_modules(perceptual_cls=_FakePerceptualLoss)):
        metrics = strategies.VQGANMetrics(data_range=1.0, spatial_dims=2, enable_lpips=True)

    assert metrics.lpips_loss is not None
    assert all(not p.requires_grad for p in metrics.lpips_loss.parameters())


def test_vqgan_metrics_update_with_tensor_inputs() -> None:
    class _FakePerceptualLoss(nn.Module):
        def __init__(
            self, spatial_dims: int, network_type: str, is_fake_3d: bool, fake_3d_ratio: float
        ):
            super().__init__()

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.tensor([0.15])

    with patch.dict(sys.modules, _fake_monai_modules(perceptual_cls=_FakePerceptualLoss)):
        metrics = strategies.VQGANMetrics(data_range=1.0, spatial_dims=2, enable_lpips=True)

    pred = torch.rand(2, 1, 32, 32) * 2 - 1
    target = torch.rand(2, 1, 32, 32) * 2 - 1
    metrics.update(pred, target)

    assert metrics._count == 1
    assert len(metrics.psnr_values) == 1
    assert len(metrics.ssim_values) == 1
    assert len(metrics.lpips_values) == 1


def test_vqgan_metrics_update_with_numpy_dict_inputs() -> None:
    with patch.dict(sys.modules, _fake_monai_modules()):
        metrics = strategies.VQGANMetrics(data_range=1.0, spatial_dims=2, enable_lpips=False)

    pred_dict = {"images": (np.random.rand(2, 1, 32, 32).astype(np.float32) * 2) - 1}
    target = (np.random.rand(2, 1, 32, 32).astype(np.float32) * 2) - 1
    metrics.update(pred_dict, target)

    assert metrics._count == 1
    assert len(metrics.psnr_values) == 1


def test_vqgan_metrics_safe_item_tuple_input() -> None:
    with patch.dict(sys.modules, _fake_monai_modules()):
        metrics = strategies.VQGANMetrics(data_range=1.0, spatial_dims=2, enable_lpips=False)

    result = metrics._safe_item((torch.tensor([1.0, 3.0]), torch.tensor([0.0])))
    assert result == 2.0


def test_vqgan_metrics_safe_item_multi_element_tensor() -> None:
    with patch.dict(sys.modules, _fake_monai_modules()):
        metrics = strategies.VQGANMetrics(data_range=1.0, spatial_dims=2, enable_lpips=False)

    result = metrics._safe_item(torch.tensor([1.0, 2.0, 3.0]))
    assert result == 2.0


def test_vqgan_metrics_compute_ssim_fallback_torchmetrics_success() -> None:
    with patch.dict(sys.modules, _fake_monai_modules()):
        metrics = strategies.VQGANMetrics(data_range=1.0, spatial_dims=2, enable_lpips=False)

    torchmetrics_mod = types.ModuleType("torchmetrics")
    torchmetrics_func_mod = types.ModuleType("torchmetrics.functional")

    def _fake_ssim(
        img1: torch.Tensor, img2: torch.Tensor, data_range: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(0.88), torch.tensor(0.0)

    torchmetrics_func_mod.structural_similarity_index_measure = _fake_ssim

    with patch.dict(
        sys.modules,
        {
            "torchmetrics": torchmetrics_mod,
            "torchmetrics.functional": torchmetrics_func_mod,
        },
    ):
        val = metrics._compute_ssim_fallback(torch.rand(1, 16, 16), torch.rand(1, 16, 16))

    assert torch.is_tensor(val)
    assert val.item() == pytest.approx(0.88)


def test_vqgan_metrics_compute_ssim_fallback_importerror_path() -> None:
    with patch.dict(sys.modules, _fake_monai_modules()):
        metrics = strategies.VQGANMetrics(data_range=1.0, spatial_dims=2, enable_lpips=False)

    with patch.dict(sys.modules, {"torchmetrics": None, "torchmetrics.functional": None}):
        val = metrics._compute_ssim_fallback(torch.zeros(1, 16, 16), torch.ones(1, 16, 16))

    assert torch.is_tensor(val)
    assert val.item() <= 1.0


def test_vqgan_metrics_compute_with_ssim_and_lpips_values() -> None:
    with patch.dict(sys.modules, _fake_monai_modules()):
        metrics = strategies.VQGANMetrics(data_range=1.0, spatial_dims=2, enable_lpips=False)

    metrics.psnr_values = [10.0, 14.0]
    metrics.ssim_values = [0.7, 0.9]
    metrics.lpips_values = [0.2, 0.4]

    out = metrics.compute()

    assert out["psnr"] == 12.0
    assert out["ssim"] == 0.8
    assert out["lpips"] == 0.30000000000000004
    assert "lpips_std" in out


def test_vqgan_metrics_export_csv_with_conditional_columns(tmp_path) -> None:
    with patch.dict(sys.modules, _fake_monai_modules()):
        metrics = strategies.VQGANMetrics(data_range=1.0, spatial_dims=2, enable_lpips=False)

    metrics.psnr_values = [20.0]
    metrics.ssim_values = []
    metrics.lpips_values = [0.1, 0.2]

    csv_path = tmp_path / "metrics.csv"
    metrics.export_csv(str(csv_path))

    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["batch", "psnr", "lpips"]
    assert rows[1][0] == "0"
    assert rows[1][1] == "20.0"
    assert rows[1][2] == "0.1"
    assert rows[2][0] == "1"
    assert rows[2][1] == "0.2"
