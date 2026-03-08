"""Tests for MaskGITTask."""

from pathlib import Path

import pytest
import torch

from src.maskgit3d.tasks.maskgit_task import MaskGITTask
from src.maskgit3d.models.vqvae import VQVAE


@pytest.fixture
def vqvae_checkpoint(tmp_path: Path) -> str:
    vqvae = VQVAE()
    ckpt_path = str(tmp_path / "vqvae.ckpt")
    torch.save({"state_dict": vqvae.state_dict()}, ckpt_path)
    return ckpt_path


def test_maskgit_task_init(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    assert task.maskgit is not None
    assert task.vqvae is not None
    assert not any(p.requires_grad for p in task.vqvae.parameters())


def test_maskgit_task_configure_optimizers(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    optimizers, schedulers = task.configure_optimizers()
    assert len(optimizers) == 1
    assert isinstance(optimizers[0], torch.optim.AdamW)
    assert len(schedulers) == 1


def test_maskgit_task_forward(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()
    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        output = task(x)
    assert output.shape == x.shape


def test_maskgit_task_encode_images_to_tokens(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()
    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        tokens = task.encode_images_to_tokens(x)
    assert tokens.dim() == 4
    assert tokens.dtype == torch.long


def test_maskgit_task_compute_loss_from_tokens(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()
    tokens = torch.randint(0, 100, (2, 4, 4, 4))
    loss, metrics = task._compute_loss_from_tokens(tokens, mask_ratio=0.5)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    assert "mask_acc" in metrics
    assert "mask_ratio" in metrics


def test_maskgit_task_training_step(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    x = torch.randn(1, 1, 16, 16, 16)
    loss = task.training_step(x, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_maskgit_task_validation_step(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    x = torch.randn(1, 1, 16, 16, 16)
    task.validation_step(x, 0)


def test_maskgit_task_test_step(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    x = torch.randn(1, 1, 16, 16, 16)
    task.test_step(x, 0)


def test_maskgit_task_sliding_window_disabled(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=1e-4,
        sliding_window={"enabled": False},
    )
    inferer = task._get_sliding_window_inferer()
    assert inferer is None


def test_maskgit_task_sliding_window_enabled(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=1e-4,
        sliding_window={"enabled": True, "roi_size": [32, 32, 32], "overlap": 0.25},
    )
    inferer = task._get_sliding_window_inferer()
    assert inferer is not None
    assert inferer.roi_size == (32, 32, 32)
    assert inferer.overlap == 0.25


def test_maskgit_task_load_vqvae_without_state_dict_key(tmp_path: Path):
    vqvae = VQVAE()
    ckpt_path = str(tmp_path / "vqvae_direct.ckpt")
    torch.save(vqvae.state_dict(), ckpt_path)

    task = MaskGITTask(
        vqvae_ckpt_path=ckpt_path, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    assert task.maskgit is not None
    assert task.vqvae is not None


def test_maskgit_task_checkpoint_reload_does_not_require_original_vqvae_checkpoint(
    tmp_path: Path,
    vqvae_checkpoint: str,
):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=1e-4,
    )
    checkpoint_path = tmp_path / "maskgit-task.ckpt"
    torch.save(
        {
            "state_dict": task.state_dict(),
            "hyper_parameters": dict(task.hparams),
            "pytorch-lightning_version": "2.5.0",
        },
        checkpoint_path,
    )

    Path(vqvae_checkpoint).unlink()

    reloaded_task = MaskGITTask.load_from_checkpoint(str(checkpoint_path))

    assert reloaded_task.vqvae is not None
    assert reloaded_task.maskgit is not None


def test_maskgit_task_get_divisible_pad(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )

    pad1 = task._get_divisible_pad()
    pad2 = task._get_divisible_pad()

    assert pad1 is not None
    assert pad1 is pad2


def test_maskgit_task_warmup_scheduler(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=1e-4,
        warmup_steps=10,
    )

    optimizers, schedulers = task.configure_optimizers()
    assert len(schedulers) == 1


def test_maskgit_task_validation_step_with_logging(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    class MockLogger:
        def __init__(self):
            self.logs = {}

        def __call__(self, name, value, **kwargs):
            self.logs[name] = value

    task.log = MockLogger()  # type: ignore[assignment]

    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        task.validation_step(x, 0)


def test_maskgit_task_test_step_with_logging(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    class MockLogger:
        def __init__(self):
            self.logs = {}

        def __call__(self, name, value, **kwargs):
            self.logs[name] = value

    task.log = MockLogger()  # type: ignore[assignment]

    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        task.test_step(x, 0)


def test_maskgit_task_test_step_with_sliding_window_logging(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=1e-4,
        sliding_window={"enabled": True, "roi_size": [16, 16, 16], "overlap": 0.25},
    )
    task.eval()

    logs: dict[str, float] = {}
    task.log = lambda name, value, **kwargs: logs.update({name: value})  # type: ignore[assignment]

    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        task.test_step(x, 0)

    assert "test/sliding_window_enabled" in logs


def test_maskgit_task_training_step_with_tuple_batch(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )

    x = torch.randn(1, 1, 16, 16, 16)
    batch = (x, torch.tensor([0]))

    loss = task.training_step(batch, 0)  # type: ignore[arg-type]
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_maskgit_task_validation_step_with_tuple_batch(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    x = torch.randn(1, 1, 16, 16, 16)
    batch = (x, torch.tensor([0]))

    with torch.no_grad():
        task.validation_step(batch, 0)  # type: ignore[arg-type]


def test_maskgit_task_test_step_with_tuple_batch(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    x = torch.randn(1, 1, 16, 16, 16)
    batch = (x, torch.tensor([0]))

    with torch.no_grad():
        task.test_step(batch, 0)  # type: ignore[arg-type]


def test_maskgit_task_compute_loss_from_tokens_random_mask_ratio(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    tokens = torch.randint(0, 100, (2, 4, 4, 4))

    with torch.no_grad():
        loss, metrics = task._compute_loss_from_tokens(tokens, mask_ratio=None)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    assert "mask_acc" in metrics
    assert "mask_ratio" in metrics
    assert 0 <= metrics["mask_ratio"] <= 1


def test_maskgit_task_encode_images_to_tokens_with_sliding_window(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=1e-4,
        sliding_window={"enabled": True, "roi_size": [16, 16, 16], "overlap": 0.25},
    )
    task.eval()

    x = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        tokens = task.encode_images_to_tokens(x)

    assert tokens.dim() == 4
    assert tokens.dtype == torch.long
    assert tokens.shape[0] == 1


def test_maskgit_task_validation_step_batch_idx_zero_generates_sample(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    logs: dict[str, float] = {}
    task.log = lambda name, value, **kwargs: logs.update({name: value})  # type: ignore[assignment]

    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        task.validation_step(x, 0)

    assert "val/sample_shape" in logs


def test_maskgit_task_validation_step_batch_idx_nonzero(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    logs: dict[str, float] = {}
    task.log = lambda name, value, **kwargs: logs.update({name: value})  # type: ignore[assignment]

    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        task.validation_step(x, 1)

    assert "val/sample_shape" not in logs


def test_maskgit_task_no_vqvae_checkpoint(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=None,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=1e-4,
    )
    assert task.maskgit is not None
    assert task.vqvae is not None
    assert not any(p.requires_grad for p in task.vqvae.parameters())
