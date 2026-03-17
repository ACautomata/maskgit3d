"""Tests for MaskGITTask."""

from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig

from src.maskgit3d.models.vqvae import VQVAE
from src.maskgit3d.tasks.maskgit_task import MaskGITTask


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
    result = task.configure_optimizers()
    assert isinstance(result, dict)
    assert "optimizer" in result
    assert "lr_scheduler" in result
    optimizer = result["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)


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


def test_maskgit_task_compute_masked_loss(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()
    tokens = torch.randint(0, 100, (2, 4, 4, 4))
    loss, raw_data = task._compute_masked_loss(tokens, mask_ratio=0.5)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    assert "correct" in raw_data
    assert "total" in raw_data
    assert "mask_ratio" in raw_data
    assert isinstance(raw_data["correct"], int)
    assert isinstance(raw_data["total"], int)
    assert raw_data["mask_ratio"] == 0.5


def test_maskgit_task_training_step(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    x = torch.randn(1, 1, 16, 16, 16)
    outputs = task.training_step(x, 0)
    assert isinstance(outputs, dict)
    assert "loss" in outputs
    assert "log_data" in outputs
    assert isinstance(outputs["loss"], torch.Tensor)
    assert outputs["loss"].item() >= 0


def test_maskgit_task_validation_step(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    x = torch.randn(1, 1, 16, 16, 16)
    outputs = task.validation_step(x, 0)
    assert outputs is None


def test_maskgit_task_test_step(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()
    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        outputs = task.test_step(x, 0)
    assert isinstance(outputs, dict)
    assert "generated_latent" in outputs
    assert "input_shape" in outputs
    assert "token_shape" in outputs
    assert isinstance(outputs["generated_latent"], torch.Tensor)


def test_maskgit_task_sliding_window_disabled(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=1e-4,
        sliding_window={"enabled": False},
    )
    inferer = task.maskgit._get_sliding_window_inferer()
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
    inferer = task.maskgit._get_sliding_window_inferer()
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


def test_maskgit_task_pad_to_divisible(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )

    x = torch.randn(1, 1, 33, 33, 33)
    x_padded = task.maskgit._pad_to_divisible(x)

    assert x_padded.shape[2] % 16 == 0
    assert x_padded.shape[3] % 16 == 0
    assert x_padded.shape[4] % 16 == 0
    assert x_padded.shape[1] == x.shape[1]


def test_maskgit_task_warmup_scheduler(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=1e-4,
        warmup_steps=10,
    )

    result = task.configure_optimizers()
    assert isinstance(result, dict)
    assert "lr_scheduler" in result


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


def test_maskgit_task_test_step_generates_latent(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        outputs = task.test_step(x, 0)

    assert isinstance(outputs, dict)
    assert "generated_latent" in outputs
    assert isinstance(outputs["generated_latent"], torch.Tensor)
    assert outputs["generated_latent"].shape[0] == 1  # Batch size


def test_maskgit_task_test_step_with_sliding_window_generates_latent(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=1e-4,
        sliding_window={"enabled": True, "roi_size": [16, 16, 16], "overlap": 0.25},
    )
    task.eval()

    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        outputs = task.test_step(x, 0)

    assert isinstance(outputs, dict)
    assert "generated_latent" in outputs
    assert isinstance(outputs["generated_latent"], torch.Tensor)


def test_maskgit_task_training_step_with_list_batch(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )

    x = torch.randn(1, 1, 16, 16, 16)
    batch = [x]

    outputs = task.training_step(batch, 0)  # type: ignore[arg-type]
    assert isinstance(outputs, dict)
    assert "loss" in outputs
    assert outputs["loss"].item() >= 0
    assert isinstance(outputs["log_data"]["correct"], int)
    assert isinstance(outputs["log_data"]["total"], int)


def test_maskgit_task_validation_step_with_list_batch(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    x = torch.randn(1, 1, 16, 16, 16)
    batch = [x]

    with torch.no_grad():
        outputs = task.validation_step(batch, 0)  # type: ignore[arg-type]
        assert outputs is None


def test_maskgit_task_test_step_with_list_batch(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    x = torch.randn(1, 1, 16, 16, 16)
    batch = [x]

    with torch.no_grad():
        outputs = task.test_step(batch, 0)  # type: ignore[arg-type]
        assert isinstance(outputs, dict)
        assert "generated_latent" in outputs


def test_maskgit_task_compute_masked_loss_random_mask_ratio(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    tokens = torch.randint(0, 100, (2, 4, 4, 4))

    with torch.no_grad():
        loss, raw_data = task._compute_masked_loss(tokens, mask_ratio=None)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    assert "correct" in raw_data
    assert "total" in raw_data
    assert "mask_ratio" in raw_data
    assert isinstance(raw_data["correct"], int)
    assert isinstance(raw_data["total"], int)
    assert 0 <= raw_data["mask_ratio"] <= 1


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

    task.vqvae.encode = lambda patch: (  # type: ignore[assignment]
        torch.zeros((patch.shape[0], 256, 1, 1, 1)),
        torch.tensor(0.0),
        torch.zeros((patch.shape[0], 1, 1, 1), dtype=torch.long),
    )

    x = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        tokens = task.encode_images_to_tokens(x)

    assert tokens.dim() == 4
    assert tokens.dtype == torch.long
    assert tokens.shape[0] == 1


def test_maskgit_task_validation_step_logs_metrics_and_returns_none(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    logged: dict[str, object] = {}

    def capture_log(name: str, value: object, **_: object) -> None:
        logged[name] = value

    task.log = capture_log  # type: ignore[method-assign]

    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        outputs = task.validation_step(x, 0)

    assert outputs is None
    assert "val_loss" in logged
    assert "val_mask_acc" in logged
    assert "val_mask_ratio" in logged


def test_maskgit_task_validation_step_returns_consistent_structure(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        outputs_batch_0 = task.validation_step(x, 0)
        outputs_batch_1 = task.validation_step(x, 1)

    assert outputs_batch_0 is None
    assert outputs_batch_1 is None


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


def test_maskgit_task_compute_masked_loss_ensures_at_least_one_masked(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    tokens = torch.randint(0, 100, (2, 4, 4, 4))

    with torch.no_grad():
        loss, raw_data = task._compute_masked_loss(tokens, mask_ratio=0.0)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_maskgit_task_warmup_scheduler_after_warmup(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=1e-4,
        warmup_steps=5,
    )

    result = task.configure_optimizers()
    assert isinstance(result, dict)
    optimizer = result["optimizer"]
    scheduler = result["lr_scheduler"]["scheduler"]

    for _ in range(10):
        optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr()[0] == 1e-4


class TestMaskGITTaskModelConfig:
    """Tests for MaskGITTask accepting model via DictConfig."""

    def test_accepts_model_config(self, vqvae_checkpoint):
        """MaskGITTask can be constructed with model_config DictConfig."""
        from omegaconf import DictConfig
        from maskgit3d.tasks.maskgit_task import MaskGITTask
        from maskgit3d.models.maskgit import MaskGIT

        model_cfg = DictConfig(
            {
                "_target_": "maskgit3d.models.maskgit.MaskGIT",
                "hidden_size": 128,
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 4.0,
                "dropout": 0.0,
            }
        )
        task = MaskGITTask(
            model_config=model_cfg,
            vqvae_ckpt_path=str(vqvae_checkpoint),
        )
        assert isinstance(task.maskgit, MaskGIT)
        assert task.maskgit.hidden_size == 128

    def test_model_config_overrides_scalar_params(self, vqvae_checkpoint):
        """When model_config is provided, scalar transformer params are ignored."""
        from omegaconf import DictConfig
        from maskgit3d.tasks.maskgit_task import MaskGITTask

        model_cfg = DictConfig(
            {
                "_target_": "maskgit3d.models.maskgit.MaskGIT",
                "hidden_size": 256,
                "num_layers": 4,
                "num_heads": 8,
            }
        )
        task = MaskGITTask(
            model_config=model_cfg,
            vqvae_ckpt_path=str(vqvae_checkpoint),
            hidden_size=128,  # should be ignored
            num_layers=2,
        )
        assert task.maskgit.hidden_size == 256

    def test_scalar_params_still_work(self, vqvae_checkpoint):
        """Backward compat: scalar params still work without model_config."""
        from maskgit3d.tasks.maskgit_task import MaskGITTask
        from maskgit3d.models.maskgit import MaskGIT

        task = MaskGITTask(
            vqvae_ckpt_path=str(vqvae_checkpoint),
            hidden_size=128,
            num_layers=2,
            num_heads=4,
        )
        assert isinstance(task.maskgit, MaskGIT)

    def test_model_config_saved_in_hparams(self, vqvae_checkpoint):
        """model_config should be in hparams for checkpoint loading."""
        from omegaconf import DictConfig
        from maskgit3d.tasks.maskgit_task import MaskGITTask

        model_cfg = DictConfig(
            {
                "_target_": "maskgit3d.models.maskgit.MaskGIT",
                "hidden_size": 128,
                "num_layers": 2,
                "num_heads": 4,
            }
        )
        task = MaskGITTask(model_config=model_cfg, vqvae_ckpt_path=str(vqvae_checkpoint))
        assert "model_config" in task.hparams


class TestMaskGITTaskOptimizerConfig:
    """Tests for MaskGITTask accepting optimizer via DictConfig."""

    def test_accepts_optimizer_config(self, vqvae_checkpoint):
        """MaskGITTask uses optimizer_config for optimizer."""
        opt_cfg = DictConfig(
            {
                "_target_": "torch.optim.Adam",  # different from default AdamW
                "lr": 2e-4,
            }
        )
        task = MaskGITTask(
            model_config=DictConfig(
                {
                    "_target_": "maskgit3d.models.maskgit.MaskGIT",
                    "hidden_size": 128,
                    "num_layers": 2,
                    "num_heads": 4,
                    "mlp_ratio": 4.0,
                    "dropout": 0.0,
                }
            ),
            vqvae_ckpt_path=str(vqvae_checkpoint),
            optimizer_config=opt_cfg,
        )
        result = task.configure_optimizers()
        optimizer = result["optimizer"]
        assert isinstance(optimizer, torch.optim.Adam)

    def test_optimizer_config_with_scheduler(self, vqvae_checkpoint):
        """MaskGITTask scheduler still works with optimizer_config."""
        opt_cfg = DictConfig(
            {
                "_target_": "torch.optim.AdamW",
                "lr": 1e-4,
            }
        )
        task = MaskGITTask(
            model_config=DictConfig(
                {
                    "_target_": "maskgit3d.models.maskgit.MaskGIT",
                    "hidden_size": 128,
                    "num_layers": 2,
                    "num_heads": 4,
                }
            ),
            vqvae_ckpt_path=str(vqvae_checkpoint),
            optimizer_config=opt_cfg,
            warmup_steps=100,
        )
        result = task.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_optimizer_config_fallback_to_scalar(self, vqvae_checkpoint):
        """Without optimizer_config, uses hardcoded AdamW with lr/weight_decay."""
        task = MaskGITTask(
            model_config=DictConfig(
                {
                    "_target_": "maskgit3d.models.maskgit.MaskGIT",
                    "hidden_size": 128,
                    "num_layers": 2,
                    "num_heads": 4,
                }
            ),
            vqvae_ckpt_path=str(vqvae_checkpoint),
            lr=1e-4,
            weight_decay=0.01,
        )
        result = task.configure_optimizers()
        optimizer = result["optimizer"]
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_optimizer_config_saved_in_hparams(self, vqvae_checkpoint):
        """optimizer_config should be in hparams for checkpoint loading."""
        opt_cfg = DictConfig(
            {
                "_target_": "torch.optim.AdamW",
                "lr": 1e-4,
            }
        )
        task = MaskGITTask(
            model_config=DictConfig(
                {
                    "_target_": "maskgit3d.models.maskgit.MaskGIT",
                    "hidden_size": 128,
                    "num_layers": 2,
                    "num_heads": 4,
                }
            ),
            vqvae_ckpt_path=str(vqvae_checkpoint),
            optimizer_config=opt_cfg,
        )
        assert "optimizer_config" in task.hparams
