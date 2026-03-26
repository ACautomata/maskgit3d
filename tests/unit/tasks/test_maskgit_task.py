"""Tests for MaskGITTask."""

import warnings
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from omegaconf import DictConfig

from maskgit3d.models.vqvae import VQVAE
from maskgit3d.tasks.maskgit_task import MaskGITTask


@pytest.fixture
def vqvae_checkpoint(tmp_path: Path) -> str:
    vqvae = VQVAE()
    ckpt_path = str(tmp_path / "vqvae.ckpt")
    torch.save({"state_dict": vqvae.state_dict()}, ckpt_path)
    return ckpt_path


@pytest.fixture
def maskgit_with_components(vqvae_checkpoint: str):
    """Fixture providing fully constructed MaskGIT components for injection path."""
    from maskgit3d.models.maskgit import MaskGIT
    from maskgit3d.runtime.optimizer_factory import TransformerOptimizerFactory
    from maskgit3d.training import MaskGITTrainingSteps

    vqvae = VQVAE()
    vqvae.eval()
    vqvae.requires_grad_(False)

    maskgit = MaskGIT(
        vqvae=vqvae,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        gamma_type="cosine",
    )

    training_steps = MaskGITTrainingSteps(maskgit=maskgit)

    optimizer_factory = TransformerOptimizerFactory(
        lr=1e-4,
        weight_decay=0.05,
        warmup_steps=1000,
    )

    return maskgit, vqvae, training_steps, optimizer_factory


def test_maskgit_task_accepts_injected_components(maskgit_with_components):
    """MaskGITTask accepts injected model, vqvae, training_steps, and optimizer_factory."""
    maskgit, vqvae, training_steps, optimizer_factory = maskgit_with_components

    task = MaskGITTask(
        model=maskgit,
        vqvae=vqvae,
        training_steps=training_steps,
        optimizer_factory=optimizer_factory,
    )

    assert task.maskgit is maskgit
    assert task.vqvae is vqvae
    assert task.training_steps is training_steps
    assert task.optimizer_factory is optimizer_factory
    assert task.lr == 1e-4
    assert task.weight_decay == 0.05
    assert task.warmup_steps == 1000
    assert not any(p.requires_grad for p in task.vqvae.parameters())


def test_maskgit_task_configure_optimizers_uses_injected_factory(maskgit_with_components):
    """MaskGITTask.configure_optimizers uses injected optimizer_factory when available."""
    maskgit, vqvae, training_steps, optimizer_factory = maskgit_with_components

    task = MaskGITTask(
        model=maskgit,
        vqvae=vqvae,
        training_steps=training_steps,
        optimizer_factory=optimizer_factory,
    )

    # Mock the trainer to provide estimated_stepping_batches
    class MockTrainer:
        estimated_stepping_batches = 1000

    task._trainer = MockTrainer()  # type: ignore[attr-defined]

    result = task.configure_optimizers()

    assert isinstance(result, dict)
    assert "optimizer" in result
    assert "lr_scheduler" in result
    optimizer = result["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    scheduler = result["lr_scheduler"]["scheduler"]
    assert scheduler is not None


def test_maskgit_task_legacy_path_emits_deprecation(vqvae_checkpoint):
    """Legacy constructor path emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        task = MaskGITTask(
            vqvae_ckpt_path=vqvae_checkpoint,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            lr=1e-4,
        )
        assert task.maskgit is not None
        assert task.vqvae is not None

    deprecation_warnings = [
        w for w in caught_warnings if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 1
    assert "legacy" in str(deprecation_warnings[0].message).lower()


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

    # Mock the trainer to provide estimated_stepping_batches
    class MockTrainer:
        estimated_stepping_batches = 1000

    task._trainer = MockTrainer()  # type: ignore[attr-defined]

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
    output: dict[str, Any] = cast(dict[str, Any], task.training_step(x, 0))
    loss_value = output.get("loss")

    assert isinstance(output, dict)
    assert set(output) == {"loss"}
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.item() >= 0


def test_maskgit_task_validation_step(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()
    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        outputs = task.validation_step(x, 0)
    assert isinstance(outputs, dict)
    assert "generated_images" in outputs
    assert "x_real" in outputs
    assert "masked_logits" in outputs
    assert "masked_targets" in outputs
    assert set(outputs) == {"x_real", "generated_images", "masked_logits", "masked_targets"}
    assert torch.equal(outputs["x_real"], x.cpu())


def test_maskgit_task_test_step(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()
    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        outputs = task.test_step(x, 0)
    assert isinstance(outputs, dict)
    assert "generated_images" in outputs
    assert "x_real" in outputs
    assert "masked_logits" in outputs
    assert "masked_targets" in outputs
    assert set(outputs) == {"x_real", "generated_images", "masked_logits", "masked_targets"}
    assert isinstance(outputs["generated_images"], torch.Tensor)


def test_maskgit_task_predict_step(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()
    x = torch.randn(1, 1, 16, 16, 16)

    with torch.no_grad():
        outputs = task.predict_step(x, 0)

    assert isinstance(outputs, dict)
    assert "x_real" in outputs
    assert "generated_images" in outputs
    assert set(outputs) == {"x_real", "generated_images"}
    assert torch.equal(outputs["x_real"], x.cpu())


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

    assert x_padded.shape[2] % task.maskgit._downsampling_factor == 0
    assert x_padded.shape[3] % task.maskgit._downsampling_factor == 0
    assert x_padded.shape[4] % task.maskgit._downsampling_factor == 0
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

    # Mock the trainer to provide estimated_stepping_batches
    class MockTrainer:
        estimated_stepping_batches = 1000

    task._trainer = MockTrainer()  # type: ignore[attr-defined]

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

    mock_logger = MockLogger()
    task.log = mock_logger  # type: ignore[assignment]

    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        task.validation_step(x, 0)

    assert mock_logger.logs == {}


def test_maskgit_task_test_step_generates_images(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        outputs = task.test_step(x, 0)

    assert isinstance(outputs, dict)
    assert "generated_images" in outputs
    assert isinstance(outputs["generated_images"], torch.Tensor)
    assert outputs["generated_images"].shape[0] == 1


def test_maskgit_task_test_step_with_sliding_window_generates_images(vqvae_checkpoint: str):
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
    assert "generated_images" in outputs
    assert isinstance(outputs["generated_images"], torch.Tensor)


def test_maskgit_task_training_step_with_list_batch(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )

    x = torch.randn(1, 1, 16, 16, 16)
    batch = [x]

    output: dict[str, Any] = cast(
        dict[str, Any],
        task.training_step(batch, 0),  # type: ignore[arg-type]
    )
    loss_value = output.get("loss")

    assert isinstance(output, dict)
    assert set(output) == {"loss"}
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.item() >= 0


def test_maskgit_task_validation_step_with_list_batch(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    x = torch.randn(1, 1, 16, 16, 16)
    batch = [x]

    with torch.no_grad():
        outputs = task.validation_step(batch, 0)  # type: ignore[arg-type]
        assert isinstance(outputs, dict)
        assert "generated_images" in outputs
        assert "x_real" in outputs
        assert "masked_logits" in outputs


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
        assert "generated_images" in outputs
        assert "x_real" in outputs
        assert "masked_logits" in outputs


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
        torch.zeros((patch.shape[0], 256, 1, 1, 1)),
    )

    x = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        tokens = task.encode_images_to_tokens(x)

    assert tokens.dim() == 4
    assert tokens.dtype == torch.long
    assert tokens.shape[0] == 1


def test_maskgit_task_validation_step_does_not_log_metrics_directly(vqvae_checkpoint: str):
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

    assert isinstance(outputs, dict)
    assert "generated_images" in outputs
    assert logged == {}


def test_maskgit_task_validation_step_returns_consistent_structure(vqvae_checkpoint: str):
    task = MaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint, hidden_size=128, num_layers=2, num_heads=4, lr=1e-4
    )
    task.eval()

    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        outputs_batch_0 = task.validation_step(x, 0)
        outputs_batch_1 = task.validation_step(x, 1)

    assert isinstance(outputs_batch_0, dict)
    assert isinstance(outputs_batch_1, dict)
    assert "generated_images" in outputs_batch_0
    assert "generated_images" in outputs_batch_1
    assert "x_real" in outputs_batch_0
    assert "x_real" in outputs_batch_1
    assert "masked_logits" in outputs_batch_0
    assert "masked_logits" in outputs_batch_1


def test_maskgit_task_no_vqvae_checkpoint():
    # Should raise ValueError when vqvae_ckpt_path is None in builder
    from omegaconf import OmegaConf

    from maskgit3d.runtime.composition import build_maskgit_task

    cfg = OmegaConf.create(
        {
            "task": {
                "_target_": "maskgit3d.tasks.maskgit_task.MaskGITTask",
                "vqvae_ckpt_path": None,
                "hidden_size": 128,
                "num_layers": 2,
                "num_heads": 4,
            },
            "model": {"_target_": "maskgit3d.models.maskgit.MaskGIT"},
        }
    )

    with pytest.raises(ValueError, match="vqvae_ckpt_path is required"):
        build_maskgit_task(cfg)


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

    # Mock the trainer to provide estimated_stepping_batches
    class MockTrainer:
        estimated_stepping_batches = 100

    task._trainer = MockTrainer()  # type: ignore[attr-defined]

    result = task.configure_optimizers()
    assert isinstance(result, dict)
    optimizer = result["optimizer"]
    scheduler = result["lr_scheduler"]["scheduler"]

    for _ in range(10):
        optimizer.step()
        scheduler.step()

    # After warmup + 5 more steps with cosine decay, LR is slightly below max
    # due to cosine decay factor (total_steps=100, progress after step 10 is ~5%)
    assert scheduler.get_last_lr()[0] == pytest.approx(9.93e-5, rel=0.01)


class TestMaskGITTaskModelConfig:
    """Tests for MaskGITTask accepting model via DictConfig."""

    def test_accepts_model_config(self, vqvae_checkpoint):
        """MaskGITTask can be constructed with model_config DictConfig."""
        from omegaconf import DictConfig

        from maskgit3d.models.maskgit import MaskGIT
        from maskgit3d.tasks.maskgit_task import MaskGITTask

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
        from maskgit3d.models.maskgit import MaskGIT
        from maskgit3d.tasks.maskgit_task import MaskGITTask

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

        # Mock the trainer to provide estimated_stepping_batches
        class MockTrainer:
            estimated_stepping_batches = 1000

        task._trainer = MockTrainer()  # type: ignore[attr-defined]

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

        # Mock the trainer to provide estimated_stepping_batches
        class MockTrainer:
            estimated_stepping_batches = 1000

        task._trainer = MockTrainer()  # type: ignore[attr-defined]

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

        # Mock the trainer to provide estimated_stepping_batches
        class MockTrainer:
            estimated_stepping_batches = 1000

        task._trainer = MockTrainer()  # type: ignore[attr-defined]

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
