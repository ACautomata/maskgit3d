"""Tests for InContextMaskGITTask."""

import warnings
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from omegaconf import DictConfig

from src.maskgit3d.models.incontext import InContextMaskGIT
from src.maskgit3d.models.vqvae import VQVAE
from src.maskgit3d.tasks.incontext_task import InContextMaskGITTask


@pytest.fixture
def vqvae_checkpoint(tmp_path: Path) -> str:
    vqvae = VQVAE()
    ckpt_path = str(tmp_path / "vqvae.ckpt")
    torch.save({"state_dict": vqvae.state_dict()}, ckpt_path)
    return ckpt_path


@pytest.fixture
def incontext_with_components(vqvae_checkpoint: str):
    """Fixture providing fully constructed InContextMaskGIT components for injection path."""
    from src.maskgit3d.runtime.optimizer_factory import TransformerOptimizerFactory
    from src.maskgit3d.training.incontext_steps import InContextTrainingSteps

    vqvae = VQVAE()
    vqvae.eval()
    vqvae.requires_grad_(False)

    incontext_model = InContextMaskGIT(
        vqvae=vqvae,
        num_modalities=4,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        gamma_type="cosine",
    )

    training_steps = InContextTrainingSteps(model=incontext_model)

    optimizer_factory = TransformerOptimizerFactory(
        lr=2e-4,
        weight_decay=0.05,
        warmup_steps=1000,
    )

    return incontext_model, vqvae, training_steps, optimizer_factory


def test_incontext_task_accepts_injected_components(incontext_with_components):
    """InContextMaskGITTask accepts injected model, vqvae, training_steps, and optimizer_factory."""
    incontext_model, vqvae, training_steps, optimizer_factory = incontext_with_components

    task = InContextMaskGITTask(
        model=incontext_model,
        vqvae=vqvae,
        training_steps=training_steps,
        optimizer_factory=optimizer_factory,
    )

    assert task.incontext_model is incontext_model
    assert task.vqvae is vqvae
    assert task.training_steps is training_steps
    assert task.optimizer_factory is optimizer_factory
    assert task.lr == 2e-4
    assert task.weight_decay == 0.05
    assert task.warmup_steps == 1000
    assert not any(p.requires_grad for p in task.vqvae.parameters())


def test_incontext_task_configure_optimizers_uses_injected_factory(incontext_with_components):
    """InContextMaskGITTask.configure_optimizers uses injected optimizer_factory when available."""
    incontext_model, vqvae, training_steps, optimizer_factory = incontext_with_components

    task = InContextMaskGITTask(
        model=incontext_model,
        vqvae=vqvae,
        training_steps=training_steps,
        optimizer_factory=optimizer_factory,
    )

    class MockTrainer:
        estimated_stepping_batches = 1000

    task._trainer = MockTrainer()

    result = task.configure_optimizers()

    assert isinstance(result, dict)
    assert "optimizer" in result
    assert "lr_scheduler" in result
    optimizer = result["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    scheduler = result["lr_scheduler"]["scheduler"]
    assert scheduler is not None


def test_incontext_task_legacy_path_emits_deprecation(vqvae_checkpoint):
    """Legacy constructor path emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        task = InContextMaskGITTask(
            vqvae_ckpt_path=vqvae_checkpoint,
            num_modalities=4,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            lr=2e-4,
        )
        assert task.incontext_model is not None
        assert task.vqvae is not None

    deprecation_warnings = [
        w for w in caught_warnings if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 1
    assert "legacy" in str(deprecation_warnings[0].message).lower()


def test_incontext_task_init(vqvae_checkpoint: str):
    task = InContextMaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint,
        num_modalities=4,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=2e-4,
    )
    assert task.incontext_model is not None
    assert task.vqvae is not None
    assert not any(p.requires_grad for p in task.vqvae.parameters())


def test_incontext_task_vqvae_is_frozen(vqvae_checkpoint: str):
    """VQVAE should be frozen after initialization."""
    task = InContextMaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint,
        num_modalities=4,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=2e-4,
    )
    assert task.vqvae is not None
    assert not task.vqvae.training
    assert not any(p.requires_grad for p in task.vqvae.parameters())


def test_incontext_task_configure_optimizers(vqvae_checkpoint: str):
    task = InContextMaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint,
        num_modalities=4,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=2e-4,
    )

    class MockTrainer:
        estimated_stepping_batches = 1000

    task._trainer = MockTrainer()

    result = task.configure_optimizers()
    assert isinstance(result, dict)
    assert "optimizer" in result
    assert "lr_scheduler" in result
    optimizer = result["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)


def test_incontext_task_forward(vqvae_checkpoint: str):
    task = InContextMaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint,
        num_modalities=4,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=2e-4,
    )
    task.eval()

    context_images = [torch.randn(1, 1, 16, 16, 16)]
    context_modality_ids = [0]
    target_image = torch.randn(1, 1, 16, 16, 16)
    target_modality_id = 1

    with torch.no_grad():
        output = task(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
        )
    assert output.shape == target_image.shape


def test_incontext_task_training_step_returns_loss(vqvae_checkpoint: str):
    """training_step should return a dict with 'loss' tensor."""
    task = InContextMaskGITTask(
        vqvae_ckpt_path=vqvae_checkpoint,
        num_modalities=4,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        lr=2e-4,
    )

    batch = {
        "context_images": [torch.randn(1, 1, 16, 16, 16)],
        "context_modality_ids": [0],
        "target_image": torch.randn(1, 1, 16, 16, 16),
        "target_modality_id": 1,
    }

    output: dict[str, Any] = cast(dict[str, Any], task.training_step(batch, 0))
    loss_value = output.get("loss")

    assert isinstance(output, dict)
    assert set(output) == {"loss"}
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.item() >= 0


def test_incontext_task_training_step_with_injected_components(incontext_with_components):
    """training_step works with injected components."""
    incontext_model, vqvae, training_steps, optimizer_factory = incontext_with_components

    task = InContextMaskGITTask(
        model=incontext_model,
        vqvae=vqvae,
        training_steps=training_steps,
        optimizer_factory=optimizer_factory,
    )

    batch = {
        "context_images": [torch.randn(1, 1, 16, 16, 16)],
        "context_modality_ids": [0],
        "target_image": torch.randn(1, 1, 16, 16, 16),
        "target_modality_id": 1,
    }

    output: dict[str, Any] = cast(dict[str, Any], task.training_step(batch, 0))
    loss_value = output.get("loss")

    assert isinstance(output, dict)
    assert "loss" in output
    assert isinstance(loss_value, torch.Tensor)


class TestInContextMaskGITTaskModelConfig:
    """Tests for InContextMaskGITTask accepting model via DictConfig."""

    def test_accepts_model_config(self, vqvae_checkpoint):
        """InContextMaskGITTask can be constructed with model_config DictConfig."""
        model_cfg = DictConfig(
            {
                "_target_": "maskgit3d.models.incontext.InContextMaskGIT",
                "num_modalities": 4,
                "hidden_size": 128,
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 4.0,
                "dropout": 0.0,
            }
        )
        task = InContextMaskGITTask(
            model_config=model_cfg,
            vqvae_ckpt_path=str(vqvae_checkpoint),
        )
        assert task.incontext_model is not None
        assert hasattr(task.incontext_model, "tokenizer")
        assert hasattr(task.incontext_model, "transformer")

    def test_model_config_saved_in_hparams(self, vqvae_checkpoint):
        """model_config should be in hparams for checkpoint loading."""
        model_cfg = DictConfig(
            {
                "_target_": "maskgit3d.models.incontext.InContextMaskGIT",
                "num_modalities": 4,
                "hidden_size": 128,
                "num_layers": 2,
                "num_heads": 4,
            }
        )
        task = InContextMaskGITTask(model_config=model_cfg, vqvae_ckpt_path=str(vqvae_checkpoint))
        assert "model_config" in task.hparams


class TestInContextMaskGITTaskOptimizerConfig:
    """Tests for InContextMaskGITTask accepting optimizer via DictConfig."""

    def test_accepts_optimizer_config(self, vqvae_checkpoint):
        """InContextMaskGITTask uses optimizer_config for optimizer."""
        opt_cfg = DictConfig(
            {
                "_target_": "torch.optim.Adam",
                "lr": 2e-4,
            }
        )
        task = InContextMaskGITTask(
            model_config=DictConfig(
                {
                    "_target_": "maskgit3d.models.incontext.InContextMaskGIT",
                    "num_modalities": 4,
                    "hidden_size": 128,
                    "num_layers": 2,
                    "num_heads": 4,
                }
            ),
            vqvae_ckpt_path=str(vqvae_checkpoint),
            optimizer_config=opt_cfg,
        )

        class MockTrainer:
            estimated_stepping_batches = 1000

        task._trainer = MockTrainer()

        result = task.configure_optimizers()
        optimizer = result["optimizer"]
        assert isinstance(optimizer, torch.optim.Adam)

    def test_optimizer_config_saved_in_hparams(self, vqvae_checkpoint):
        """optimizer_config should be in hparams for checkpoint loading."""
        opt_cfg = DictConfig(
            {
                "_target_": "torch.optim.AdamW",
                "lr": 2e-4,
            }
        )
        task = InContextMaskGITTask(
            model_config=DictConfig(
                {
                    "_target_": "maskgit3d.models.incontext.InContextMaskGIT",
                    "num_modalities": 4,
                    "hidden_size": 128,
                    "num_layers": 2,
                    "num_heads": 4,
                }
            ),
            vqvae_ckpt_path=str(vqvae_checkpoint),
            optimizer_config=opt_cfg,
        )
        assert "optimizer_config" in task.hparams
