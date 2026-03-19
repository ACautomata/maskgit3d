from __future__ import annotations

from omegaconf import OmegaConf

from src.maskgit3d.losses.vq_perceptual_loss import VQPerceptualLoss
from src.maskgit3d.models.vqvae import VQVAE
from src.maskgit3d.runtime.composition import build_maskgit_task, build_vqvae_task
from src.maskgit3d.runtime.model_factory import create_vqvae_model
from src.maskgit3d.tasks.maskgit_task import MaskGITTask
from src.maskgit3d.tasks.vqvae_task import VQVAETask
from src.maskgit3d.training.maskgit_steps import MaskGITTrainingSteps
from src.maskgit3d.training.vqvae_steps import VQVAETrainingSteps


def _vqvae_model_config():
    return {
        "_target_": "maskgit3d.models.vqvae.VQVAE",
        "in_channels": 1,
        "out_channels": 1,
        "latent_channels": 32,
        "num_embeddings": 32,
        "embedding_dim": 32,
        "num_channels": [32, 64],
        "num_res_blocks": [1, 1],
        "attention_levels": [False, False],
        "commitment_cost": 0.25,
        "quantizer_type": "vq",
        "fsq_levels": [8, 8, 8, 5, 5, 5],
        "num_splits": 1,
        "dim_split": 0,
    }


def _matches_type(value: object, expected_type: type[object]) -> bool:
    return isinstance(value, expected_type) or value.__class__.__name__ == expected_type.__name__


def test_build_vqvae_task_returns_assembled_task() -> None:
    cfg = OmegaConf.create(
        {
            "task": {
                "_target_": "maskgit3d.tasks.vqvae_task.VQVAETask",
                "use_perceptual": False,
                "lambda_perceptual": 0.0,
                "gradient_clip_enabled": True,
                "gradient_clip_val": 1.0,
                "sliding_window": {"enabled": False},
            },
            "model": _vqvae_model_config(),
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4},
            "data": {"crop_size": [16, 16, 16]},
        }
    )

    task = build_vqvae_task(cfg)

    assert _matches_type(task, VQVAETask)
    assert _matches_type(task.vqvae, VQVAE)
    assert _matches_type(task.loss_fn, VQPerceptualLoss)
    assert _matches_type(task.training_steps, VQVAETrainingSteps)


def test_build_maskgit_task_returns_assembled_task(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "task": {
                "_target_": "maskgit3d.tasks.maskgit_task.MaskGITTask",
                "vqvae_ckpt_path": "/tmp/vqvae.ckpt",
                "warmup_steps": 10,
            },
            "model": {
                "_target_": "maskgit3d.models.maskgit.MaskGIT",
                "hidden_size": 64,
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
                "gamma_type": "cosine",
                "num_iterations": 4,
                "temperature": 1.0,
            },
            "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1e-4, "weight_decay": 0.01},
        }
    )

    monkeypatch.setattr(
        "src.maskgit3d.runtime.composition.load_vqvae_from_checkpoint",
        lambda _: create_vqvae_model(OmegaConf.create(_vqvae_model_config())),
    )

    task = build_maskgit_task(cfg)

    assert _matches_type(task, MaskGITTask)
    assert _matches_type(task.vqvae, VQVAE)
    assert all(not param.requires_grad for param in task.vqvae.parameters())
    assert _matches_type(task.training_steps, MaskGITTrainingSteps)
