"""Checkpoint compatibility tests for VQVAETask and MaskGITTask."""

from pathlib import Path

import pytest
import torch

from src.maskgit3d.tasks.maskgit_task import MaskGITTask
from src.maskgit3d.tasks.vqvae_task import VQVAETask


@pytest.fixture
def vqvae_ckpt_path(tmp_path: Path) -> str:
    from src.maskgit3d.models.vqvae import VQVAE

    vqvae = VQVAE()
    ckpt_path = str(tmp_path / "vqvae.ckpt")
    torch.save({"state_dict": vqvae.state_dict()}, ckpt_path)
    return ckpt_path


class TestVQVAECheckpointCompatibility:
    """Tests for VQVAETask checkpoint compatibility."""

    def test_vqvae_checkpoint_loads_with_new_signature(self, tmp_path: Path) -> None:
        """Legacy VQVAETask checkpoint loads with new dependency injection signature."""
        legacy_task = VQVAETask(
            in_channels=1,
            out_channels=1,
            latent_channels=64,
            num_embeddings=100,
            embedding_dim=64,
            lr_g=1e-4,
            lr_d=1e-4,
            use_perceptual=False,
        )

        ckpt_path = tmp_path / "vqvae_legacy.ckpt"
        torch.save(
            {
                "state_dict": legacy_task.state_dict(),
                "hyper_parameters": dict(legacy_task.hparams),
                "pytorch-lightning_version": "2.5.0",
            },
            ckpt_path,
        )

        with pytest.warns(DeprecationWarning, match="deprecated"):
            reloaded_task = VQVAETask.load_from_checkpoint(str(ckpt_path))

        assert reloaded_task.vqvae is not None
        assert reloaded_task.loss_fn is not None
        assert reloaded_task.training_steps is not None
        assert reloaded_task.automatic_optimization is False

        assert reloaded_task.hparams["in_channels"] == 1
        assert reloaded_task.hparams["out_channels"] == 1
        assert reloaded_task.hparams["latent_channels"] == 64
        assert reloaded_task.hparams["num_embeddings"] == 100
        assert reloaded_task.hparams["embedding_dim"] == 64


class TestMaskGITCheckpointCompatibility:
    """Tests for MaskGITTask checkpoint compatibility."""

    def test_maskgit_checkpoint_loads_with_new_signature(
        self, tmp_path: Path, vqvae_ckpt_path: str
    ) -> None:
        """Legacy MaskGITTask checkpoint loads with new dependency injection signature."""
        legacy_task = MaskGITTask(
            vqvae_ckpt_path=vqvae_ckpt_path,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            lr=1e-4,
            weight_decay=0.05,
            warmup_steps=500,
        )

        ckpt_path = tmp_path / "maskgit_legacy.ckpt"
        torch.save(
            {
                "state_dict": legacy_task.state_dict(),
                "hyper_parameters": dict(legacy_task.hparams),
                "pytorch-lightning_version": "2.5.0",
            },
            ckpt_path,
        )

        with pytest.warns(DeprecationWarning, match="deprecated"):
            reloaded_task = MaskGITTask.load_from_checkpoint(str(ckpt_path))

        assert reloaded_task.maskgit is not None
        assert reloaded_task.vqvae is not None
        assert reloaded_task.training_steps is not None

        assert reloaded_task.hparams["hidden_size"] == 128
        assert reloaded_task.hparams["num_layers"] == 2
        assert reloaded_task.hparams["num_heads"] == 4
        assert reloaded_task.lr == 1e-4
        assert reloaded_task.weight_decay == 0.05
        assert reloaded_task.warmup_steps == 500
