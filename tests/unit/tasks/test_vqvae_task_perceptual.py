"""Tests for VQVAETask perceptual loss integration."""

import pytest
import torch

from src.maskgit3d.tasks.vqvae_task import VQVAETask


class TestVQVAETaskPerceptualInit:
    """Tests for perceptual loss initialization in VQVAETask."""

    def test_init_with_perceptual_enabled(self) -> None:
        """Task initializes perceptual loss when use_perceptual=True."""
        task = VQVAETask(
            in_channels=1,
            out_channels=1,
            num_embeddings=100,
            embedding_dim=16,
            use_perceptual=True,
            lambda_perceptual=0.1,
            perceptual_network="alex",
        )

        assert task.loss_fn.use_perceptual is True
        assert task.loss_fn.lambda_perceptual == 0.1
        assert task.loss_fn.perceptual_loss is not None

    def test_init_with_perceptual_disabled(self) -> None:
        """Task skips perceptual loss creation when use_perceptual=False."""
        task = VQVAETask(
            in_channels=1,
            out_channels=1,
            num_embeddings=100,
            embedding_dim=16,
            use_perceptual=False,
        )

        assert task.loss_fn.use_perceptual is False
        assert task.loss_fn.perceptual_loss is None

    def test_init_default_perceptual_is_enabled(self) -> None:
        """Perceptual loss is enabled by default."""
        task = VQVAETask(
            in_channels=1,
            out_channels=1,
            num_embeddings=100,
            embedding_dim=16,
        )

        assert task.loss_fn.use_perceptual is True
        assert task.loss_fn.lambda_perceptual == 0.1
        assert task.loss_fn.perceptual_loss is not None

    def test_init_custom_lambda_perceptual(self) -> None:
        """Custom lambda_perceptual value is stored."""
        task = VQVAETask(
            in_channels=1,
            out_channels=1,
            num_embeddings=100,
            embedding_dim=16,
            lambda_perceptual=0.5,
        )

        assert task.loss_fn.lambda_perceptual == 0.5


class TestVQVAETaskPerceptualTraining:
    """Tests for perceptual loss in training_step."""

    def _make_task_and_optimizers(
        self, use_perceptual: bool = True, lambda_perceptual: float = 0.1
    ):
        """Create task with optimizers and mocked Lightning methods."""
        task = VQVAETask(
            in_channels=1,
            out_channels=1,
            num_embeddings=100,
            embedding_dim=16,
            use_perceptual=use_perceptual,
            lambda_perceptual=lambda_perceptual,
        )

        opt_g = torch.optim.Adam(list(task.vqvae.parameters()), lr=1e-4)
        opt_d = torch.optim.Adam(task.loss_fn.discriminator.parameters(), lr=1e-4)

        task.manual_backward = lambda loss: loss.backward()  # type: ignore[assignment]

        logged: dict[str, float] = {}
        task.log = lambda name, val, **kw: logged.update(  # type: ignore[assignment]
            {name: val.item() if isinstance(val, torch.Tensor) else val}
        )

        return task, [opt_g, opt_d], logged

    def test_training_step_includes_perceptual_in_total_loss(self) -> None:
        """Training step total loss includes perceptual component."""
        task, optimizers, logged = self._make_task_and_optimizers(use_perceptual=True)

        batch = torch.randn(2, 1, 32, 32, 32)

        task.training_step(batch, batch_idx=0, optimizers=optimizers)

        assert "train/p_loss" in logged
        assert isinstance(logged["train/p_loss"], float)
        assert logged["train/p_loss"] >= 0.0

    def test_training_step_without_perceptual(self) -> None:
        """Training step works without perceptual loss (no perceptual log)."""
        task, optimizers, logged = self._make_task_and_optimizers(use_perceptual=False)

        batch = torch.randn(2, 1, 32, 32, 32)

        task.training_step(batch, batch_idx=0, optimizers=optimizers)

        assert "train/nll_loss" in logged
        assert "train/total_loss" in logged

    def test_perceptual_loss_affects_total_loss(self) -> None:
        """Total loss is higher with perceptual enabled vs disabled (same input)."""
        torch.manual_seed(42)
        batch = torch.randn(2, 1, 32, 32, 32)

        def run_step(use_perceptual: bool) -> float:
            torch.manual_seed(0)
            task, optimizers, logged = self._make_task_and_optimizers(
                use_perceptual=use_perceptual, lambda_perceptual=0.1
            )
            task.training_step(batch, batch_idx=0, optimizers=optimizers)
            return logged["train/total_loss"]

        loss_with_perceptual = run_step(use_perceptual=True)
        loss_without_perceptual = run_step(use_perceptual=False)

        assert loss_with_perceptual != loss_without_perceptual


class TestVQVAETaskAdaptiveWeight:
    """Tests for adaptive weight calculation in VQVAETask."""

    def test_init_adaptive_weight_enabled(self) -> None:
        """Task initializes with adaptive weight enabled by default."""
        task = VQVAETask(
            in_channels=1,
            out_channels=1,
            num_embeddings=100,
            embedding_dim=16,
        )

        assert task.loss_fn.use_adaptive_weight is True

    def test_init_adaptive_weight_disabled(self) -> None:
        """Task can disable adaptive weight."""
        task = VQVAETask(
            in_channels=1,
            out_channels=1,
            num_embeddings=100,
            embedding_dim=16,
            use_adaptive_weight=False,
        )

        assert task.loss_fn.use_adaptive_weight is False

    def test_init_disc_start(self) -> None:
        """Task accepts disc_start parameter for warmup."""
        task = VQVAETask(
            in_channels=1,
            out_channels=1,
            num_embeddings=100,
            embedding_dim=16,
            disc_start=50001,
        )

        assert task.loss_fn.disc_start == 50001

    def test_training_step_logs_adaptive_weight(self) -> None:
        """Training step logs the adaptive weight value."""
        task = VQVAETask(
            in_channels=1,
            out_channels=1,
            num_embeddings=100,
            embedding_dim=16,
            use_adaptive_weight=True,
        )

        opt_g = torch.optim.Adam(list(task.vqvae.parameters()), lr=1e-4)
        opt_d = torch.optim.Adam(task.loss_fn.discriminator.parameters(), lr=1e-4)

        task.manual_backward = lambda loss: loss.backward()  # type: ignore[assignment]

        logged: dict[str, float] = {}
        task.log = lambda name, val, **kw: logged.update(  # type: ignore[assignment]
            {name: val.item() if isinstance(val, torch.Tensor) else val}
        )

        batch = torch.randn(2, 1, 32, 32, 32)
        task.training_step(batch, batch_idx=0, optimizers=[opt_g, opt_d])

        assert "train/d_weight" in logged
        assert logged["train/d_weight"] >= 0.0
