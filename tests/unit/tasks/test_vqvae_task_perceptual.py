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

        assert task.use_perceptual is True
        assert task.lambda_perceptual == 0.1
        assert hasattr(task, "perceptual_loss")
        assert task.perceptual_loss is not None

    def test_init_with_perceptual_disabled(self) -> None:
        """Task skips perceptual loss creation when use_perceptual=False."""
        task = VQVAETask(
            in_channels=1,
            out_channels=1,
            num_embeddings=100,
            embedding_dim=16,
            use_perceptual=False,
        )

        assert task.use_perceptual is False
        # perceptual_loss should not exist or be None
        assert not hasattr(task, "perceptual_loss") or task.perceptual_loss is None

    def test_init_default_perceptual_is_enabled(self) -> None:
        """Perceptual loss is enabled by default."""
        task = VQVAETask(
            in_channels=1,
            out_channels=1,
            num_embeddings=100,
            embedding_dim=16,
        )

        assert task.use_perceptual is True
        assert task.lambda_perceptual == 0.1
        assert task.perceptual_loss is not None

    def test_init_custom_lambda_perceptual(self) -> None:
        """Custom lambda_perceptual value is stored."""
        task = VQVAETask(
            in_channels=1,
            out_channels=1,
            num_embeddings=100,
            embedding_dim=16,
            lambda_perceptual=0.5,
        )

        assert task.lambda_perceptual == 0.5


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
        opt_d = torch.optim.Adam(task.discriminator.parameters(), lr=1e-4)

        # Mock manual_backward to just call .backward()
        task.manual_backward = lambda loss: loss.backward()  # type: ignore[assignment]

        # Mock self.log to capture logged values
        logged: dict[str, float] = {}
        task.log = lambda name, val, **kw: logged.update(  # type: ignore[assignment]
            {name: val.item() if isinstance(val, torch.Tensor) else val}
        )

        return task, [opt_g, opt_d], logged

    def test_training_step_includes_perceptual_in_total_loss(self) -> None:
        """Training step total loss includes perceptual component."""
        task, optimizers, logged = self._make_task_and_optimizers(use_perceptual=True)

        # Use 32x32x32 spatial size to avoid discriminator InstanceNorm issue
        batch = torch.randn(2, 1, 32, 32, 32)

        loss_g = task.training_step(batch, batch_idx=0, optimizers=optimizers)

        # Perceptual loss should be logged
        assert "train/loss_perceptual" in logged
        assert isinstance(logged["train/loss_perceptual"], float)
        assert logged["train/loss_perceptual"] >= 0.0

    def test_training_step_without_perceptual(self) -> None:
        """Training step works without perceptual loss (no perceptual log)."""
        task, optimizers, logged = self._make_task_and_optimizers(use_perceptual=False)

        batch = torch.randn(2, 1, 32, 32, 32)

        loss_g = task.training_step(batch, batch_idx=0, optimizers=optimizers)

        # Perceptual loss should NOT be logged
        assert "train/loss_perceptual" not in logged
        # Other losses should still be logged
        assert "train/loss_l1" in logged
        assert "train/loss_g" in logged

    def test_perceptual_loss_affects_total_loss(self) -> None:
        """Total loss is higher with perceptual enabled vs disabled (same input)."""
        torch.manual_seed(42)
        batch = torch.randn(2, 1, 32, 32, 32)

        def run_step(use_perceptual: bool) -> float:
            torch.manual_seed(0)  # Same init for fair comparison
            task, optimizers, logged = self._make_task_and_optimizers(
                use_perceptual=use_perceptual, lambda_perceptual=0.1
            )
            task.training_step(batch.clone(), batch_idx=0, optimizers=optimizers)
            return logged["train/loss_g"]

        loss_with = run_step(use_perceptual=True)
        loss_without = run_step(use_perceptual=False)

        # They should differ since perceptual loss adds a non-zero component
        assert loss_with != loss_without
