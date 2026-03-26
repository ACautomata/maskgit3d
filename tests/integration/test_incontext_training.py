"""Integration tests for InContext training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.maskgit3d.models.incontext import InContextMaskGIT
from src.maskgit3d.models.incontext.types import InContextSample
from src.maskgit3d.models.vqvae import VQVAE
from src.maskgit3d.runtime.checkpoints import VQVAECheckpointLoader
from src.maskgit3d.runtime.composition import build_incontext_task
from src.maskgit3d.runtime.optimizer_factory import TransformerOptimizerFactory
from src.maskgit3d.tasks.incontext_task import InContextMaskGITTask
from src.maskgit3d.training.incontext_steps import InContextTrainingSteps


@pytest.fixture
def small_vqvae() -> VQVAE:
    return VQVAE(
        in_channels=1,
        out_channels=1,
        latent_channels=32,
        num_embeddings=64,
        embedding_dim=32,
        num_channels=(32, 64),
        num_res_blocks=(1, 1),
        attention_levels=(False, False),
    )


@pytest.fixture
def small_incontext_model(small_vqvae: VQVAE) -> InContextMaskGIT:
    return InContextMaskGIT(
        vqvae=small_vqvae,
        num_modalities=4,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        gamma_type="cosine",
    )


@pytest.fixture
def vqvae_checkpoint(tmp_path: Path, small_vqvae: VQVAE) -> str:
    ckpt_path = str(tmp_path / "vqvae.ckpt")
    torch.save({"state_dict": small_vqvae.state_dict()}, ckpt_path)
    return ckpt_path


class TestInContextMaskGITTrainingStep:
    """Tests for InContextMaskGIT compute_loss() end-to-end."""

    def test_compute_loss_returns_scalar_differentiable_finite(
        self, small_incontext_model: InContextMaskGIT
    ) -> None:
        """Verify compute_loss returns a scalar, differentiable, finite loss."""
        batch_size = 2
        spatial_size = 16

        context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
        context_modality_ids = [0]
        target_image = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
        target_modality_id = 1

        loss, metrics = small_incontext_model.compute_loss(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
        )

        assert loss.ndim == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)
        assert "mask_acc" in metrics
        assert "mask_ratio" in metrics

    def test_compute_loss_with_multiple_context_modalities(
        self, small_incontext_model: InContextMaskGIT
    ) -> None:
        """Test compute_loss with multiple context modalities."""
        batch_size = 1
        spatial_size = 16

        context_images = [
            torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size),
            torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size),
        ]
        context_modality_ids = [0, 1]
        target_image = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
        target_modality_id = 2

        loss, metrics = small_incontext_model.compute_loss(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
        )

        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert isinstance(metrics, dict)

    def test_compute_loss_backward_pass(self, small_incontext_model: InContextMaskGIT) -> None:
        """Verify backward pass works without errors."""
        batch_size = 1
        spatial_size = 16

        context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
        context_modality_ids = [0]
        target_image = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
        target_modality_id = 1

        loss, _ = small_incontext_model.compute_loss(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
        )

        loss.backward()


class TestInContextMaskGITGeneration:
    """Tests for InContextMaskGIT generate() end-to-end."""

    def test_generate_output_shape_matches_target(
        self, small_incontext_model: InContextMaskGIT
    ) -> None:
        """Verify generated output shape matches expected target shape."""
        batch_size = 2
        spatial_size = 16

        context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
        context_modality_ids = [0]
        target_modality_id = 1
        target_shape = (batch_size, spatial_size, spatial_size, spatial_size)

        output = small_incontext_model.generate(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
            target_shape=target_shape,
        )

        assert output.shape == (batch_size, 1, spatial_size, spatial_size, spatial_size)

    def test_generate_with_multiple_contexts(self, small_incontext_model: InContextMaskGIT) -> None:
        """Test generation with multiple context modalities."""
        batch_size = 1
        spatial_size = 16

        context_images = [
            torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size),
            torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size),
        ]
        context_modality_ids = [0, 1]
        target_modality_id = 2
        target_shape = (batch_size, spatial_size, spatial_size, spatial_size)

        output = small_incontext_model.generate(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
            target_shape=target_shape,
        )

        assert output.shape == (batch_size, 1, spatial_size, spatial_size, spatial_size)

    def test_generate_different_iterations(self, small_incontext_model: InContextMaskGIT) -> None:
        """Test generation with different numbers of iterations."""
        batch_size = 1
        spatial_size = 16

        context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
        context_modality_ids = [0]
        target_modality_id = 1
        target_shape = (batch_size, spatial_size, spatial_size, spatial_size)

        output_single = small_incontext_model.generate(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
            target_shape=target_shape,
            num_iterations=1,
        )

        assert output_single.shape == (batch_size, 1, spatial_size, spatial_size, spatial_size)


class TestInContextTaskLightningIntegration:
    """Tests for InContextMaskGITTask Lightning integration."""

    @pytest.fixture
    def task_with_injected_components(
        self, small_vqvae: VQVAE, small_incontext_model: InContextMaskGIT
    ) -> InContextMaskGITTask:
        """Create task with injected components."""
        small_vqvae.eval()
        small_vqvae.requires_grad_(False)

        training_steps = InContextTrainingSteps(model=small_incontext_model)

        optimizer_factory = TransformerOptimizerFactory(
            lr=2e-4,
            weight_decay=0.05,
            warmup_steps=100,
        )

        return InContextMaskGITTask(
            model=small_incontext_model,
            vqvae=small_vqvae,
            training_steps=training_steps,
            optimizer_factory=optimizer_factory,
        )

    def test_training_step_returns_loss(
        self, task_with_injected_components: InContextMaskGITTask
    ) -> None:
        """Verify training_step returns a dict with loss tensor."""
        batch = {
            "context_images": [torch.randn(1, 1, 16, 16, 16)],
            "context_modality_ids": [0],
            "target_image": torch.randn(1, 1, 16, 16, 16),
            "target_modality_id": 1,
        }

        output: dict[str, Any] = cast(
            dict[str, Any], task_with_injected_components.training_step(batch, 0)
        )

        assert isinstance(output, dict)
        assert "loss" in output
        assert isinstance(output["loss"], torch.Tensor)
        assert output["loss"].ndim == 0
        assert torch.isfinite(output["loss"])

    def test_training_step_loss_is_differentiable(
        self, task_with_injected_components: InContextMaskGITTask
    ) -> None:
        """Verify training_step loss can be backpropagated."""
        batch = {
            "context_images": [torch.randn(1, 1, 16, 16, 16)],
            "context_modality_ids": [0],
            "target_image": torch.randn(1, 1, 16, 16, 16),
            "target_modality_id": 1,
        }

        output = task_with_injected_components.training_step(batch, 0)
        loss = output["loss"]

        loss.backward()

    def test_validation_step_returns_correct_keys(
        self, task_with_injected_components: InContextMaskGITTask
    ) -> None:
        """Verify validation_step returns expected output keys."""
        task_with_injected_components.eval()

        batch = {
            "context_images": [torch.randn(1, 1, 16, 16, 16)],
            "context_modality_ids": [0],
            "target_image": torch.randn(1, 1, 16, 16, 16),
            "target_modality_id": 1,
        }

        with torch.no_grad():
            output = task_with_injected_components.validation_step(batch, 0)

        assert "context_images" in output
        assert "target_image" in output
        assert "generated_image" in output
        assert "loss" in output

    def test_predict_step_returns_generated_image(
        self, task_with_injected_components: InContextMaskGITTask
    ) -> None:
        """Verify predict_step returns generated image."""
        task_with_injected_components.eval()

        batch = {
            "context_images": [torch.randn(1, 1, 16, 16, 16)],
            "context_modality_ids": [0],
            "target_image": torch.randn(1, 1, 16, 16, 16),
            "target_modality_id": 1,
        }

        with torch.no_grad():
            output = task_with_injected_components.predict_step(batch, 0)

        assert "generated_image" in output
        assert output["generated_image"].shape == (1, 1, 16, 16, 16)


class TestAny2OneBatching:
    """Tests for any2one batching pipeline with variable context per sample."""

    @pytest.fixture
    def mock_vqvae(self) -> VQVAE:
        """Create a mock VQVAE for testing."""
        vqvae = Mock()
        vqvae.quantizer = Mock()
        vqvae.quantizer.num_embeddings = 8192
        vqvae.quantizer.embedding_dim = 32
        vqvae.encoder = Mock()
        vqvae.encoder.encoder = Mock()
        vqvae.encoder.encoder.num_channels = [64, 128, 256]

        def mock_encode(x: torch.Tensor) -> tuple:
            B, C, D, H, W = x.shape
            f = 4  # downsampling factor matching num_channels [64, 128, 256]
            latent_D, latent_H, latent_W = D // f, H // f, W // f
            z_e = torch.randn(B, 32, latent_D, latent_H, latent_W)
            indices = torch.randint(0, 8192, (B, latent_D, latent_H, latent_W))
            return (None, None, indices, z_e)

        def mock_quantizer(z_e: torch.Tensor) -> tuple:
            B, C, D, H, W = z_e.shape
            indices = torch.randint(0, 8192, (B, D, H, W))
            return (z_e, None, indices)

        vqvae.encode = Mock(side_effect=mock_encode)
        vqvae.quantizer.decode_from_indices = Mock(
            side_effect=lambda x: torch.randn(
                x.shape[0], 1, x.shape[1] * 4, x.shape[2] * 4, x.shape[3] * 4
            )
        )
        vqvae.quantizer.side_effect = mock_quantizer
        return vqvae

    @pytest.fixture
    def incontext_model(self, mock_vqvae: VQVAE) -> InContextMaskGIT:
        """Create an InContextMaskGIT model with mocked VQVAE."""
        return InContextMaskGIT(
            vqvae=mock_vqvae,
            num_modalities=4,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
        )

    @pytest.fixture
    def samples_with_variable_context(self) -> list[InContextSample]:
        """Create samples with different numbers of context modalities."""
        sample0 = InContextSample(
            context_images=[torch.randn(1, 32, 32, 32)],
            context_modality_ids=[0],
            target_image=torch.randn(1, 32, 32, 32),
            target_modality_id=1,
            mask_ratio=0.5,
        )
        sample1 = InContextSample(
            context_images=[
                torch.randn(1, 32, 32, 32),
                torch.randn(1, 32, 32, 32),
            ],
            context_modality_ids=[0, 2],
            target_image=torch.randn(1, 32, 32, 32),
            target_modality_id=1,
            mask_ratio=0.5,
        )
        return [sample0, sample1]

    def test_prepare_batch_variable_context(
        self,
        incontext_model: InContextMaskGIT,
        samples_with_variable_context: list[InContextSample],
    ) -> None:
        """Test prepare_batch with samples having different context counts."""
        batch = incontext_model.prepare_batch(samples_with_variable_context)

        # Verify output shapes
        assert batch.sequences.shape[0] == 2  # batch size = 2
        assert batch.attention_mask.shape[0] == 2
        assert batch.target_mask.shape[0] == 2
        assert batch.labels.shape[0] == 2

        # All should be padded to max length
        max_len = batch.sequences.shape[1]
        assert batch.attention_mask.shape[1] == max_len
        assert batch.target_mask.shape[1] == max_len
        assert batch.labels.shape[1] == max_len

        # Verify attention_mask is valid (0 or 1 values)
        assert batch.attention_mask.min() >= 0.0
        assert batch.attention_mask.max() <= 1.0

        # Verify each sample has some attention (at least CLS + some tokens)
        for i in range(2):
            assert batch.attention_mask[i].sum() > 0

        # Verify target_mask is correct for each sample
        for i in range(2):
            # target_mask should indicate target positions
            assert batch.target_mask[i].dtype == torch.bool
            # At least some target positions should exist
            assert batch.target_mask[i].any()

    def test_compute_loss_from_prepared_variable_context(
        self,
        incontext_model: InContextMaskGIT,
        samples_with_variable_context: list[InContextSample],
    ) -> None:
        """Test compute_loss_from_prepared with variable context batch."""
        batch = incontext_model.prepare_batch(samples_with_variable_context)

        loss, metrics = incontext_model.compute_loss_from_prepared(batch)

        # Verify loss is scalar and differentiable
        assert loss.ndim == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)

        # Verify loss.backward() works
        loss.backward()

        # Check metrics
        assert "mask_acc" in metrics
        assert "mask_ratio" in metrics

    def test_training_step_any2one(
        self,
        incontext_model: InContextMaskGIT,
        samples_with_variable_context: list[InContextSample],
    ) -> None:
        """Test training_step_any2one with variable context samples."""
        training_steps = InContextTrainingSteps(model=incontext_model)

        result = training_steps.training_step_any2one(samples_with_variable_context)

        # Verify loss is returned
        assert "loss" in result
        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].ndim == 0
        assert torch.isfinite(result["loss"])


class TestInContextAny2OneIntegration:
    """Integration tests for any2one batching pipeline."""

    @pytest.fixture
    def mock_vqvae(self) -> VQVAE:
        """Create a mock VQVAE for testing."""
        vqvae = Mock()
        vqvae.quantizer = Mock()
        vqvae.quantizer.num_embeddings = 8192
        vqvae.quantizer.embedding_dim = 32
        vqvae.encoder = Mock()
        vqvae.encoder.encoder = Mock()
        vqvae.encoder.encoder.num_channels = [64, 128, 256]

        def mock_encode(x: torch.Tensor) -> tuple:
            B, C, D, H, W = x.shape
            f = 4
            latent_D, latent_H, latent_W = D // f, H // f, W // f
            z_e = torch.randn(B, 32, latent_D, latent_H, latent_W)
            indices = torch.randint(0, 8192, (B, latent_D, latent_H, latent_W))
            return (None, None, indices, z_e)

        def mock_quantizer(z_e: torch.Tensor) -> tuple:
            B, C, D, H, W = z_e.shape
            indices = torch.randint(0, 8192, (B, D, H, W))
            return (z_e, None, indices)

        vqvae.encode = Mock(side_effect=mock_encode)
        vqvae.quantizer.decode_from_indices = Mock(
            side_effect=lambda x: torch.randn(
                x.shape[0], 1, x.shape[1] * 4, x.shape[2] * 4, x.shape[3] * 4
            )
        )
        vqvae.quantizer.side_effect = mock_quantizer
        return vqvae

    @pytest.fixture
    def incontext_model(self, mock_vqvae: VQVAE) -> InContextMaskGIT:
        """Create an InContextMaskGIT model with mocked VQVAE."""
        return InContextMaskGIT(
            vqvae=mock_vqvae,
            num_modalities=4,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
        )

    def test_any2one_single_context(self, incontext_model: InContextMaskGIT) -> None:
        """Test any2one with single context modality."""
        samples = [
            InContextSample(
                context_images=[torch.randn(1, 32, 32, 32)],
                context_modality_ids=[0],
                target_image=torch.randn(1, 32, 32, 32),
                target_modality_id=1,
                mask_ratio=0.5,
            ),
            InContextSample(
                context_images=[torch.randn(1, 32, 32, 32)],
                context_modality_ids=[1],
                target_image=torch.randn(1, 32, 32, 32),
                target_modality_id=2,
                mask_ratio=0.5,
            ),
        ]

        training_steps = InContextTrainingSteps(model=incontext_model)
        result = training_steps.training_step_any2one(samples)

        assert "loss" in result
        assert result["loss"].ndim == 0
        assert torch.isfinite(result["loss"])

    def test_any2one_no_context(self, incontext_model: InContextMaskGIT) -> None:
        """Test any2one with no context (empty context lists)."""
        samples = [
            InContextSample(
                context_images=[],
                context_modality_ids=[],
                target_image=torch.randn(1, 32, 32, 32),
                target_modality_id=0,
                mask_ratio=0.5,
            ),
            InContextSample(
                context_images=[],
                context_modality_ids=[],
                target_image=torch.randn(1, 32, 32, 32),
                target_modality_id=1,
                mask_ratio=0.5,
            ),
        ]

        training_steps = InContextTrainingSteps(model=incontext_model)
        result = training_steps.training_step_any2one(samples)

        assert "loss" in result
        assert result["loss"].ndim == 0
        assert torch.isfinite(result["loss"])

    def test_any2one_variable_context_counts(self, incontext_model: InContextMaskGIT) -> None:
        """Test any2one with variable context counts (1, 2, 3 modalities)."""
        samples = [
            InContextSample(
                context_images=[torch.randn(1, 32, 32, 32)],
                context_modality_ids=[0],
                target_image=torch.randn(1, 32, 32, 32),
                target_modality_id=1,
                mask_ratio=0.5,
            ),
            InContextSample(
                context_images=[
                    torch.randn(1, 32, 32, 32),
                    torch.randn(1, 32, 32, 32),
                ],
                context_modality_ids=[0, 2],
                target_image=torch.randn(1, 32, 32, 32),
                target_modality_id=1,
                mask_ratio=0.5,
            ),
            InContextSample(
                context_images=[
                    torch.randn(1, 32, 32, 32),
                    torch.randn(1, 32, 32, 32),
                    torch.randn(1, 32, 32, 32),
                ],
                context_modality_ids=[0, 1, 2],
                target_image=torch.randn(1, 32, 32, 32),
                target_modality_id=3,
                mask_ratio=0.5,
            ),
        ]

        training_steps = InContextTrainingSteps(model=incontext_model)
        result = training_steps.training_step_any2one(samples)

        assert "loss" in result
        assert result["loss"].ndim == 0
        assert torch.isfinite(result["loss"])

    def test_any2one_different_modalities_per_sample(
        self, incontext_model: InContextMaskGIT
    ) -> None:
        """Test any2one with different context modalities per sample."""
        samples = [
            InContextSample(
                context_images=[torch.randn(1, 32, 32, 32)],
                context_modality_ids=[0],
                target_image=torch.randn(1, 32, 32, 32),
                target_modality_id=2,
                mask_ratio=0.5,
            ),
            InContextSample(
                context_images=[torch.randn(1, 32, 32, 32)],
                context_modality_ids=[1],
                target_image=torch.randn(1, 32, 32, 32),
                target_modality_id=3,
                mask_ratio=0.5,
            ),
            InContextSample(
                context_images=[
                    torch.randn(1, 32, 32, 32),
                    torch.randn(1, 32, 32, 32),
                ],
                context_modality_ids=[2, 3],
                target_image=torch.randn(1, 32, 32, 32),
                target_modality_id=0,
                mask_ratio=0.5,
            ),
        ]

        training_steps = InContextTrainingSteps(model=incontext_model)
        result = training_steps.training_step_any2one(samples)

        assert "loss" in result
        assert result["loss"].ndim == 0
        assert torch.isfinite(result["loss"])
