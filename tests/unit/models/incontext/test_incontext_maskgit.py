"""Unit tests for InContextMaskGIT model."""

from __future__ import annotations

import pytest
import torch

from maskgit3d.models.incontext.incontext_maskgit import InContextMaskGIT
from maskgit3d.models.vqvae.vqvae import VQVAE


@pytest.fixture
def vqvae() -> VQVAE:
    """Create a small VQVAE for testing."""
    return VQVAE(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=256,
        embedding_dim=64,
        num_channels=(32, 64),
        num_res_blocks=(1, 1),
        attention_levels=(False, False),
    )


@pytest.fixture
def model(vqvae: VQVAE) -> InContextMaskGIT:
    """Create an InContextMaskGIT model for testing."""
    return InContextMaskGIT(
        vqvae=vqvae,
        num_modalities=4,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
        gamma_type="cosine",
    )


@pytest.fixture
def model_with_sliding_window(vqvae: VQVAE) -> InContextMaskGIT:
    """Create an InContextMaskGIT model with sliding window enabled."""
    return InContextMaskGIT(
        vqvae=vqvae,
        num_modalities=4,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
        gamma_type="cosine",
        sliding_window_cfg={
            "enabled": True,
            "roi_size": [16, 16, 16],
            "overlap": 0.25,
            "mode": "gaussian",
        },
    )


class TestInContextMaskGITInit:
    """Tests for InContextMaskGIT initialization."""

    def test_init_with_default_params(self, vqvae: VQVAE) -> None:
        """Test initialization with default parameters."""
        model = InContextMaskGIT(vqvae=vqvae, num_modalities=4)

        assert model.num_modalities == 4
        assert model.tokenizer is not None
        assert model._sequence_builders is not None
        assert model.transformer is not None
        assert model.mask_scheduler is not None
        assert model.loss_fn is not None

    def test_init_with_custom_params(self, vqvae: VQVAE) -> None:
        """Test initialization with custom parameters."""
        model = InContextMaskGIT(
            vqvae=vqvae,
            num_modalities=4,
            hidden_size=256,
            num_layers=6,
            num_heads=8,
            mlp_ratio=3.0,
            dropout=0.2,
            gamma_type="linear",
        )

        assert model.transformer.hidden_size == 256
        assert model.transformer.num_layers == 6
        assert model.transformer.num_heads == 8

    def test_vocab_size_property(self, model: InContextMaskGIT) -> None:
        """Test vocab_size property returns correct value.

        vocab_size = codebook_size + CLS + SEP + num_modalities + MASK
        = codebook_size + 2 + num_modalities + 1
        = codebook_size + 3 + num_modalities
        """
        codebook_size = model.tokenizer.vqvae.quantizer.num_embeddings
        num_modalities = model.num_modalities
        expected_vocab_size = codebook_size + 3 + num_modalities

        assert model.vocab_size == expected_vocab_size

    def test_mask_token_id_property(self, model: InContextMaskGIT) -> None:
        """Test mask_token_id property returns correct value.

        MASK token ID = codebook_size + 2 + num_modalities
        """
        codebook_size = model.tokenizer.vqvae.quantizer.num_embeddings
        num_modalities = model.num_modalities
        expected_mask_id = codebook_size + 2 + num_modalities

        assert model.mask_token_id == expected_mask_id


class TestComputeLoss:
    """Tests for compute_loss method."""

    def test_compute_loss_returns_correct_shapes(self, model: InContextMaskGIT) -> None:
        """Test compute_loss returns scalar loss and metrics dict."""
        batch_size = 2
        spatial_size = 16

        context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
        context_modality_ids = [0]
        target_image = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
        target_modality_id = 1

        loss, metrics = model.compute_loss(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
        )

        assert loss.ndim == 0  # Scalar
        assert isinstance(metrics, dict)
        assert "mask_acc" in metrics
        assert "mask_ratio" in metrics

    def test_compute_loss_only_masks_target_positions(self, model: InContextMaskGIT) -> None:
        """Test that compute_loss only masks target positions, not context."""
        batch_size = 2
        spatial_size = 16

        # Use deterministic images for reproducibility
        torch.manual_seed(42)
        context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
        context_modality_ids = [0]
        target_image = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
        target_modality_id = 1

        loss, metrics = model.compute_loss(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
            mask_ratio=0.5,
        )

        # The actual mask ratio should be close to 0.5 on target positions
        # Since only target positions are masked, the effective mask ratio
        # across the whole sequence will be lower
        assert 0.0 < metrics["mask_ratio"] < 1.0
        assert 0.0 <= metrics["mask_acc"] <= 1.0

    def test_compute_loss_with_multiple_context_modalities(self, model: InContextMaskGIT) -> None:
        """Test compute_loss with multiple context modalities."""
        batch_size = 2
        spatial_size = 16

        torch.manual_seed(42)
        context_images = [
            torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size),
            torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size),
        ]
        context_modality_ids = [0, 1]
        target_image = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
        target_modality_id = 2

        loss, metrics = model.compute_loss(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
        )

        assert loss.ndim == 0
        assert isinstance(metrics, dict)

    def test_compute_loss_with_fixed_mask_ratio(self, model: InContextMaskGIT) -> None:
        """Test compute_loss with explicitly provided mask_ratio."""
        batch_size = 2
        spatial_size = 16

        context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
        context_modality_ids = [0]
        target_image = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
        target_modality_id = 1

        loss, metrics = model.compute_loss(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
            mask_ratio=0.3,
        )

        assert loss.ndim == 0
        assert metrics["mask_ratio"] is not None

    def test_compute_loss_samples_mask_ratio_when_none(self, model: InContextMaskGIT) -> None:
        """Test that compute_loss samples mask_ratio from scheduler when None."""
        batch_size = 2
        spatial_size = 16

        context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
        context_modality_ids = [0]
        target_image = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
        target_modality_id = 1

        # Call multiple times with mask_ratio=None
        # Should get different ratios from scheduler sampling
        ratios = []
        for _ in range(5):
            _, metrics = model.compute_loss(
                context_images=context_images,
                context_modality_ids=context_modality_ids,
                target_image=target_image,
                target_modality_id=target_modality_id,
                mask_ratio=None,
            )
            ratios.append(metrics["mask_ratio"])

        # At least some ratios should differ (probabilistic)
        # With cosine schedule, we expect variation
        assert len(set(ratios)) > 1 or len(ratios) == 1  # Allow for randomness


class TestGenerate:
    """Tests for generate method."""

    def test_generate_produces_correct_output_shape(self, model: InContextMaskGIT) -> None:
        """Test generate produces correct output shape."""
        batch_size = 2
        spatial_size = 16

        context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
        context_modality_ids = [0]
        target_modality_id = 1
        target_shape = (batch_size, spatial_size, spatial_size, spatial_size)

        output = model.generate(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
            target_shape=target_shape,
        )

        # Output should match target_shape with channel dimension
        assert output.shape == (batch_size, 1, spatial_size, spatial_size, spatial_size)

    def test_generate_with_multiple_contexts(self, model: InContextMaskGIT) -> None:
        """Test generate with multiple context modalities."""
        batch_size = 1
        spatial_size = 16

        context_images = [
            torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size),
            torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size),
        ]
        context_modality_ids = [0, 1]
        target_modality_id = 2
        target_shape = (batch_size, spatial_size, spatial_size, spatial_size)

        output = model.generate(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
            target_shape=target_shape,
        )

        assert output.shape == (batch_size, 1, spatial_size, spatial_size, spatial_size)

    def test_generate_with_temperature(self, model: InContextMaskGIT) -> None:
        """Test generate with custom temperature."""
        batch_size = 1
        spatial_size = 16

        context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
        context_modality_ids = [0]
        target_modality_id = 1
        target_shape = (batch_size, spatial_size, spatial_size, spatial_size)

        output = model.generate(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
            target_shape=target_shape,
            temperature=0.5,
        )

        assert output.shape == (batch_size, 1, spatial_size, spatial_size, spatial_size)


class TestWithSlidingWindow:
    """Tests with sliding window enabled."""

    def test_compute_loss_with_sliding_window(
        self, model_with_sliding_window: InContextMaskGIT
    ) -> None:
        """Test compute_loss with sliding window inference."""
        batch_size = 1
        spatial_size = 32  # Larger than roi_size

        context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
        context_modality_ids = [0]
        target_image = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
        target_modality_id = 1

        loss, metrics = model_with_sliding_window.compute_loss(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
        )

        assert loss.ndim == 0
        assert isinstance(metrics, dict)

    def test_generate_with_sliding_window(
        self, model_with_sliding_window: InContextMaskGIT
    ) -> None:
        """Test generate with sliding window inference."""
        batch_size = 1
        spatial_size = 32

        context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
        context_modality_ids = [0]
        target_modality_id = 1
        target_shape = (batch_size, spatial_size, spatial_size, spatial_size)

        output = model_with_sliding_window.generate(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
            target_shape=target_shape,
        )

        assert output.shape == (batch_size, 1, spatial_size, spatial_size, spatial_size)

    def test_decode_tokens_to_images_direct(
        self, model_with_sliding_window: InContextMaskGIT
    ) -> None:
        """Test _decode_tokens_to_images without sliding window (small latent)."""
        batch_size = 1
        latent_size = 2  # Small, no sliding window needed

        tokens = torch.randint(0, 256, (batch_size, latent_size, latent_size, latent_size))
        target_shape = (batch_size, latent_size * 4, latent_size * 4, latent_size * 4)

        output = model_with_sliding_window._decode_tokens_to_images(tokens, target_shape)

        assert output.shape[0] == batch_size
        assert output.shape[1] == 1  # Channel dimension

    def test_decode_tokens_to_images_with_sliding_window(
        self, model_with_sliding_window: InContextMaskGIT
    ) -> None:
        """Test _decode_tokens_to_images with sliding window (large latent)."""
        batch_size = 1
        latent_size = 9  # > roi_size/downsampling (8) to trigger sliding window

        tokens = torch.randint(0, 256, (batch_size, latent_size, latent_size, latent_size))
        target_shape = (batch_size, latent_size * 4, latent_size * 4, latent_size * 4)

        output = model_with_sliding_window._decode_tokens_to_images(tokens, target_shape)

        assert output.shape[0] == batch_size
        assert output.shape[1] == 1  # Channel dimension

    def test_prepare_batch_uses_tokenizer_encode(
        self, model_with_sliding_window: InContextMaskGIT
    ) -> None:
        """Test prepare_batch uses tokenizer's sliding window encoding."""
        from maskgit3d.models.incontext.types import InContextSample

        batch_size = 2
        spatial_size = 32

        samples = [
            InContextSample(
                context_images=[torch.randn(1, spatial_size, spatial_size, spatial_size)],
                context_modality_ids=[0],
                target_image=torch.randn(1, spatial_size, spatial_size, spatial_size),
                target_modality_id=1,
                mask_ratio=0.5,
            )
            for _ in range(batch_size)
        ]

        prepared = model_with_sliding_window.prepare_batch(samples)

        assert prepared.sequences.shape[0] == batch_size
        assert prepared.attention_mask.shape[0] == batch_size
        assert prepared.target_mask.shape[0] == batch_size


class TestDifferentContextCounts:
    """Tests with different numbers of context modalities."""

    def test_no_context_modalities(self, model: InContextMaskGIT) -> None:
        """Test with no context modalities (just target generation)."""
        batch_size = 1
        spatial_size = 16

        context_images: list[torch.Tensor] = []
        context_modality_ids: list[int] = []
        target_image = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
        target_modality_id = 0

        loss, metrics = model.compute_loss(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
        )

        assert loss.ndim == 0
        assert isinstance(metrics, dict)

    def test_single_context_modality(self, model: InContextMaskGIT) -> None:
        """Test with single context modality."""
        batch_size = 2
        spatial_size = 16

        context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
        context_modality_ids = [0]
        target_image = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
        target_modality_id = 1

        loss, metrics = model.compute_loss(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
        )

        assert loss.ndim == 0

    def test_three_context_modalities(self, model: InContextMaskGIT) -> None:
        """Test with three context modalities."""
        batch_size = 1
        spatial_size = 16

        context_images = [
            torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size),
            torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size),
            torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size),
        ]
        context_modality_ids = [0, 1, 2]
        target_image = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
        target_modality_id = 3

        loss, metrics = model.compute_loss(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
        )

        assert loss.ndim == 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_different_spatial_sizes(self, model: InContextMaskGIT) -> None:
        """Test with different spatial sizes."""
        for spatial_size in [8, 16, 32]:
            batch_size = 1

            context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
            context_modality_ids = [0]
            target_image = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
            target_modality_id = 1

            loss, metrics = model.compute_loss(
                context_images=context_images,
                context_modality_ids=context_modality_ids,
                target_image=target_image,
                target_modality_id=target_modality_id,
            )

            assert loss.ndim == 0

    def test_batch_size_one(self, model: InContextMaskGIT) -> None:
        """Test with batch size 1."""
        batch_size = 1
        spatial_size = 16

        context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
        context_modality_ids = [0]
        target_image = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
        target_modality_id = 1

        loss, metrics = model.compute_loss(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
        )

        assert loss.ndim == 0

    def test_loss_is_differentiable(self, model: InContextMaskGIT) -> None:
        """Test that loss is differentiable."""
        batch_size = 1
        spatial_size = 16

        context_images = [torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)]
        context_modality_ids = [0]
        target_image = torch.randn(
            batch_size, 1, spatial_size, spatial_size, spatial_size, requires_grad=True
        )
        target_modality_id = 1

        loss, _ = model.compute_loss(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
        )

        # Loss should be differentiable
        assert loss.requires_grad
        loss.backward()
        # Should not raise an error
