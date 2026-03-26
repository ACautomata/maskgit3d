"""Unit tests for InContextTokenizer."""

from __future__ import annotations

import pytest
import torch

from maskgit3d.models.incontext.tokenizer import InContextTokenizer
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
def tokenizer(vqvae: VQVAE) -> InContextTokenizer:
    """Create an InContextTokenizer for testing."""
    return InContextTokenizer(
        vqvae=vqvae,
        num_modalities=4,
        downsampling_factor=2,
    )


@pytest.fixture
def tokenizer_with_sliding_window(vqvae: VQVAE) -> InContextTokenizer:
    """Create an InContextTokenizer with sliding window enabled."""
    return InContextTokenizer(
        vqvae=vqvae,
        num_modalities=4,
        downsampling_factor=2,
        sliding_window_cfg={
            "enabled": True,
            "roi_size": [16, 16, 16],
            "overlap": 0.25,
            "mode": "gaussian",
        },
    )


class TestInContextTokenizerInit:
    """Tests for InContextTokenizer initialization."""

    def test_init_creates_modality_embeddings(self, vqvae: VQVAE) -> None:
        tokenizer = InContextTokenizer(vqvae=vqvae, num_modalities=4, downsampling_factor=2)
        assert tokenizer.modality_embeddings is not None
        assert tokenizer.modality_embeddings.num_embeddings == 4
        assert tokenizer.modality_embeddings.embedding_dim == vqvae.quantizer.embedding_dim

    def test_init_default_sliding_window_cfg(self, vqvae: VQVAE) -> None:
        tokenizer = InContextTokenizer(vqvae=vqvae, num_modalities=4, downsampling_factor=2)
        assert tokenizer.sliding_window_cfg == {}
        assert tokenizer._sliding_window_inferer is None

    def test_init_with_sliding_window_cfg(self, vqvae: VQVAE) -> None:
        cfg = {"enabled": True, "roi_size": [16, 16, 16]}
        tokenizer = InContextTokenizer(
            vqvae=vqvae, num_modalities=4, downsampling_factor=2, sliding_window_cfg=cfg
        )
        assert tokenizer.sliding_window_cfg == cfg


class TestEncodeImagesToLatents:
    """Tests for encode_images_to_latents method."""

    def test_encode_returns_correct_shape(self, tokenizer: InContextTokenizer) -> None:
        batch_size = 2
        spatial_size = 32
        images = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)

        indices = tokenizer.encode_images_to_latents(images)

        expected_latent_size = spatial_size // tokenizer.downsampling_factor
        assert indices.shape == (
            batch_size,
            expected_latent_size,
            expected_latent_size,
            expected_latent_size,
        )

    def test_encode_without_sliding_window(self, tokenizer: InContextTokenizer) -> None:
        images = torch.randn(2, 1, 32, 32, 32)

        indices = tokenizer.encode_images_to_latents(images)

        assert indices.shape == (2, 16, 16, 16)
        assert indices.dtype == torch.long

    def test_encode_with_sliding_window(
        self, tokenizer_with_sliding_window: InContextTokenizer
    ) -> None:
        images = torch.randn(1, 1, 32, 32, 32)

        indices = tokenizer_with_sliding_window.encode_images_to_latents(images)

        assert indices.shape == (1, 16, 16, 16)

    def test_encode_different_spatial_sizes(self, tokenizer: InContextTokenizer) -> None:
        for spatial_size in [16, 32, 64]:
            images = torch.randn(1, 1, spatial_size, spatial_size, spatial_size)
            indices = tokenizer.encode_images_to_latents(images)
            expected = spatial_size // tokenizer.downsampling_factor
            assert indices.shape == (1, expected, expected, expected)


class TestEncodeModalities:
    """Tests for encode_modalities method."""

    def test_encode_single_modality(self, tokenizer: InContextTokenizer) -> None:
        images = torch.randn(2, 1, 32, 32, 32)

        indices_list = tokenizer.encode_modalities([images], [0])

        assert len(indices_list) == 1
        assert indices_list[0].shape == (2, 16, 16, 16)

    def test_encode_multiple_modalities(self, tokenizer: InContextTokenizer) -> None:
        t1_images = torch.randn(2, 1, 32, 32, 32)
        t2_images = torch.randn(2, 1, 32, 32, 32)

        indices_list = tokenizer.encode_modalities([t1_images, t2_images], [0, 1])

        assert len(indices_list) == 2
        for indices in indices_list:
            assert indices.shape == (2, 16, 16, 16)

    def test_encode_raises_on_length_mismatch(self, tokenizer: InContextTokenizer) -> None:
        images = torch.randn(2, 1, 32, 32, 32)

        with pytest.raises(ValueError, match="Length mismatch"):
            tokenizer.encode_modalities([images], [0, 1])

    def test_encode_raises_on_invalid_modality_id(self, tokenizer: InContextTokenizer) -> None:
        images = torch.randn(2, 1, 32, 32, 32)

        with pytest.raises(ValueError, match="out of range"):
            tokenizer.encode_modalities([images], [10])

    def test_encode_raises_on_negative_modality_id(self, tokenizer: InContextTokenizer) -> None:
        images = torch.randn(2, 1, 32, 32, 32)

        with pytest.raises(ValueError, match="out of range"):
            tokenizer.encode_modalities([images], [-1])

    def test_different_modalities_produce_different_indices(
        self, tokenizer: InContextTokenizer
    ) -> None:
        torch.manual_seed(42)
        images = torch.randn(2, 1, 32, 32, 32)

        indices_0 = tokenizer.encode_modalities([images], [0])[0]
        indices_1 = tokenizer.encode_modalities([images], [1])[0]

        assert not torch.equal(indices_0, indices_1)


class TestForward:
    """Tests for forward method."""

    def test_forward_without_modality_id(self, tokenizer: InContextTokenizer) -> None:
        images = torch.randn(2, 1, 32, 32, 32)

        indices = tokenizer(images)

        assert indices.shape == (2, 16, 16, 16)

    def test_forward_with_modality_id(self, tokenizer: InContextTokenizer) -> None:
        images = torch.randn(2, 1, 32, 32, 32)

        indices = tokenizer(images, modality_id=0)

        assert indices.shape == (2, 16, 16, 16)

    def test_forward_consistency_with_encode_methods(self, tokenizer: InContextTokenizer) -> None:
        tokenizer.eval()
        images = torch.randn(2, 1, 32, 32, 32)

        indices_forward_no_mod = tokenizer(images)
        indices_encode = tokenizer.encode_images_to_latents(images)

        assert torch.equal(indices_forward_no_mod, indices_encode)

        indices_forward_with_mod = tokenizer(images, modality_id=0)
        indices_encode_mod = tokenizer.encode_modalities([images], [0])[0]

        assert torch.equal(indices_forward_with_mod, indices_encode_mod)


class TestModalityEmbedding:
    """Tests for modality embedding behavior."""

    def test_modality_embeddings_are_learnable(self, tokenizer: InContextTokenizer) -> None:
        assert tokenizer.modality_embeddings.weight.requires_grad

    def test_modality_embedding_affects_latent(self, tokenizer: InContextTokenizer) -> None:
        torch.manual_seed(42)
        images = torch.randn(1, 1, 16, 16, 16)

        indices_no_mod = tokenizer.encode_images_to_latents(images)
        indices_with_mod = tokenizer.encode_modalities([images], [0])[0]

        assert not torch.equal(indices_no_mod, indices_with_mod)

    def test_different_modality_embeddings_are_different(
        self, tokenizer: InContextTokenizer
    ) -> None:
        emb_0 = tokenizer.modality_embeddings.weight[0]
        emb_1 = tokenizer.modality_embeddings.weight[1]

        assert not torch.equal(emb_0, emb_1)


class TestSlidingWindow:
    """Tests for sliding window inference."""

    def test_sliding_window_larger_than_image(
        self, tokenizer_with_sliding_window: InContextTokenizer
    ) -> None:
        images = torch.randn(1, 1, 16, 16, 16)

        indices = tokenizer_with_sliding_window.encode_images_to_latents(images)

        assert indices.shape == (1, 8, 8, 8)

    def test_sliding_window_smaller_than_image(
        self, tokenizer_with_sliding_window: InContextTokenizer
    ) -> None:
        images = torch.randn(1, 1, 64, 64, 64)

        indices = tokenizer_with_sliding_window.encode_images_to_latents(images)

        assert indices.shape == (1, 32, 32, 32)

    def test_sliding_window_produces_valid_indices(
        self, tokenizer_with_sliding_window: InContextTokenizer
    ) -> None:
        images = torch.randn(1, 1, 32, 32, 32)

        indices = tokenizer_with_sliding_window.encode_images_to_latents(images)

        num_embeddings = tokenizer_with_sliding_window.vqvae.quantizer.num_embeddings
        assert indices.min() >= 0
        assert indices.max() < num_embeddings


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_batch_size_one(self, tokenizer: InContextTokenizer) -> None:
        images = torch.randn(1, 1, 32, 32, 32)

        indices = tokenizer.encode_images_to_latents(images)

        assert indices.shape == (1, 16, 16, 16)

    def test_minimum_spatial_size(self, tokenizer: InContextTokenizer) -> None:
        min_size = tokenizer.downsampling_factor
        images = torch.randn(1, 1, min_size, min_size, min_size)

        indices = tokenizer.encode_images_to_latents(images)

        assert indices.shape == (1, 1, 1, 1)

    def test_multi_channel_input(self, vqvae: VQVAE) -> None:
        multi_channel_vqvae = VQVAE(
            in_channels=3,
            out_channels=3,
            latent_channels=64,
            num_embeddings=256,
            embedding_dim=64,
            num_channels=(32, 64),
            num_res_blocks=(1, 1),
            attention_levels=(False, False),
        )
        tokenizer = InContextTokenizer(
            vqvae=multi_channel_vqvae, num_modalities=2, downsampling_factor=2
        )
        images = torch.randn(2, 3, 32, 32, 32)

        indices = tokenizer.encode_images_to_latents(images)

        assert indices.shape == (2, 16, 16, 16)

    def test_max_modality_id(self, tokenizer: InContextTokenizer) -> None:
        images = torch.randn(1, 1, 16, 16, 16)
        max_mod_id = tokenizer.num_modalities - 1

        indices = tokenizer.encode_modalities([images], [max_mod_id])

        assert len(indices) == 1
        assert indices[0].shape == (1, 8, 8, 8)


class TestEncodeWithModality:
    """Tests for encode_with_modality method."""

    def test_encode_with_modality_returns_correct_shape(
        self, tokenizer: InContextTokenizer
    ) -> None:
        images = torch.randn(2, 1, 32, 32, 32)
        modality_id = 1

        indices = tokenizer.encode_with_modality(images, modality_id)

        assert indices.shape == (2, 16, 16, 16)

    def test_encode_with_modality_adds_modality_embedding(
        self, tokenizer: InContextTokenizer
    ) -> None:
        images = torch.randn(1, 1, 16, 16, 16)

        indices_no_mod = tokenizer.encode_images_to_latents(images)
        indices_with_mod = tokenizer.encode_with_modality(images, 0)

        assert not torch.equal(indices_no_mod, indices_with_mod)

    def test_encode_with_modality_different_modalities(self, tokenizer: InContextTokenizer) -> None:
        images = torch.randn(1, 1, 16, 16, 16)

        indices_mod0 = tokenizer.encode_with_modality(images, 0)
        indices_mod1 = tokenizer.encode_with_modality(images, 1)

        assert not torch.equal(indices_mod0, indices_mod1)

    def test_encode_with_modality_with_sliding_window(
        self, tokenizer_with_sliding_window: InContextTokenizer
    ) -> None:
        images = torch.randn(1, 1, 32, 32, 32)
        modality_id = 1

        indices = tokenizer_with_sliding_window.encode_with_modality(images, modality_id)

        assert indices.shape == (1, 16, 16, 16)

    def test_encode_with_modality_invalid_modality_id(self, tokenizer: InContextTokenizer) -> None:
        images = torch.randn(1, 1, 16, 16, 16)

        with pytest.raises(ValueError, match="modality_id"):
            tokenizer.encode_with_modality(images, -1)

        with pytest.raises(ValueError, match="modality_id"):
            tokenizer.encode_with_modality(images, 100)
