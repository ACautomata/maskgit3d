"""Tests for MaskGIT model - additional coverage."""

import pytest
import torch

from maskgit3d.models.vqvae import VQVAE
from maskgit3d.models.maskgit import MaskGIT


@pytest.fixture
def small_vqvae():
    """Create a small VQVAE for testing."""
    model = VQVAE(
        in_channels=1,
        out_channels=1,
        latent_channels=32,
        num_embeddings=50,
        embedding_dim=32,
    )
    model.eval()
    model.requires_grad_(False)
    return model


@pytest.fixture
def small_maskgit(small_vqvae):
    """Create a small MaskGIT for testing."""
    model = MaskGIT(
        vqvae=small_vqvae,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
    )
    model.eval()
    return model


class TestMaskGITTokenConversion:
    """Tests for token conversion methods."""

    def test_to_transformer_tokens_basic(self, small_maskgit) -> None:
        """Test basic token conversion."""
        tokens = torch.tensor([0, 1, 2, 3])
        converted = small_maskgit._to_transformer_tokens(tokens)

        # Tokens should be shifted
        expected = (tokens + 1) % small_maskgit.codebook_size
        assert torch.equal(converted, expected)

    def test_to_transformer_tokens_out_of_range(self, small_maskgit) -> None:
        tokens = torch.tensor([-1, 0, 1])
        with pytest.raises(ValueError, match="VQVAE token indices out of range"):
            small_maskgit._to_transformer_tokens(tokens)

        tokens = torch.tensor([0, small_maskgit.codebook_size])
        with pytest.raises(ValueError, match="VQVAE token indices out of range"):
            small_maskgit._to_transformer_tokens(tokens)

    def test_to_vq_tokens_basic(self, small_maskgit) -> None:
        """Test basic VQ token conversion."""
        # Create tokens in valid range (0 to codebook_size-1)
        tokens = torch.tensor([1, 2, 3, 4])
        converted = small_maskgit._to_vq_tokens(tokens)

        expected = (tokens - 1) % small_maskgit.codebook_size
        assert torch.equal(converted, expected)

    def test_to_vq_tokens_with_mask_token(self, small_maskgit) -> None:
        """Test that mask tokens raise error."""
        tokens = torch.tensor([0, small_maskgit.mask_token_id, 2])
        with pytest.raises(ValueError, match="Cannot decode tokens that still contain mask token"):
            small_maskgit._to_vq_tokens(tokens)

    def test_to_vq_tokens_out_of_range(self, small_maskgit) -> None:
        tokens = torch.tensor([-1, 1, 2])
        with pytest.raises(ValueError, match="Token indices out of range"):
            small_maskgit._to_vq_tokens(tokens)

        tokens = torch.tensor([small_maskgit.codebook_size + 1, 1, 2])
        with pytest.raises(ValueError, match="Token indices out of range"):
            small_maskgit._to_vq_tokens(tokens)


class TestMaskGITComputeLoss:
    """Tests for compute_loss method."""

    def test_compute_loss_basic(self, small_maskgit) -> None:
        """Test basic compute_loss call."""
        x = torch.randn(2, 1, 8, 8, 8)

        loss, metrics = small_maskgit.compute_loss(x)

        assert loss.ndim == 0  # Scalar loss
        assert "loss" in metrics
        assert "mask_acc" in metrics
        assert "mask_ratio" in metrics

    def test_compute_loss_with_mask_ratio(self, small_maskgit) -> None:
        x = torch.randn(2, 1, 8, 8, 8)

        loss, metrics = small_maskgit.compute_loss(x, mask_ratio=0.5)

        assert "mask_ratio" in metrics

    def test_compute_loss_different_shapes(self, small_maskgit) -> None:
        """Test compute_loss with different input shapes."""
        # Different spatial sizes
        for size in [4, 8, 12]:
            x = torch.randn(1, 1, size, size, size)
            loss, metrics = small_maskgit.compute_loss(x)
            assert loss.ndim == 0


class TestMaskGITGenerate:
    """Tests for generate method."""

    def test_generate_with_invalid_shape(self, small_maskgit) -> None:
        """Test that invalid shape raises error."""
        with pytest.raises(ValueError, match="shape must be a 4D tuple"):
            small_maskgit.generate(shape=(1, 2, 2))  # 3D instead of 4D

        with pytest.raises(ValueError, match="shape must be a 4D tuple"):
            small_maskgit.generate(shape=(1, 2, 2, 2, 2))  # 5D instead of 4D

    def test_generate_basic(self, small_maskgit) -> None:
        """Test basic generation."""
        with torch.no_grad():
            output = small_maskgit.generate(shape=(1, 4, 4, 4), num_iterations=2)

        assert output.shape[0] == 1

    def test_generate_default_shape(self, small_maskgit) -> None:
        """Test generation with default shape."""
        with torch.no_grad():
            output = small_maskgit.generate(num_iterations=2)

        assert output.shape[0] == 1


class TestMaskGITProperties:
    """Tests for MaskGIT properties."""

    def test_codebook_size(self, small_maskgit, small_vqvae) -> None:
        """Test codebook_size property."""
        assert small_maskgit.codebook_size == small_vqvae.quantizer.num_embeddings

    def test_num_tokens(self, small_maskgit) -> None:
        """Test num_tokens property."""
        assert small_maskgit.num_tokens == small_maskgit.codebook_size + 1


class TestMaskGITForward:
    """Tests for forward pass."""

    def test_forward_basic(self, small_maskgit) -> None:
        """Test basic forward pass."""
        x = torch.randn(1, 1, 8, 8, 8)
        output = small_maskgit(x)
        assert output.shape == x.shape

    def test_forward_with_different_spatial_sizes(self, small_maskgit) -> None:
        """Test forward pass with different spatial sizes."""
        x = torch.randn(1, 1, 16, 16, 16)
        output = small_maskgit(x)
        assert output.shape == x.shape

    def test_encode_decode_tokens(self, small_maskgit) -> None:
        """Test encode and decode tokens."""
        x = torch.randn(1, 1, 8, 8, 8)

        with torch.no_grad():
            tokens = small_maskgit.encode_tokens(x)
            decoded = small_maskgit.decode_tokens(tokens)

        assert tokens.dim() == 4
        assert decoded.shape[2:] == x.shape[2:]
