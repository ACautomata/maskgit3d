"""Tests for maskgit/transformer.py"""

import pytest
import torch

from maskgit3d.infrastructure.maskgit.transformer import (
    MaskGITTransformer,
    MaskGITTransformerConfig,
    PositionalEncoding3D,
    TransformerBlock,
)


class TestPositionalEncoding3D:
    """Test PositionalEncoding3D class."""

    def test_initialization(self):
        """Test positional encoding initialization."""
        pos_enc = PositionalEncoding3D(num_tokens=64, embed_dim=128)
        assert pos_enc.num_tokens == 64
        assert pos_enc.embed_dim == 128
        assert pos_enc.pos_embed.shape == (1, 64, 128)

    def test_forward_adds_positional_info(self):
        """Test that forward adds positional embeddings."""
        pos_enc = PositionalEncoding3D(num_tokens=64, embed_dim=128)
        x = torch.randn(2, 64, 128)

        output = pos_enc(x)
        assert output.shape == x.shape
        # Output should be different from input (pos embeddings added)
        assert not torch.allclose(output, x)


class TestTransformerBlock:
    """Test TransformerBlock class."""

    @pytest.fixture
    def block(self):
        """Create a transformer block for testing."""
        return TransformerBlock(
            hidden_size=64,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
        )

    def test_forward_shape(self, block):
        """Test that forward preserves shape."""
        x = torch.randn(2, 16, 64)  # [B, N, C]
        output = block(x)
        assert output.shape == x.shape

    def test_forward_with_mask(self, block):
        """Test forward with attention mask."""
        x = torch.randn(2, 16, 64)
        mask = torch.zeros(16, 16)
        mask[0, :] = float("-inf")  # Mask first position

        output = block(x, mask=mask)
        assert output.shape == x.shape


class TestMaskGITTransformer:
    """Test MaskGITTransformer class."""

    @pytest.fixture
    def small_transformer(self):
        """Create a small transformer for testing."""
        return MaskGITTransformer(
            vocab_size=128,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            mlp_ratio=2.0,
            dropout=0.0,
            max_seq_len=256,
        )

    def test_initialization(self, small_transformer):
        """Test transformer initialization."""
        assert small_transformer.vocab_size == 128
        assert small_transformer.hidden_size == 64
        assert small_transformer.num_layers == 2
        assert small_transformer.num_heads == 2
        assert len(small_transformer.blocks) == 2

    def test_encode_returns_logits(self, small_transformer):
        """Test encode method returns logits."""
        tokens = torch.randint(0, 128, (2, 32))
        logits = small_transformer.encode(tokens, return_logits=True)

        assert logits.shape == (2, 32, 128)  # [B, N, vocab_size]

    def test_encode_returns_embeddings(self, small_transformer):
        """Test encode method can return embeddings."""
        tokens = torch.randint(0, 128, (2, 32))
        embeddings = small_transformer.encode(tokens, return_logits=False)

        assert embeddings.shape == (2, 32, 64)  # [B, N, hidden_size]

    def test_forward_without_mask(self, small_transformer):
        """Test forward pass without masking."""
        tokens = torch.randint(0, 128, (2, 32))
        logits = small_transformer(tokens)

        assert logits.shape == (2, 32, 128)

    def test_forward_with_mask(self, small_transformer):
        """Test forward pass with masking."""
        tokens = torch.randint(0, 128, (2, 32))
        mask_indices = torch.zeros(2, 32, dtype=torch.bool)
        mask_indices[:, 10:20] = True

        logits = small_transformer(tokens, mask_indices=mask_indices)
        assert logits.shape == (2, 32, 128)

    def test_predict_masked(self, small_transformer):
        """Test predict_masked method."""
        tokens = torch.randint(0, 128, (2, 32))
        masked_logits, targets, mask = small_transformer.predict_masked(tokens, mask_ratio=0.3)

        # Check shapes
        assert masked_logits.ndim == 2  # [num_masked, vocab_size]
        assert masked_logits.shape[1] == 128
        assert targets.ndim == 1  # [num_masked]
        assert mask.shape == (2, 32)
        assert mask.dtype == torch.bool

        # Check that at least some tokens are masked
        assert mask.sum() > 0

    def test_predict_masked_ensures_at_least_one(self, small_transformer):
        """Test that predict_masked ensures at least one token is masked per sample."""
        tokens = torch.randint(0, 128, (2, 4))  # Small sequence
        masked_logits, targets, mask = small_transformer.predict_masked(
            tokens,
            mask_ratio=0.01,  # Very low ratio
        )

        # Should still have at least 2 masked tokens (1 per sample)
        assert mask.sum() >= 2


class TestMaskGITTransformerConfig:
    """Test MaskGITTransformerConfig class."""

    def test_base_config(self):
        """Test BASE configuration."""
        config = MaskGITTransformerConfig.from_name("base")
        assert config["hidden_size"] == 768
        assert config["num_layers"] == 12
        assert config["num_heads"] == 12

    def test_large_config(self):
        """Test LARGE configuration."""
        config = MaskGITTransformerConfig.from_name("large")
        assert config["hidden_size"] == 1024
        assert config["num_layers"] == 24
        assert config["num_heads"] == 16

    def test_small_config(self):
        """Test SMALL configuration."""
        config = MaskGITTransformerConfig.from_name("small")
        assert config["hidden_size"] == 384
        assert config["num_layers"] == 6
        assert config["num_heads"] == 6

    def test_invalid_config_name(self):
        """Test that invalid config name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            MaskGITTransformerConfig.from_name("invalid")
        assert "Unknown config" in str(exc_info.value)
