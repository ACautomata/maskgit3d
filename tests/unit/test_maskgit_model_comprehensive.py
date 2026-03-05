"""Additional tests for maskgit_model to reach 90%+ coverage."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from maskgit3d.infrastructure.maskgit.maskgit_model import MaskGITModel, MaskGITModelConfig


class MockVQGAN(nn.Module):
    """Mock VQGAN for testing."""

    def __init__(self):
        super().__init__()
        self._codebook_size = 512
        self._latent_shape = (1, 256, 4, 4, 4)
        self.quantize = MockQuantize(512, 256)

    @property
    def codebook_size(self):
        return self._codebook_size

    @property
    def latent_shape(self):
        return self._latent_shape

    def encode(self, x):
        batch_size = x.shape[0]
        z = torch.randn(batch_size, 256, 4, 4, 4, device=x.device)
        indices = torch.randint(0, 512, (batch_size, 4, 4, 4), device=x.device)
        info = (None, None, indices)
        return z, torch.tensor(0.1), info

    def decode(self, quant):
        batch_size = quant.shape[0]
        return torch.randn(batch_size, 1, 16, 16, 16, device=quant.device)

    def decode_code(self, codes):
        batch_size = codes.shape[0]
        return torch.randn(batch_size, 1, 16, 16, 16, device=codes.device)


class MockQuantize(nn.Module):
    """Mock quantize module."""

    def __init__(self, n_embed, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_embed, embed_dim)

    def forward(self, z):
        batch_size = z.shape[0]
        indices = torch.randint(0, 512, (batch_size, 4, 4, 4), device=z.device)
        info = (None, None, indices)
        return z, torch.tensor(0.1), info

    def get_codebook_entry(self, indices, shape=None):
        if shape is not None:
            if len(shape) == 5:
                b, d, h, w, c = shape
            else:
                b, d, h, w = shape
            indices = indices.view(b, d, h, w)
            z_q = self.embedding(indices)
            z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
            return z_q
        return self.embedding(indices)


class MockTransformer(nn.Module):
    """Mock transformer for testing."""

    def __init__(self):
        super().__init__()
        self.codebook_size = 512
        self.embed = nn.Embedding(513, 256)

    def forward(self, tokens, mask_indices=None):
        batch_size, seq_len = tokens.shape
        return torch.randn(batch_size, seq_len, self.codebook_size, device=tokens.device)

    def encode(self, tokens, return_logits=False):
        batch_size = tokens.shape[0]
        seq_len = tokens.numel() // batch_size
        if return_logits:
            return torch.randn(batch_size, seq_len, self.codebook_size, device=tokens.device)
        return tokens


class TestMaskGITModelComprehensive:
    """Comprehensive tests for MaskGITModel."""

    @pytest.fixture
    def model(self):
        vqgan = MockVQGAN()
        transformer = MockTransformer()
        return MaskGITModel(vqgan=vqgan, transformer=transformer, mask_ratio=0.5)

    def test_init(self, model):
        """Test model initialization."""
        assert model.vqgan is not None
        assert model.transformer is not None
        assert model.codebook_size == 512

    def test_embed_dim_property(self, model):
        """Test embed_dim property."""
        assert model.embed_dim == 256

    def test_num_tokens_property(self, model):
        """Test num_tokens property includes mask token."""
        assert model.num_tokens == 513  # 512 + 1

    def test_codebook_size_property(self, model):
        """Test codebook_size property."""
        assert model.codebook_size == 512

    def test_latent_shape_property(self, model):
        """Test latent_shape property."""
        assert model.latent_shape == (1, 256, 4, 4, 4)

    def test_device_property(self, model):
        """Test device property."""
        device = model.device
        assert isinstance(device, torch.device)

    def test_forward(self, model):
        """Test forward pass."""
        batch_size = 2
        x = torch.randn(batch_size, 1, 16, 16, 16)

        output = model(x)

        assert output.shape[0] == batch_size

    def test_encode_tokens(self, model):
        """Test encode_tokens method."""
        batch_size = 2
        x = torch.randn(batch_size, 1, 16, 16, 16)

        tokens = model.encode_tokens(x)

        assert tokens.shape[0] == batch_size
        assert len(tokens.shape) == 4

    def test_decode_tokens_4d(self, model):
        """Test decode_tokens with 4D tokens."""
        batch_size = 2
        tokens = torch.randint(0, 512, (batch_size, 4, 4, 4))

        output = model.decode_tokens(tokens)

        assert output.shape[0] == batch_size

    def test_decode_tokens_2d(self, model):
        """Test decode_tokens with 2D tokens."""
        batch_size = 2
        tokens = torch.randint(0, 512, (batch_size, 64))

        output = model.decode_tokens(tokens)

        assert output.shape[0] == batch_size

    def test_generate_with_shape(self, model):
        """Test generate with explicit shape."""
        shape = (2, 4, 4, 4)

        with patch.object(model, "decode_tokens", return_value=torch.randn(2, 1, 16, 16, 16)):
            output = model.generate(shape=shape)

        assert output.shape[0] == 2

    def test_generate_without_shape(self, model):
        """Test generate without shape uses latent_shape."""
        with patch.object(model, "decode_tokens", return_value=torch.randn(1, 1, 16, 16, 16)):
            output = model.generate(shape=(1, 4, 4, 4))

        assert output.shape[0] == 1

    def test_generate_invalid_shape(self, model):
        """Test generate raises error for invalid shape."""
        with pytest.raises(ValueError, match="shape must be a 4D tuple"):
            model.generate(shape=(8, 8))

    def test_compute_maskgit_loss(self, model):
        """Test compute_maskgit_loss method."""
        batch_size = 2
        x = torch.randn(batch_size, 1, 16, 16, 16)

        loss, metrics = model.compute_maskgit_loss(x, mask_ratio=0.5)

        assert isinstance(loss, torch.Tensor)
        assert "loss" in metrics
        assert "mask_acc" in metrics
        assert "mask_ratio" in metrics

    def test_save_checkpoint(self, model, tmp_path):
        """Test save_checkpoint method."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        model.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Verify content
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        assert "vqgan" in checkpoint
        assert "transformer" in checkpoint

    def test_load_checkpoint(self, model, tmp_path):
        """Test load_checkpoint method."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        # First save
        model.save_checkpoint(str(checkpoint_path))

        # Then load
        model.load_checkpoint(str(checkpoint_path))

        # Should not raise


class TestMaskGITModelConfigComprehensive:
    """Comprehensive tests for MaskGITModelConfig."""

    def test_create_config_defaults(self):
        """Test create_config with default values."""
        config = MaskGITModelConfig.create_config()

        assert config["in_channels"] == 1
        assert config["codebook_size"] == 1024
        assert config["embed_dim"] == 256
        assert config["latent_channels"] == 256
        assert config["resolution"] == 64
        assert config["channel_multipliers"] == (1, 1, 2, 2, 4)
        assert config["transformer_hidden"] == 768
        assert config["transformer_layers"] == 12
        assert config["transformer_heads"] == 12
        assert config["mask_ratio"] == 0.5

    def test_create_config_custom(self):
        """Test create_config with custom values."""
        config = MaskGITModelConfig.create_config(
            image_size=128,
            in_channels=3,
            codebook_size=2048,
            embed_dim=512,
            latent_channels=512,
            transformer_hidden=1024,
            transformer_layers=24,
            transformer_heads=16,
            mask_ratio=0.75,
        )

        assert config["in_channels"] == 3
        assert config["codebook_size"] == 2048
        assert config["embed_dim"] == 512
        assert config["latent_channels"] == 512
        assert config["resolution"] == 128
        assert config["transformer_hidden"] == 1024
        assert config["transformer_layers"] == 24
        assert config["transformer_heads"] == 16
        assert config["mask_ratio"] == 0.75
