"""Additional tests for maskgit_model to improve coverage."""

import pytest
import torch
import torch.nn as nn

from maskgit3d.infrastructure.maskgit.maskgit_model import MaskGITModel, MaskGITModelConfig


class MockVQGAN(nn.Module):
    """Mock VQGAN for testing."""

    def __init__(self):
        super().__init__()
        self._codebook_size = 512
        self._latent_shape = (256, 4, 4, 4)

        # Create quantize module with embedding
        self.quantize = MockQuantize(512, 256)

    @property
    def codebook_size(self):
        return self._codebook_size

    @property
    def latent_shape(self):
        return self._latent_shape

    def encode(self, x):
        """Mock encode."""
        batch_size = x.shape[0]
        z = torch.randn(batch_size, 256, 4, 4, 4, device=x.device)
        indices = torch.randint(0, 512, (batch_size, 4, 4, 4), device=x.device)
        info = (None, None, indices)
        return z, torch.tensor(0.1), info

    def decode(self, quant):
        """Mock decode."""
        batch_size = quant.shape[0]
        return torch.randn(batch_size, 1, 16, 16, 16, device=quant.device)

    def decode_code(self, codes):
        """Mock decode_code."""
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
        if shape is not None:
            b, d, h, w, c = shape
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
        self.embed = nn.Embedding(513, 256)  # +1 for mask token

    def forward(self, tokens, mask_indices=None):
        batch_size, seq_len = tokens.shape
        return torch.randn(batch_size, seq_len, self.codebook_size, device=tokens.device)

    def encode(self, tokens, return_logits=False):
        batch_size = tokens.shape[0]
        seq_len = tokens.numel() // batch_size
        if return_logits:
            return torch.randn(batch_size, seq_len, self.codebook_size, device=tokens.device)
        return tokens


class TestMaskGITModelInit:
    """Tests for MaskGITModel initialization."""

    def test_init(self):
        """Test model initialization."""
        vqgan = MockVQGAN()
        transformer = MockTransformer()
        model = MaskGITModel(vqgan=vqgan, transformer=transformer, mask_ratio=0.5)

        assert model.vqgan is not None
        assert model.transformer is not None
        assert model.mask_ratio == 0.5
        assert model.codebook_size == 512

    def test_num_tokens_property(self):
        """Test num_tokens property."""
        vqgan = MockVQGAN()
        transformer = MockTransformer()
        model = MaskGITModel(vqgan=vqgan, transformer=transformer)

        # Should be codebook_size + 1 (for mask token)
        assert model.num_tokens == 513

    def test_device_property(self):
        """Test device property."""
        vqgan = MockVQGAN()
        transformer = MockTransformer()
        model = MaskGITModel(vqgan=vqgan, transformer=transformer)

        device = model.device
        assert isinstance(device, torch.device)


class TestMaskGITModelEncodeDecode:
    """Tests for encode/decode methods."""

    @pytest.fixture
    def model(self):
        vqgan = MockVQGAN()
        transformer = MockTransformer()
        return MaskGITModel(vqgan=vqgan, transformer=transformer)

    def test_encode_tokens(self, model):
        """Test encode_tokens method."""
        batch_size = 2
        x = torch.randn(batch_size, 1, 16, 16, 16)

        tokens = model.encode_tokens(x)

        assert tokens.shape[0] == batch_size
        assert len(tokens.shape) == 4  # [B, D, H, W]

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


class TestMaskGITModelCheckpoint:
    """Tests for checkpoint methods."""

    @pytest.fixture
    def model(self):
        vqgan = MockVQGAN()
        transformer = MockTransformer()
        return MaskGITModel(vqgan=vqgan, transformer=transformer)

    def test_save_and_load_checkpoint(self, model, tmp_path):
        """Test save and load checkpoint."""
        checkpoint_path = tmp_path / "test_checkpoint.pth"

        # Save
        model.save_checkpoint(str(checkpoint_path))
        assert checkpoint_path.exists()

        # Load
        model.load_checkpoint(str(checkpoint_path))


class TestMaskGITModelConfig:
    """Tests for MaskGITModelConfig."""

    def test_create_config(self):
        """Test create_config."""
        config = MaskGITModelConfig.create_config()

        assert "in_channels" in config
        assert "codebook_size" in config
        assert "embed_dim" in config
        assert "transformer_hidden" in config

    def test_create_config_custom(self):
        """Test create_config with custom values."""
        config = MaskGITModelConfig.create_config(
            image_size=128,
            codebook_size=1024,
        )

        assert config["resolution"] == 128
        assert config["codebook_size"] == 1024
