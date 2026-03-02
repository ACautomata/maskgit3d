"""Additional tests for base_vq_model to improve coverage."""

import pytest
import torch
import torch.nn as nn

from maskgit3d.infrastructure.vqgan.base_vq_model import BaseVQModel


class TestableVQModel(BaseVQModel):
    """Testable implementation of BaseVQModel."""

    def __init__(self):
        super().__init__(
            in_channels=1,
            codebook_size=512,
            embed_dim=256,
            latent_channels=256,
        )
        # Create quantize with get_codebook_entry method
        self.quantize = MockQuantize(512, 256)
        self.decoder = nn.Conv3d(256, 1, 1)

    def encode(self, x):
        batch_size = x.shape[0]
        z = torch.randn(batch_size, 256, 4, 4, 4, device=x.device)
        info = (None, None, torch.randint(0, 512, (batch_size, 4, 4, 4), device=x.device))
        return z, torch.tensor(0.1), info

    def decode(self, quant):
        return self.decoder(quant)

    @property
    def latent_shape(self):
        return (256, 4, 4, 4)


class MockQuantize(nn.Module):
    """Mock quantize with get_codebook_entry."""

    def __init__(self, n_embed, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_embed, embed_dim)

    def forward(self, z):
        batch_size = z.shape[0]
        info = (None, None, torch.randint(0, 512, (batch_size, 4, 4, 4), device=z.device))
        return z, torch.tensor(0.1), info

    def get_codebook_entry(self, indices, shape=None):
        if shape is not None:
            b, d, h, w, c = shape
            indices = indices.view(b, d, h, w)
            z_q = self.embedding(indices)
            z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
            return z_q
        return self.embedding(indices)


class TestBaseVQModelDecodeCode:
    """Tests for decode_code method."""

    @pytest.fixture
    def model(self):
        return TestableVQModel()

    def test_decode_code_4d(self, model):
        """Test decode_code with 4D input."""
        code = torch.randint(0, 512, (2, 4, 4, 4))

        output = model.decode_code(code)

        assert output.shape[0] == 2

    def test_decode_code_2d(self, model):
        """Test decode_code with 2D input."""
        code = torch.randint(0, 512, (2, 64))

        output = model.decode_code(code)

        assert output.shape[0] == 2

    def test_decode_code_1d(self, model):
        """Test decode_code with 1D input."""
        code = torch.randint(0, 512, (64,))

        output = model.decode_code(code)

        assert output.shape[0] == 1


class TestBaseVQModelCheckpoint:
    """Tests for checkpoint methods."""

    @pytest.fixture
    def model(self):
        return TestableVQModel()

    def test_save_and_load_checkpoint(self, model, tmp_path):
        """Test save and load checkpoint."""
        checkpoint_path = tmp_path / "test.pth"

        initial_weight = model.decoder.weight.data.clone()

        model.save_checkpoint(str(checkpoint_path))
        assert checkpoint_path.exists()

        with torch.no_grad():
            model.decoder.weight.fill_(0.0)

        model.load_checkpoint(str(checkpoint_path))

        assert torch.allclose(model.decoder.weight, initial_weight)


class TestBaseVQModelProperties:
    """Tests for properties."""

    def test_codebook_size(self):
        """Test codebook_size property."""
        model = TestableVQModel()
        assert model.codebook_size == 512

    def test_device(self):
        """Test device property."""
        model = TestableVQModel()
        assert isinstance(model.device, torch.device)
