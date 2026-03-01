"""Tests for vqgan/base_vq_model.py"""

import pytest
import torch
import torch.nn as nn
from maskgit3d.infrastructure.vqgan.base_vq_model import BaseVQModel


class DummyVQModel(BaseVQModel):
    """Dummy implementation for testing BaseVQModel."""

    def __init__(
        self,
        in_channels: int = 1,
        codebook_size: int = 64,
        embed_dim: int = 16,
        latent_channels: int = 16,
    ):
        super().__init__(in_channels, codebook_size, embed_dim, latent_channels)
        # Simple encoder/decoder for testing
        self.encoder_conv = nn.Conv3d(in_channels, latent_channels, 3, padding=1)
        self.decoder_conv = nn.Conv3d(latent_channels, in_channels, 3, padding=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Encode input to quantized representation."""
        quant = self.encoder_conv(x)
        loss = torch.tensor(0.1)
        return quant, loss, {}

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """Decode quantized representation to reconstruction."""
        return self.decoder_conv(quant)

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        """Return dummy latent shape."""
        return (self._latent_channels, 16, 16, 16)


class TestBaseVQModel:
    """Test BaseVQModel abstract class."""

    @pytest.fixture
    def model(self):
        """Create a dummy model for testing."""
        return DummyVQModel()

    def test_forward_returns_single_tensor(self, model):
        """Test that forward() returns only reconstruction tensor."""
        x = torch.randn(2, 1, 16, 16, 16)
        output = model.forward(x)

        # Should return a single tensor, not a tuple
        assert isinstance(output, torch.Tensor)
        assert output.shape == x.shape

    def test_forward_with_loss_returns_tuple(self, model):
        """Test that forward_with_loss() returns (reconstruction, loss)."""
        x = torch.randn(2, 1, 16, 16, 16)
        output, loss = model.forward_with_loss(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == x.shape
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar tensor

    def test_encode_method(self, model):
        """Test encode method returns correct structure."""
        x = torch.randn(2, 1, 16, 16, 16)
        quant, loss, info = model.encode(x)

        assert isinstance(quant, torch.Tensor)
        assert isinstance(loss, torch.Tensor)
        assert isinstance(info, dict)

    def test_decode_method(self, model):
        """Test decode method."""
        quant = torch.randn(2, 16, 16, 16, 16)
        output = model.decode(quant)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2  # Batch size preserved

    def test_codebook_size_property(self, model):
        """Test codebook_size property."""
        assert model.codebook_size == 64

    def test_device_property(self, model):
        """Test device property returns correct device."""
        device = model.device
        assert isinstance(device, torch.device)

    def test_save_and_load_checkpoint(self, model, tmp_path):
        """Test checkpoint saving and loading."""
        checkpoint_path = tmp_path / "test_checkpoint.pth"

        # Get original state
        original_weight = model.encoder_conv.weight.clone()

        # Save checkpoint
        model.save_checkpoint(str(checkpoint_path))
        assert checkpoint_path.exists()

        # Modify model weights
        with torch.no_grad():
            model.encoder_conv.weight.fill_(0.0)

        # Load checkpoint
        model.load_checkpoint(str(checkpoint_path))

        # Verify weights restored
        assert torch.allclose(model.encoder_conv.weight, original_weight)


class TestBaseVQModelCannotInstantiate:
    """Test that BaseVQModel cannot be instantiated directly."""

    def test_abstract_class_cannot_instantiate(self):
        """Test that BaseVQModel is abstract and requires implementation."""
        with pytest.raises(TypeError) as exc_info:
            BaseVQModel(
                in_channels=1,
                codebook_size=64,
                embed_dim=16,
                latent_channels=16,
            )
        assert (
            "abstract" in str(exc_info.value).lower()
            or "instantiate" in str(exc_info.value).lower()
        )
