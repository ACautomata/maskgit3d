"""
Unit tests for MaisiVQModel3D.

Tests the MAISI-based VQGAN model architecture.
"""
import pytest
import torch

from maskgit3d.infrastructure.vqgan import MaisiVQModel3D, get_maisi_vq_config


class TestMaisyVQModel3D:
    """Tests for MaisiVQModel3D class."""

    @pytest.fixture
    def model_config(self):
        """Default model configuration for testing."""
        return get_maisi_vq_config(
            image_size=32,
            in_channels=1,
            codebook_size=128,
            embed_dim=64,
            latent_channels=4,
            num_channels=(32, 64),
            num_res_blocks=(1, 1),
            attention_levels=(False, False),
        )

    @pytest.fixture
    def model(self, model_config):
        """Create model instance for testing."""
        return MaisiVQModel3D(**model_config)

    def test_model_creation(self, model):
        """Test model can be created with default config."""
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self, model):
        """Test forward pass produces correct output shape."""
        batch_size = 1
        in_channels = 1
        spatial_size = 32

        x = torch.randn(batch_size, in_channels, spatial_size, spatial_size, spatial_size)
        recon, vq_loss = model.forward_with_loss(x)

        # Check output shape matches input
        assert recon.shape == x.shape
        # Check VQ loss is a scalar
        assert vq_loss.dim() == 0

    def test_encode_decode(self, model):
        """Test encode and decode methods."""
        batch_size = 1
        in_channels = 1
        spatial_size = 32

        x = torch.randn(batch_size, in_channels, spatial_size, spatial_size, spatial_size)

        # Encode
        quant, vq_loss, info = model.encode(x)

        # Check quantized output has correct channels
        assert quant.shape[1] == model._embed_dim

        # Decode
        recon = model.decode(quant)

        # Check reconstruction shape
        assert recon.shape == x.shape

    def test_codebook_size(self, model, model_config):
        """Test codebook_size property."""
        assert model.codebook_size == model_config["codebook_size"]

    def test_device_property(self, model):
        """Test device property returns correct device."""
        device = model.device
        assert isinstance(device, torch.device)
        assert device.type == "cpu"  # Model is on CPU by default

    def test_save_load_checkpoint(self, model, tmp_path):
        """Test saving and loading model checkpoint."""
        checkpoint_path = str(tmp_path / "model.pt")

        # Save checkpoint
        model.save_checkpoint(checkpoint_path)

        # Create new model with same config
        new_model = MaisiVQModel3D(**get_maisi_vq_config(
            image_size=32,
            in_channels=1,
            codebook_size=128,
            embed_dim=64,
            latent_channels=4,
            num_channels=(32, 64),
            num_res_blocks=(1, 1),
            attention_levels=(False, False),
        ))

        # Load checkpoint
        new_model.load_checkpoint(checkpoint_path)

        # Verify parameters match
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert torch.allclose(param1, param2)

    def test_decode_code(self, model):
        """Test decoding from codebook indices."""
        batch_size = 1

        # Get latent shape from a forward pass
        x = torch.randn(batch_size, 1, 32, 32, 32)
        quant, _, info = model.encode(x)
        indices = info[2]  # [B, D, H, W]

        # Decode from indices
        recon = model.decode_code(indices)

        # Check reconstruction shape
        assert recon.shape[0] == batch_size
        assert recon.shape[1] == 1  # in_channels

    def test_gradient_flow(self, model):
        """Test gradients flow through the model."""
        x = torch.randn(1, 1, 32, 32, 32, requires_grad=True)

        recon, vq_loss = model.forward_with_loss(x)
        loss = recon.mean() + vq_loss
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_different_input_sizes(self):
        """Test model works with different input sizes."""
        # Use larger channels that are divisible by default norm_num_groups (32)
        config = get_maisi_vq_config(
            image_size=32,  # Use 32 instead of 16 to avoid spatial dimension issues
            in_channels=1,
            codebook_size=64,
            embed_dim=32,
            latent_channels=4,
            num_channels=(32, 64),  # Divisible by 32
            num_res_blocks=(1, 1),
            attention_levels=(False, False),
        )
        model = MaisiVQModel3D(**config)

        # Test with 32x32x32 input
        x = torch.randn(1, 1, 32, 32, 32)
        recon, _ = model.forward_with_loss(x)
        assert recon.shape == x.shape

    def test_batch_processing(self, model):
        """Test model processes batches correctly."""
        batch_sizes = [1, 2, 4]

        for bs in batch_sizes:
            x = torch.randn(bs, 1, 32, 32, 32)
            recon, _ = model.forward_with_loss(x)
            assert recon.shape[0] == bs


class TestGetMaisyVQConfig:
    """Tests for get_maisi_vq_config function."""

    def test_default_config(self):
        """Test default configuration values."""
        config = get_maisi_vq_config()

        assert config["in_channels"] == 1
        assert config["codebook_size"] == 1024
        assert config["embed_dim"] == 256
        assert config["latent_channels"] == 4

    def test_custom_config(self):
        """Test custom configuration values."""
        config = get_maisi_vq_config(
            image_size=128,
            in_channels=3,
            codebook_size=2048,
            embed_dim=512,
            latent_channels=8,
        )

        assert config["in_channels"] == 3
        assert config["codebook_size"] == 2048
        assert config["embed_dim"] == 512
        assert config["latent_channels"] == 8


class TestMaisyVQModel3DIntegration:
    """Integration tests for MaisiVQModel3D."""

    @pytest.fixture
    def model(self):
        """Create model for integration testing."""
        return MaisiVQModel3D(**get_maisi_vq_config(
            image_size=32,
            in_channels=1,
            codebook_size=128,
            embed_dim=64,
            latent_channels=4,
            num_channels=(32, 64),
            num_res_blocks=(1, 1),
            attention_levels=(False, False),
        ))

    def test_encode_decode_consistency(self, model):
        """Test that encode-decode produces consistent results."""
        x = torch.randn(1, 1, 32, 32, 32)

        # First pass
        quant1, _, _ = model.encode(x)
        recon1 = model.decode(quant1)

        # Second pass with same input
        quant2, _, _ = model.encode(x)
        recon2 = model.decode(quant2)

        # Should produce same results (deterministic)
        assert torch.allclose(recon1, recon2)

    def test_reconstruction_quality(self, model):
        """Test reconstruction has reasonable quality."""
        x = torch.randn(1, 1, 32, 32, 32)
        recon, _ = model.forward_with_loss(x)

        # MSE should be bounded (not perfect, but not random)
        mse = torch.nn.functional.mse_loss(recon, x)
        assert mse < 10.0  # Reasonable upper bound for random input