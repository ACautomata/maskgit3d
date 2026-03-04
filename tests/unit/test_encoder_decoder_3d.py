"""Tests for encoder_decoder_3d module to improve coverage."""

import torch
import torch.nn as nn

from maskgit3d.infrastructure.vqgan.encoder_decoder_3d import (
    AttnBlock3d,
    Decoder3d,
    Downsample3d,
    Encoder3d,
    Normalize3d,
    ResBlock3d,
    Upsample3d,
    get_encoder_decoder_config_3d,
    get_timestep_embedding_3d,
    nonlinearity,
)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_timestep_embedding_3d(self):
        """Test timestep embedding generation."""
        timesteps = torch.tensor([0, 1, 2])
        embedding_dim = 64
        result = get_timestep_embedding_3d(timesteps, embedding_dim)

        assert result.shape == (3, embedding_dim)
        assert not torch.isnan(result).any()

    def test_get_timestep_embedding_3d_odd_dim(self):
        """Test timestep embedding with odd dimension (padded)."""
        timesteps = torch.tensor([0, 1])
        embedding_dim = 65  # Odd dimension
        result = get_timestep_embedding_3d(timesteps, embedding_dim)

        assert result.shape == (2, embedding_dim)
        assert not torch.isnan(result).any()

    def test_nonlinearity(self):
        """Test swish activation."""
        x = torch.randn(2, 3, 4, 4, 4)
        result = nonlinearity(x)

        assert result.shape == x.shape
        # Swish: x * sigmoid(x)
        expected = x * torch.sigmoid(x)
        assert torch.allclose(result, expected)

    def test_normalize3d(self):
        """Test 3D group normalization."""
        norm = Normalize3d(in_channels=32)
        x = torch.randn(2, 32, 8, 8, 8)
        result = norm(x)

        assert result.shape == x.shape
        assert isinstance(norm, nn.GroupNorm)
        assert norm.num_groups == 32


class TestUpsample3d:
    """Tests for Upsample3d module."""

    def test_init_with_conv(self):
        """Test initialization with convolution."""
        up = Upsample3d(in_channels=64, with_conv=True)

        assert up.with_conv is True
        assert hasattr(up, "conv")
        assert up.conv.in_channels == 64
        assert up.conv.out_channels == 64

    def test_init_without_conv(self):
        """Test initialization without convolution."""
        up = Upsample3d(in_channels=64, with_conv=False)

        assert up.with_conv is False
        assert not hasattr(up, "conv")

    def test_forward_with_conv(self):
        """Test forward pass with convolution."""
        up = Upsample3d(in_channels=32, with_conv=True)
        x = torch.randn(1, 32, 4, 4, 4)
        result = up(x)

        # Should upsample by 2x
        assert result.shape == (1, 32, 8, 8, 8)

    def test_forward_without_conv(self):
        """Test forward pass without convolution."""
        up = Upsample3d(in_channels=32, with_conv=False)
        x = torch.randn(1, 32, 4, 4, 4)
        result = up(x)

        assert result.shape == (1, 32, 8, 8, 8)


class TestDownsample3d:
    """Tests for Downsample3d module."""

    def test_init_with_conv(self):
        """Test initialization with convolution."""
        down = Downsample3d(in_channels=64, with_conv=True)

        assert down.with_conv is True
        assert hasattr(down, "conv")

    def test_init_without_conv(self):
        """Test initialization without convolution."""
        down = Downsample3d(in_channels=64, with_conv=False)

        assert down.with_conv is False
        assert not hasattr(down, "conv")

    def test_forward_with_conv(self):
        """Test forward pass with convolution."""
        down = Downsample3d(in_channels=32, with_conv=True)
        x = torch.randn(1, 32, 8, 8, 8)
        result = down(x)

        # Should downsample by 2x
        assert result.shape == (1, 32, 4, 4, 4)

    def test_forward_without_conv(self):
        """Test forward pass without convolution (avg pool)."""
        down = Downsample3d(in_channels=32, with_conv=False)
        x = torch.randn(1, 32, 8, 8, 8)
        result = down(x)

        assert result.shape == (1, 32, 4, 4, 4)


class TestResBlock3d:
    """Tests for ResBlock3d module."""

    def test_init_same_channels(self):
        """Test initialization with same input/output channels."""
        block = ResBlock3d(in_channels=32, out_channels=32, dropout=0.1)

        assert block.in_channels == 32
        assert block.out_channels == 32
        assert isinstance(block.shortcut, nn.Identity)

    def test_init_different_channels(self):
        """Test initialization with different input/output channels."""
        block = ResBlock3d(in_channels=32, out_channels=64, dropout=0.1)

        assert block.in_channels == 32
        assert block.out_channels == 64
        assert isinstance(block.shortcut, nn.Conv3d)

    def test_init_none_out_channels(self):
        """Test initialization with None output channels (same as input)."""
        block = ResBlock3d(in_channels=32, out_channels=None, dropout=0.1)

        assert block.out_channels == 32

    def test_init_with_temb(self):
        """Test initialization with timestep embedding."""
        block = ResBlock3d(in_channels=32, temb_channels=64)

        assert hasattr(block, "temb_proj")
        assert block.temb_proj.in_features == 64
        assert block.temb_proj.out_features == 32

    def test_forward_without_temb(self):
        """Test forward pass without timestep embedding."""
        block = ResBlock3d(in_channels=32, out_channels=32)
        x = torch.randn(1, 32, 4, 4, 4)
        result = block(x)

        assert result.shape == (1, 32, 4, 4, 4)

    def test_forward_with_temb(self):
        """Test forward pass with timestep embedding."""
        block = ResBlock3d(in_channels=32, temb_channels=64)
        x = torch.randn(1, 32, 4, 4, 4)
        temb = torch.randn(1, 64)
        result = block(x, temb)

        assert result.shape == (1, 32, 4, 4, 4)

    def test_forward_different_channels(self):
        """Test forward pass with different input/output channels."""
        block = ResBlock3d(in_channels=32, out_channels=64)
        x = torch.randn(1, 32, 4, 4, 4)
        result = block(x)

        assert result.shape == (1, 64, 4, 4, 4)


class TestAttnBlock3d:
    """Tests for AttnBlock3d module."""

    def test_init(self):
        """Test initialization."""
        attn = AttnBlock3d(in_channels=32)

        assert attn.in_channels == 32
        assert isinstance(attn.norm, nn.GroupNorm)
        assert isinstance(attn.q, nn.Conv3d)
        assert isinstance(attn.k, nn.Conv3d)
        assert isinstance(attn.v, nn.Conv3d)
        assert isinstance(attn.proj_out, nn.Conv3d)

    def test_forward(self):
        """Test forward pass."""
        attn = AttnBlock3d(in_channels=32)
        x = torch.randn(1, 32, 4, 4, 4)
        result = attn(x)

        assert result.shape == (1, 32, 4, 4, 4)
        # Output should be different from input (residual added)
        assert not torch.equal(result, x)


class TestEncoder3d:
    """Tests for Encoder3d module."""

    def test_init(self):
        """Test initialization."""
        encoder = Encoder3d(
            in_channels=1,
            hidden_channels=32,
            channel_multipliers=(1, 2, 4),
            num_res_blocks=1,
            resolution=16,
            attn_resolutions=(4,),
        )

        assert encoder.resolution == 16
        assert encoder.hidden_channels == 32
        assert encoder.gradient_checkpointing is False

    def test_enable_gradient_checkpointing(self):
        """Test gradient checkpointing enable."""
        encoder = Encoder3d(in_channels=1, hidden_channels=32)
        encoder.enable_gradient_checkpointing()

        assert encoder.gradient_checkpointing is True

    def test_disable_gradient_checkpointing(self):
        """Test gradient checkpointing disable."""
        encoder = Encoder3d(in_channels=1, hidden_channels=32)
        encoder.enable_gradient_checkpointing()
        encoder.disable_gradient_checkpointing()

        assert encoder.gradient_checkpointing is False

    def test_forward(self):
        """Test forward pass without gradient checkpointing."""
        encoder = Encoder3d(
            in_channels=1,
            hidden_channels=32,
            channel_multipliers=(1, 2),
            num_res_blocks=1,
            resolution=8,
        )
        x = torch.randn(1, 1, 8, 8, 8)
        result = encoder(x)

        # Should downsample by len(channel_multipliers)
        assert result.shape[0] == 1
        assert result.shape[2] < 8  # Downsampled

    def test_forward_with_checkpointing(self):
        """Test forward pass with gradient checkpointing."""
        encoder = Encoder3d(
            in_channels=1,
            hidden_channels=32,
            channel_multipliers=(1, 2),
            num_res_blocks=1,
            resolution=8,
        )
        encoder.enable_gradient_checkpointing()
        x = torch.randn(1, 1, 8, 8, 8)
        result = encoder(x)

        assert result.shape[0] == 1

    def test_forward_with_attention(self):
        """Test forward pass with attention blocks."""
        encoder = Encoder3d(
            in_channels=1,
            hidden_channels=32,
            channel_multipliers=(1, 2),
            num_res_blocks=1,
            resolution=8,
            attn_resolutions=(4,),  # Apply attention at resolution 4
        )
        x = torch.randn(1, 1, 8, 8, 8)
        result = encoder(x)

        assert result.shape[0] == 1


class TestDecoder3d:
    """Tests for Decoder3d module."""

    def test_init(self):
        """Test initialization."""
        decoder = Decoder3d(
            z_channels=128,
            out_channels=1,
            hidden_channels=32,
            channel_multipliers=(1, 2, 4),
            num_res_blocks=1,
            resolution=8,
        )

        assert decoder.resolution == 8
        assert decoder.hidden_channels == 32
        assert decoder.gradient_checkpointing is False

    def test_enable_gradient_checkpointing(self):
        """Test gradient checkpointing enable."""
        decoder = Decoder3d(z_channels=128, out_channels=1)
        decoder.enable_gradient_checkpointing()

        assert decoder.gradient_checkpointing is True

    def test_disable_gradient_checkpointing(self):
        """Test gradient checkpointing disable."""
        decoder = Decoder3d(z_channels=128, out_channels=1)
        decoder.enable_gradient_checkpointing()
        decoder.disable_gradient_checkpointing()

        assert decoder.gradient_checkpointing is False

    def test_forward(self):
        """Test forward pass without gradient checkpointing."""
        decoder = Decoder3d(
            z_channels=64,
            out_channels=1,
            hidden_channels=32,
            channel_multipliers=(1, 2),
            num_res_blocks=1,
            resolution=8,
        )
        # Input latent
        z = torch.randn(1, 64, 2, 2, 2)
        result = decoder(z)

        assert result.shape[0] == 1
        assert result.shape[1] == 1  # out_channels

    def test_forward_with_checkpointing(self):
        """Test forward pass with gradient checkpointing."""
        decoder = Decoder3d(
            z_channels=64,
            out_channels=1,
            hidden_channels=32,
            channel_multipliers=(1, 2),
            num_res_blocks=1,
            resolution=8,
        )
        decoder.enable_gradient_checkpointing()
        z = torch.randn(1, 64, 2, 2, 2)
        result = decoder(z)

        assert result.shape[0] == 1

    def test_forward_with_attention(self):
        """Test forward pass with attention blocks."""
        decoder = Decoder3d(
            z_channels=64,
            out_channels=1,
            hidden_channels=32,
            channel_multipliers=(1, 2),
            num_res_blocks=1,
            resolution=8,
            attn_resolutions=(2,),
        )
        z = torch.randn(1, 64, 2, 2, 2)
        result = decoder(z)

        assert result.shape[0] == 1


class TestGetEncoderDecoderConfig3d:
    """Tests for get_encoder_decoder_config_3d function."""

    def test_default_config(self):
        """Test default configuration generation."""
        config = get_encoder_decoder_config_3d()

        assert "ddconfig" in config
        assert "z_channels" in config
        assert config["z_channels"] == 256
        assert config["ddconfig"]["in_channels"] == 1
        assert config["ddconfig"]["out_channels"] == 1

    def test_custom_config(self):
        """Test custom configuration generation."""
        config = get_encoder_decoder_config_3d(
            volume_size=128,
            in_channels=3,
            out_channels=3,
            latent_channels=512,
            num_res_blocks=3,
            attn_resolutions=(16, 8),
            channel_multipliers=(1, 2, 4, 8),
        )

        assert config["ddconfig"]["resolution"] == 128
        assert config["ddconfig"]["in_channels"] == 3
        assert config["ddconfig"]["out_channels"] == 3
        assert config["z_channels"] == 512
