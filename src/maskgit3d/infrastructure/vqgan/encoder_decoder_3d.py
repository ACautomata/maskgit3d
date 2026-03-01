"""
3D Encoder and Decoder architectures for VQGAN.

This module provides 3D versions of the encoder/decoder components
for volumetric medical images (MRI, CT scans).
"""
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def get_timestep_embedding_3d(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Build sinusoidal timestep embeddings for 3D inputs.

    Args:
        timesteps: Timestep values [B]
        embedding_dim: Dimension of the embedding

    Returns:
        Timestep embeddings [B, embedding_dim]
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """Swish activation function."""
    return x * torch.sigmoid(x)


def Normalize3d(in_channels: int) -> nn.GroupNorm:
    """Group normalization layer for 3D inputs."""
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample3d(nn.Module):
    """Upsampling layer for 3D inputs with optional convolution."""

    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="trilinear", align_corners=False
        )
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample3d(nn.Module):
    """Downsampling layer for 3D inputs with optional convolution."""

    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            pad = (0, 1, 0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class ResBlock3d(nn.Module):
    """Residual block for 3D inputs with optional timestep embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        temb_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize3d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttnBlock3d(nn.Module):
    """Self-attention block for 3D inputs."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize3d(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, d, h, w = q.shape
        q = q.reshape(b, c, d * h * w).transpose(1, 2)
        k = k.reshape(b, c, d * h * w).transpose(1, 2)
        v = v.reshape(b, c, d * h * w).transpose(1, 2)

        attn = torch.bmm(q, k.transpose(2, 1))
        attn = torch.softmax(attn, dim=-1)

        h_ = torch.bmm(attn, v)
        h_ = h_.transpose(1, 2).reshape(b, c, d, h, w)
        h_ = self.proj_out(h_)

        return x + h_


class Encoder3d(nn.Module):
    """
    3D CNN Encoder for VQGAN.

    Converts volumetric medical images to latent representations.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        resolution: int = 64,
        attn_resolutions: tuple[int, ...] = (8,),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels (1 for MRI/CT)
            hidden_channels: Base number of channels
            channel_multipliers: Channel multipliers for each resolution level
            num_res_blocks: Number of residual blocks per resolution
            resolution: Input volume size (assumes cubic)
            attn_resolutions: Resolutions to apply attention
            dropout: Dropout probability
            resamp_with_conv: Whether to use conv for up/down sampling
        """
        super().__init__()
        self.resolution = resolution
        self.hidden_channels = hidden_channels

        # Input convolution
        self.conv_in = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)

        # Downsampling blocks
        curr_res = resolution
        in_ch_mult = (1,) + tuple(channel_multipliers)
        self.down = nn.ModuleList()

        for i_level in range(len(channel_multipliers)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = hidden_channels * in_ch_mult[i_level]
            block_out = hidden_channels * channel_multipliers[i_level]

            for _ in range(num_res_blocks):
                block.append(ResBlock3d(block_in, block_out, dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3d(block_in))

            down_block = nn.Module()
            down_block.block = block
            down_block.attn = attn

            if i_level != len(channel_multipliers) - 1:
                down_block.downsample = Downsample3d(block_in, resamp_with_conv)
                curr_res = curr_res // 2

            self.down.append(down_block)

        # Middle
        self.mid_block_1 = ResBlock3d(block_in, block_in, dropout)
        self.mid_attn = AttnBlock3d(block_in)
        self.mid_block_2 = ResBlock3d(block_in, block_in, dropout)

        # Output
        self.norm_out = Normalize3d(block_in)
        self.conv_out = nn.Conv3d(block_in, hidden_channels, kernel_size=3, stride=1, padding=1)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to save memory."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode volumes to latent space.

        Args:
            x: Input volumes [B, C, D, H, W]

        Returns:
            Latent representations [B, z_channels, D//downsample, H//downsample, W//downsample]
        """
        if self.gradient_checkpointing:
            return self._forward_checkpoint(x)
        else:
            return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Actual forward implementation without gradient checkpointing."""
        # Input convolution
        h = self.conv_in(x)

        # Downsampling
        for i_level in range(len(self.down)):
            for block in self.down[i_level].block:
                h = block(h)

            if hasattr(self.down[i_level], 'downsample'):
                h = self.down[i_level].downsample(h)

            if len(self.down[i_level].attn) > 0:
                for attn in self.down[i_level].attn:
                    h = attn(h)

        # Middle
        h = self.mid_block_1(h)
        h = self.mid_attn(h)
        h = self.mid_block_2(h)

        # Output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h

    def _forward_checkpoint(self, x: torch.Tensor) -> torch.Tensor:
        """Forward implementation with gradient checkpointing."""
        # Input convolution
        h = checkpoint(self.conv_in, x, use_reentrant=False)

        # Downsampling - checkpoint each level
        for i_level in range(len(self.down)):
            for block in self.down[i_level].block:
                h = checkpoint(block, h, use_reentrant=False)

            if hasattr(self.down[i_level], 'downsample'):
                h = checkpoint(self.down[i_level].downsample, h, use_reentrant=False)

            if len(self.down[i_level].attn) > 0:
                for attn in self.down[i_level].attn:
                    h = checkpoint(attn, h, use_reentrant=False)

        # Middle - checkpoint each block
        h = checkpoint(self.mid_block_1, h, use_reentrant=False)
        h = checkpoint(self.mid_attn, h, use_reentrant=False)
        h = checkpoint(self.mid_block_2, h, use_reentrant=False)

        # Output
        h = checkpoint(self.norm_out, h, use_reentrant=False)
        h = nonlinearity(h)
        h = checkpoint(self.conv_out, h, use_reentrant=False)

        return h


class Decoder3d(nn.Module):
    """
    3D CNN Decoder for VQGAN.

    Converts quantized latents back to volumetric images.
    """

    def __init__(
        self,
        z_channels: int = 256,
        out_channels: int = 1,
        hidden_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        resolution: int = 64,
        attn_resolutions: tuple[int, ...] = (8,),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
    ):
        """
        Args:
            z_channels: Number of latent channels
            out_channels: Number of output channels
            hidden_channels: Base number of channels
            channel_multipliers: Channel multipliers for each resolution
            num_res_blocks: Number of residual blocks per resolution
            resolution: Output resolution
            attn_resolutions: Resolutions to apply attention
            dropout: Dropout probability
            resamp_with_conv: Whether to use conv for up/down sampling
        """
        super().__init__()
        self.resolution = resolution
        self.hidden_channels = hidden_channels

        # Compute channel sizes
        ch_mult = channel_multipliers
        block_in = hidden_channels * ch_mult[-1]

        # Input convolution
        self.conv_in = nn.Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle
        self.mid_block_1 = ResBlock3d(block_in, block_in, dropout)
        self.mid_attn = AttnBlock3d(block_in)
        self.mid_block_2 = ResBlock3d(block_in, block_in, dropout)

        # Upsampling blocks
        self.up = nn.ModuleList()
        for i_level in reversed(range(len(ch_mult))):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = hidden_channels * ch_mult[i_level]

            for _ in range(num_res_blocks + 1):
                block.append(ResBlock3d(block_in, block_out, dropout))
                block_in = block_out
                if resolution // (2 ** (len(ch_mult) - 1 - i_level)) in attn_resolutions:
                    attn.append(AttnBlock3d(block_in))

            up_block = nn.Module()
            up_block.block = block
            up_block.attn = attn

            if i_level != 0:
                up_block.upsample = Upsample3d(block_in, resamp_with_conv)

            self.up.insert(0, up_block)

        # Output
        self.norm_out = Normalize3d(block_in)
        self.conv_out = nn.Conv3d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to save memory."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to volumes.

        Args:
            z: Latent representations [B, z_channels, D, H, W]

        Returns:
            Reconstructed volumes [B, C, D', H', W']
        """
        if self.gradient_checkpointing:
            return self._forward_checkpoint(z)
        else:
            return self._forward_impl(z)

    def _forward_impl(self, z: torch.Tensor) -> torch.Tensor:
        """Actual forward implementation without gradient checkpointing."""
        # Input convolution
        h = self.conv_in(z)

        # Middle
        h = self.mid_block_1(h)
        h = self.mid_attn(h)
        h = self.mid_block_2(h)

        # Upsampling
        for i_level in reversed(range(len(self.up))):
            if hasattr(self.up[i_level], 'upsample'):
                h = self.up[i_level].upsample(h)

            for block in self.up[i_level].block:
                h = block(h)

            if len(self.up[i_level].attn) > 0:
                for attn in self.up[i_level].attn:
                    h = attn(h)

        # Output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h

    def _forward_checkpoint(self, z: torch.Tensor) -> torch.Tensor:
        """Forward implementation with gradient checkpointing."""
        # Input convolution
        h = checkpoint(self.conv_in, z, use_reentrant=False)

        # Middle
        h = checkpoint(self.mid_block_1, h, use_reentrant=False)
        h = checkpoint(self.mid_attn, h, use_reentrant=False)
        h = checkpoint(self.mid_block_2, h, use_reentrant=False)

        # Upsampling
        for i_level in reversed(range(len(self.up))):
            if hasattr(self.up[i_level], 'upsample'):
                h = checkpoint(self.up[i_level].upsample, h, use_reentrant=False)

            for block in self.up[i_level].block:
                h = checkpoint(block, h, use_reentrant=False)

            if len(self.up[i_level].attn) > 0:
                for attn in self.up[i_level].attn:
                    h = checkpoint(attn, h, use_reentrant=False)

        # Output
        h = checkpoint(self.norm_out, h, use_reentrant=False)
        h = nonlinearity(h)
        h = checkpoint(self.conv_out, h, use_reentrant=False)

        return h


def get_encoder_decoder_config_3d(
    volume_size: int = 64,
    in_channels: int = 1,
    out_channels: int = 1,
    latent_channels: int = 256,
    num_res_blocks: int = 2,
    attn_resolutions: tuple[int, ...] = (8,),
    channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4),
) -> dict:
    """
    Generate 3D encoder/decoder configuration.

    Args:
        volume_size: Input volume size (assumes cubic)
        in_channels: Number of input channels
        out_channels: Number of output channels
        latent_channels: Number of latent channels
        num_res_blocks: Number of residual blocks
        attn_resolutions: Attention resolutions
        channel_multipliers: Channel multipliers

    Returns:
        Configuration dictionary
    """
    return {
        "ddconfig": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "hidden_channels": latent_channels // 2,
            "channel_multipliers": channel_multipliers,
            "num_res_blocks": num_res_blocks,
            "resolution": volume_size,
            "attn_resolutions": attn_resolutions,
            "dropout": 0.0,
            "resamp_with_conv": True,
        },
        "z_channels": latent_channels,
    }
