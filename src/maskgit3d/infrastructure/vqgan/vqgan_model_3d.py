"""
3D VQGAN Model Implementation.

This module provides the 3D version of VQGAN for volumetric medical images.
"""

import torch
import torch.nn as nn

from maskgit3d.infrastructure.vqgan.base_vq_model import BaseVQModel
from maskgit3d.infrastructure.vqgan.encoder_decoder_3d import Decoder3d, Encoder3d
from maskgit3d.infrastructure.vqgan.quantize import VectorQuantizer


class VQModel3D(BaseVQModel):
    """
    3D VQGAN/VQVAE Model for volumetric medical images.

    This model learns discrete latent representations through:
    1. Encoding volumes to continuous latents
    2. Quantizing to codebook entries
    3. Decoding back to volumes
    """

    def __init__(
        self,
        in_channels: int = 1,
        codebook_size: int = 1024,
        embed_dim: int = 256,
        latent_channels: int = 256,
        resolution: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (8,),
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels: Number of input channels (1 for MRI/CT)
            codebook_size: Number of codebook entries
            embed_dim: Codebook embedding dimension
            latent_channels: Latent space channels
            resolution: Input volume resolution
            channel_multipliers: Channel multipliers for encoder/decoder
            num_res_blocks: Number of residual blocks
            attn_resolutions: Resolutions for attention
            dropout: Dropout probability
        """
        super().__init__(
            in_channels=in_channels,
            codebook_size=codebook_size,
            embed_dim=embed_dim,
            latent_channels=latent_channels,
        )

        self._resolution = resolution
        self._channel_multipliers = channel_multipliers
        self._num_res_blocks = num_res_blocks
        self._attn_resolutions = attn_resolutions
        self._dropout = dropout

        # Encoder
        self.encoder = Encoder3d(
            in_channels=in_channels,
            hidden_channels=latent_channels // 2,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
        )

        # Quantizer
        self.quantize = VectorQuantizer(
            n_embed=codebook_size,
            embed_dim=embed_dim,
            beta=0.25,
        )

        # Quantization convolutions (3D)
        self.quant_conv = nn.Conv3d(latent_channels // 2, embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(embed_dim, embed_dim, 1)

        # Decoder
        self.decoder = Decoder3d(
            z_channels=embed_dim,
            out_channels=in_channels,
            hidden_channels=latent_channels // 2,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
        )

        # Eagerly compute and cache latent shape from encoder output
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, resolution, resolution, resolution)
            h = self.encoder(dummy)
            h = self.quant_conv(h)
            # h shape: [B, embed_dim, D', H', W']
            _, _, dd, hh, ww = h.shape
            self._latent_shape: tuple[int, int, int, int] = (embed_dim, dd, hh, ww)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Encode volumes to quantized latent codes.

        Args:
            x: Input volumes [B, C, D, H, W]

        Returns:
            Tuple of (quantized, commitment_loss, info)
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)  # type: ignore[operator]
        return quant, emb_loss, info

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latents to volumes.

        Args:
            quant: Quantized latents [B, C, D, H, W]

        Returns:
            Reconstructed volumes [B, C, D', H', W']
        """
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        """Get the cached shape of latent representations (C, D, H, W)."""
        return self._latent_shape
