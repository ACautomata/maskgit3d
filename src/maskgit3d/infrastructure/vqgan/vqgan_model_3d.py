"""
3D VQGAN Model Implementation.

This module provides the 3D version of VQGAN for volumetric medical images.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from maskgit3d.domain.interfaces import VQModelInterface
from maskgit3d.infrastructure.vqgan.encoder_decoder_3d import Encoder3d, Decoder3d
from maskgit3d.infrastructure.vqgan.quantize import VectorQuantizer2


class VQModel3D(nn.Module, VQModelInterface):
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
        channel_multipliers: Tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (8,),
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
        super().__init__()

        self.in_channels = in_channels
        self._codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.resolution = resolution

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
        self.quantize = VectorQuantizer2(
            n_embed=codebook_size,
            embed_dim=embed_dim,
            beta=0.25,
        )

        # Quantization convolutions (3D)
        self.quant_conv = nn.Conv3d(latent_channels // 2, embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(embed_dim, latent_channels // 2, 1)

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

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Encode volumes to quantized latent codes.

        Args:
            x: Input volumes [B, C, D, H, W]

        Returns:
            Tuple of (quantized, commitment_loss, info)
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
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

    def decode_code(self, code: torch.Tensor) -> torch.Tensor:
        """
        Decode from codebook indices directly.

        Args:
            code: Codebook indices [B, D, H, W]

        Returns:
            Reconstructed volumes [B, C, D', H', W']
        """
        # Handle different input shapes
        if code.dim() == 4:  # [B, D, H, W]
            B, D, H, W = code.shape
            code_flat = code.view(B, -1)
        else:  # [B, N]
            B = code.shape[0]
            code_flat = code
            # Try to infer D, H, W from latent shape
            N = code_flat.shape[1]
            latent_res = round(N ** (1 / 3))
            D = H = W = latent_res

        # Get quantized latents from indices
        quant_b = self.quantize.get_codebook_entry(code_flat, shape=(B, D, H, W, self.embed_dim))
        # Rearrange from DHWC to CDHW
        quant_b = quant_b.permute(0, 4, 1, 2, 3).contiguous()

        # Decode
        dec = self.decode(quant_b)
        return dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full encode-decode forward pass.

        Args:
            x: Input volumes [B, C, D, H, W]

        Returns:
            Tuple of (reconstructed_volumes, quantization_loss)
        """
        quant, diff, _ = self.encode(x)
        dec = self.decode(quant)
        return dec, diff

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        self.load_state_dict(torch.load(path, map_location="cpu"))

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device

    @property
    def codebook_size(self) -> int:
        """Get the number of codes in the codebook."""
        return self._codebook_size

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        """Get the shape of latent representations (C, D, H, W)."""
        # Infer from encoder output
        with torch.no_grad():
            dummy = torch.zeros(
                1, self.in_channels, self.resolution, self.resolution, self.resolution
            )
            h = self.encoder(dummy)
            _, _, info = self.quantize(h)
            indices = info[2]
            # Indices shape: [B, D', H', W']
            b, dd, hh, ww = indices.shape
        return (self.embed_dim, dd, hh, ww)
