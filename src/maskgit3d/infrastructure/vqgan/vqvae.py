"""
VQVAE Model Implementation.

This module provides a VQVAE model using MONAI's MaisiEncoder/MaisiDecoder
architecture with vector quantization, without KL regularization.
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
    MaisiDecoder,
    MaisiEncoder,
)

from maskgit3d.domain.interfaces import VQModelInterface
from maskgit3d.infrastructure.vqgan.quantize import VectorQuantizer2


class VQVAE(VQModelInterface):
    """
    VQVAE Model for volumetric medical images.

    This model combines:
    - MONAI's MaisiEncoder for feature extraction
    - VectorQuantizer2 for discrete latent codes
    - MONAI's MaisiDecoder for reconstruction

    Unlike the original AutoencoderKlMaisi, this model:
    - Removes KL regularization (no mu/sigma sampling)
    - Uses vector quantization for discrete latent space
    - Provides deterministic encoding
    """

    def __init__(
        self,
        in_channels: int = 1,
        codebook_size: int = 1024,
        embed_dim: int = 256,
        latent_channels: int = 4,
        num_channels: Sequence[int] = (64, 128, 256),
        num_res_blocks: Sequence[int] = (2, 2, 2),
        attention_levels: Sequence[bool] = (False, False, False),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_encoder_nonlocal_attn: bool = False,
        with_decoder_nonlocal_attn: bool = False,
        num_splits: int = 1,
        dim_split: int = 1,
        norm_float16: bool = False,
        use_flash_attention: bool = False,
        beta: float = 0.25,
    ):
        """
        Initialize VQVAE model.

        Args:
            in_channels: Number of input channels (1 for MRI/CT)
            codebook_size: Number of codebook entries
            embed_dim: Codebook embedding dimension
            latent_channels: Number of latent channels (encoder output)
            num_channels: Channel numbers for each level
            num_res_blocks: Number of residual blocks per level
            attention_levels: Whether to use attention at each level
            norm_num_groups: Number of groups for GroupNorm
            norm_eps: Epsilon for GroupNorm
            with_encoder_nonlocal_attn: Use non-local attention in encoder
            with_decoder_nonlocal_attn: Use non-local attention in decoder
            num_splits: Number of splits for memory-efficient processing
            dim_split: Dimension to split for memory optimization
            norm_float16: Use float16 for GroupNorm
            use_flash_attention: Use flash attention
            beta: Commitment loss weight for VQ
        """
        super().__init__()

        self.in_channels = in_channels
        self._codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.latent_channels = latent_channels
        self._num_levels = len(num_channels)
        self._cached_latent_spatial: tuple[int, int, int] | None = None

        # Maisi Encoder
        self.encoder = MaisiEncoder(
            spatial_dims=3,
            in_channels=in_channels,
            num_channels=num_channels,
            out_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_encoder_nonlocal_attn,
            num_splits=num_splits,
            dim_split=dim_split,
            norm_float16=norm_float16,
            use_flash_attention=use_flash_attention,
        )

        # Quantization layers
        self.quant_conv = nn.Conv3d(latent_channels, embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(embed_dim, latent_channels, 1)

        # Vector Quantizer
        self.quantize = VectorQuantizer2(
            n_embed=codebook_size,
            embed_dim=embed_dim,
            beta=beta,
        )

        # Maisi Decoder
        self.decoder = MaisiDecoder(
            spatial_dims=3,
            num_channels=num_channels,
            in_channels=latent_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
            num_splits=num_splits,
            dim_split=dim_split,
            norm_float16=norm_float16,
            use_flash_attention=use_flash_attention,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Encode volumes to quantized latent codes.

        Args:
            x: Input volumes [B, C, D, H, W]

        Returns:
            Tuple of (quantized, commitment_loss, info)
            - quantized: Quantized latent [B, embed_dim, D', H', W']
            - commitment_loss: Codebook commitment loss
            - info: Tuple of (perplexity, encodings, indices)
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)

        if self._cached_latent_spatial is None:
            self._cached_latent_spatial = tuple(quant.shape[2:])

        return quant, emb_loss, info

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latents to volumes.

        Args:
            quant: Quantized latents [B, embed_dim, D, H, W]

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
            code: Codebook indices, can be:
                - [B, D, H, W] - 4D spatial indices
                - [B, N] - 2D flattened indices
                - [N] - 1D flattened indices (single sample)

        Returns:
            Reconstructed volumes [B, C, D', H', W']
        """
        # Handle different input shapes
        if code.dim() == 4:  # [B, D, H, W]
            B, D, H, W = code.shape
            code_flat = code.view(B, -1)
            # Reshape to (B*N) for embedding lookup
            code_flat = code_flat.view(-1)  # Flatten to (B*D*H*W,)
        elif code.dim() == 2:  # [B, N]
            B = code.shape[0]
            code_flat = code.view(-1)  # Flatten to (B*N,)
            # Try to infer D, H, W from N
            N = code.shape[1]
            latent_res = round(N ** (1 / 3))
            D = H = W = latent_res
        elif code.dim() == 1:  # [N] - single sample, already flattened
            B = 1
            code_flat = code
            N = code.shape[0]
            latent_res = round(N ** (1 / 3))
            D = H = W = latent_res
        else:
            raise ValueError(f"Invalid code shape: {code.shape}. Expected 1D, 2D or 4D tensor.")

        # Get quantized latents from indices
        # VectorQuantizer2.get_codebook_entry expects shape=(B, D, H, W, C) and indices as (B*D*H*W,)
        quant_b = self.quantize.get_codebook_entry(code_flat, shape=(B, D, H, W, self.embed_dim))

        # Decode
        dec = self.decode(quant_b)
        return dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only reconstruction (ModelInterface contract).

        Args:
            x: Input volumes [B, C, D, H, W]

        Returns:
            Reconstructed volumes [B, C, D', H', W']
        """
        quant, diff, _ = self.encode(x)
        dec = self.decode(quant)
        return dec

    def forward_with_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning reconstruction and quantization loss.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstructed, quantization_loss)
        """
        quant, diff, _ = self.encode(x)
        dec = self.decode(quant)
        return dec, diff

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device

    @property
    def codebook_size(self) -> int:
        """Get the number of codes in the codebook."""
        return self._codebook_size

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        """
        Get the shape of latent representations (C, D, H, W).

        Returns cached shape after first encode(), or computes from num_levels.
        Spatial dims = input_size // 2^(num_levels - 1).
        """
        if self._cached_latent_spatial is not None:
            d, h, w = self._cached_latent_spatial
            return (self.latent_channels, d, h, w)
        return (self.latent_channels, -1, -1, -1)

    # Workaround for type checking: these methods exist on nn.Module
    # but have incompatible signatures with VQModelInterface
    def to(self, *args: Any, **kwargs: Any) -> "VQVAE":
        """Move model to device - delegates to nn.Module.to"""
        super().to(*args, **kwargs)  # type: ignore[return-value]
        return self

    def train(self, mode: bool = True) -> "VQVAE":
        """Set training mode - delegates to nn.Module.train"""
        super().train(mode)  # type: ignore[return-value]
        return self

    def eval(self) -> "VQVAE":
        """Set eval mode - delegates to nn.Module.eval"""
        super().eval()  # type: ignore[return-value]
        return self


def get_vqvae_config(
    image_size: int = 64,
    in_channels: int = 1,
    codebook_size: int = 1024,
    embed_dim: int = 256,
    latent_channels: int = 4,
    num_channels: Sequence[int] = (64, 128, 256),
    num_res_blocks: Sequence[int] = (2, 2, 2),
    attention_levels: Sequence[bool] = (False, False, False),
) -> dict:
    """
    Generate VQVAE configuration.

    Args:
        image_size: Input volume size (for reference)
        in_channels: Number of input channels
        codebook_size: Number of codebook entries
        embed_dim: Codebook embedding dimension
        latent_channels: Number of latent channels
        num_channels: Channel numbers for each level
        num_res_blocks: Number of residual blocks per level
        attention_levels: Whether to use attention at each level

    Returns:
        Configuration dictionary
    """
    return {
        "in_channels": in_channels,
        "codebook_size": codebook_size,
        "embed_dim": embed_dim,
        "latent_channels": latent_channels,
        "num_channels": num_channels,
        "num_res_blocks": num_res_blocks,
        "attention_levels": attention_levels,
        "norm_num_groups": 32,
        "norm_eps": 1e-6,
        "with_encoder_nonlocal_attn": False,
        "with_decoder_nonlocal_attn": False,
        "num_splits": 1,
        "dim_split": 1,
        "norm_float16": False,
        "use_flash_attention": False,
        "beta": 0.25,
    }


# Backward compatibility aliases
MaisiVQModel3D = VQVAE
get_maisi_vq_config = get_vqvae_config
