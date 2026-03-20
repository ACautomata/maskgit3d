from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .fsq import FSQQuantizer
from .protocol import QuantizerProtocol
from .quantizer import VectorQuantizer


class VQVAE(nn.Module):
    """Vector Quantized Variational Autoencoder for 3D medical images.

    Implements a VQ-VAE or FSQ-VAE architecture for encoding 3D medical images into
    discrete latent tokens and decoding them back to reconstructed images.

    Supports two quantizer types:
    - VQ (Vector Quantization): Traditional learned codebook
    - FSQ (Finite Scalar Quantization): No learned parameters, 100% codebook utilization

    The model consists of:
    - Encoder: Converts input images to latent representations
    - Quantizer: Maps latents to discrete embedding indices (VQ or FSQ)
    - Decoder: Reconstructs images from quantized latents

    Args:
        in_channels: Number of input channels (default: 1 for grayscale).
        out_channels: Number of output channels (default: 1 for grayscale).
        latent_channels: Number of channels in the latent space (default: 256).
        num_embeddings: Number of discrete embeddings in the codebook (default: 8192).
            Only used when quantizer_type="vq".
        embedding_dim: Dimension of each embedding vector (default: 256).
        num_channels: Tuple of channel numbers for each encoder/decoder block.
            Default: (64, 128, 256).
        num_res_blocks: Number of residual blocks at each resolution level.
            Default: (2, 2, 2).
        attention_levels: Boolean tuple indicating which levels use attention.
            Default: (False, False, False).
        commitment_cost: Weight for commitment loss in VQ training (default: 0.25).
            Only used when quantizer_type="vq".
        quantizer_type: Type of quantizer to use: "vq" or "fsq" (default: "vq").
        fsq_levels: List of quantization levels per dimension for FSQ.
            Only used when quantizer_type="fsq".
            Default: [8, 8, 8, 5, 5, 5] -> 64,000 codes.

    Attributes:
        encoder: Encoder network that processes input images.
        quant_conv: Conv3d layer that transforms encoder output to embedding dimension.
        post_quant_conv: Conv3d layer that transforms embedding back to latent channels.
        quantizer: VectorQuantizer or FSQQuantizer layer for discretizing latents.
        decoder: Decoder network that reconstructs images from quantized latents.

    Example:
        >>> # VQ mode (default)
        >>> model = VQVAE(num_embeddings=1024)
        >>> x = torch.randn(1, 1, 32, 32, 32)
        >>> x_recon, vq_loss = model(x)

        >>> # FSQ mode
        >>> model = VQVAE(quantizer_type="fsq", fsq_levels=[8, 8, 8, 5, 5, 5])
        >>> x_recon, vq_loss = model(x)  # vq_loss will be 0.0
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 256,
        num_embeddings: int = 8192,
        embedding_dim: int = 256,
        num_channels: Sequence[int] = (64, 128, 256),
        num_res_blocks: Sequence[int] = (2, 2, 2),
        attention_levels: Sequence[bool] = (False, False, False),
        commitment_cost: float = 0.25,
        quantizer_type: Literal["vq", "fsq"] = "vq",
        fsq_levels: Sequence[int] = (8, 8, 8, 5, 5, 5),
        num_splits: int = 1,
        dim_split: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_splits = num_splits
        self.dim_split = dim_split

        self.encoder = Encoder(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=latent_channels,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
            num_splits=num_splits,
            dim_split=dim_split,
        )

        self.quant_conv = nn.Conv3d(latent_channels, embedding_dim, 1)
        self.post_quant_conv = nn.Conv3d(embedding_dim, latent_channels, 1)

        if quantizer_type == "vq":
            self.quantizer: QuantizerProtocol = VectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
            )
        elif quantizer_type == "fsq":
            self.quantizer: QuantizerProtocol = FSQQuantizer(
                levels=list(fsq_levels),
                embedding_dim=embedding_dim,
            )
        else:
            raise ValueError(f"quantizer_type must be 'vq' or 'fsq', got '{quantizer_type}'")

        self.decoder = Decoder(
            spatial_dims=3,
            in_channels=latent_channels,
            out_channels=out_channels,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
            num_splits=num_splits,
            dim_split=dim_split,
        )

        self.use_gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self.use_gradient_checkpointing = True
        self.encoder.use_gradient_checkpointing = True
        self.decoder.use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.use_gradient_checkpointing = False
        self.encoder.use_gradient_checkpointing = False
        self.decoder.use_gradient_checkpointing = False

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z_e = self.quant_conv(h)
        z_q, vq_loss, indices = self.quantizer(z_e)
        return z_q, vq_loss, indices, z_e

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        z_q = self.post_quant_conv(z_q)
        out: torch.Tensor = self.decoder(z_q)
        return out

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.decode_from_indices(indices)
        return self.decode(z_q)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_q, vq_loss, _, _ = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, vq_loss
