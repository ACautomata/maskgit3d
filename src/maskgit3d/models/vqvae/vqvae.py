from collections.abc import Sequence

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .quantizer import VectorQuantizer


class VQVAE(nn.Module):
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
    ):
        super().__init__()

        self.encoder = Encoder(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=latent_channels,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
        )

        self.quant_conv = nn.Conv3d(latent_channels, embedding_dim, 1)
        self.post_quant_conv = nn.Conv3d(embedding_dim, latent_channels, 1)

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

        self.decoder = Decoder(
            spatial_dims=3,
            in_channels=latent_channels,
            out_channels=out_channels,
            num_channels=num_channels[::-1],
            num_res_blocks=num_res_blocks[::-1],
            attention_levels=attention_levels[::-1],
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = self.quant_conv(h)
        z_q, vq_loss, indices = self.quantizer(h)
        return z_q, vq_loss, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        z_q = self.post_quant_conv(z_q)
        out: torch.Tensor = self.decoder(z_q)
        return out

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.decode_from_indices(indices)
        return self.decode(z_q)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_q, vq_loss, _ = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, vq_loss
