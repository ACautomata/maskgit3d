"""Vector Quantizer for VQ-VAE."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector Quantization layer for VQ-VAE.

    Args:
        num_embeddings: Number of codebook entries
        embedding_dim: Dimension of each embedding vector
        commitment_cost: Weight for commitment loss (default: 0.25)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize continuous latent vectors.

        Args:
            z: Input tensor of shape (B, C, D, H, W)

        Returns:
            z_q: Quantized tensor (B, C, D, H, W)
            vq_loss: Scalar tensor with VQ loss
            indices: Quantized indices (B, D, H, W)
        """
        # Reshape from (B, C, D, H, W) to (B, D, H, W, C) for quantization
        z = z.permute(0, 2, 3, 4, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        # Distance: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z^T e
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)

        # Straight-through estimator: z + (z_q - z).detach() allows gradients to flow
        z_q = z + (z_q - z).detach()

        # Compute loss
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Reshape indices
        indices = encoding_indices.view(z.shape[:-1])

        # Permute back to (B, C, D, H, W)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q, vq_loss, indices

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from codebook indices only.

        Args:
            indices: Codebook indices (B, D, H, W)

        Returns:
            z_q: Quantized latents (B, C, D, H, W)
        """
        z_q = self.embedding(indices)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
        return z_q
