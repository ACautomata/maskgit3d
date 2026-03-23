"""Vector Quantizer for VQ-VAE with EMA codebook updates."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with EMA codebook updates.

    Uses Exponential Moving Average (EMA) to update the codebook entries,
    which is more stable than pure gradient-based updates. Based on the
    VQ-VAE-2 paper (Razavi et al., 2019).

    Args:
        num_embeddings: Number of codebook entries
        embedding_dim: Dimension of each embedding vector
        commitment_cost: Weight for commitment loss (default: 0.25)
        decay: EMA decay rate for codebook updates (default: 0.99)
        epsilon: Epsilon for Laplace smoothing in EMA updates (default: 1e-5)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / embedding_dim**0.5, 1.0 / embedding_dim**0.5)
        # EMA handles codebook updates — no gradients needed for embedding weights
        self.embedding.weight.requires_grad = False

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_cluster_size: torch.Tensor
        self.register_buffer("_ema_embed_sum", self.embedding.weight.data.clone())
        self._ema_embed_sum: torch.Tensor

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize continuous latent vectors.

        Args:
            z: Input tensor of shape (B, C, D, H, W)

        Returns:
            z_q: Quantized tensor (B, C, D, H, W)
            vq_loss: Scalar tensor with commitment loss only (codebook updated via EMA)
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

        # One-hot encodings for EMA update
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # EMA codebook update (only during training)
        if self.training:
            # Update cluster sizes
            batch_cluster_size = encodings.sum(0).detach()
            self._ema_cluster_size = (
                self.decay * self._ema_cluster_size + (1 - self.decay) * batch_cluster_size
            )

            # Update embedding sums
            batch_embed_sum = z_flattened.detach().t() @ encodings.detach()  # (D, K)
            self._ema_embed_sum = (
                self.decay * self._ema_embed_sum + (1 - self.decay) * batch_embed_sum.t()
            )

            # Laplace smoothing to avoid division by zero for unused codes
            n = self._ema_cluster_size.sum()
            cluster_size = (
                (self._ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )

            # Update codebook embeddings
            self.embedding.weight.data = self._ema_embed_sum / cluster_size.unsqueeze(1)

        # Look up quantized vectors
        z_q = self.embedding(encoding_indices).view(z.shape)

        # Commitment loss only (codebook is updated via EMA, not gradients)
        vq_loss = self.commitment_cost * F.mse_loss(z, z_q.detach())

        # Straight-through estimator: z + (z_q - z).detach() allows gradients to flow
        z_q = z + (z_q - z).detach()

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
        z_q: torch.Tensor = self.embedding(indices)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
        return z_q
