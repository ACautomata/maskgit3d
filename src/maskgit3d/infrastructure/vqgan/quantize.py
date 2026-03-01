"""
Vector Quantizer implementations for VQGAN.

This module provides different quantization strategies:
- VectorQuantizer: Standard VQ-VAE quantization
- VectorQuantizer2: Improved version with better performance
- EMAVectorQuantizer: EMA-based codebook updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from maskgit3d.domain.interfaces import QuantizerInterface


class VectorQuantizer(nn.Module, QuantizerInterface):
    """
    Standard VQ-VAE Quantizer.

    Maps continuous latents to discrete codebook entries by finding
    the nearest embedding in the codebook.
    """

    def __init__(
        self,
        n_embed: int,
        embed_dim: int,
        beta: float = 0.25,
        use_ema: bool = False,
        decay: float = 0.99,
        eps: float = 1e-5,
        remap: str | None = None,
        sane_index_shape: bool = False,
    ):
        """
        Args:
            n_embed: Number of embeddings in codebook
            embed_dim: Dimension of each embedding
            beta: Commitment cost weight
            use_ema: Use exponential moving average for codebook updates
            decay: EMA decay rate
            eps: EMA epsilon
            remap: Path to remapping file for index reduction
            sane_index_shape: Return indices as BHW format
        """
        super().__init__()
        self.n_e = n_embed
        self.e_dim = embed_dim
        self.beta = beta
        self.use_ema = use_ema
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            loaded = torch.load(self.remap)
            self.register_buffer("used", loaded.long())
            # pyright doesn't understand registered buffers - use type: ignore
            self.re_embed: int = int(self.used.shape[0])  # type: ignore[union-attr]
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices.")
        else:
            self.re_embed = n_embed

        self.sane_index_shape = sane_index_shape

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Quantize latent codes.

        Args:
            z: Latent from encoder [B, C, D, H, W]

        Returns:
            Tuple of (z_q, loss, info)
        """
        # Reshape: (B, C, D, H, W) -> (B, D, H, W, C)
        z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # Compute distances to codebook entries
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # Find nearest codebook entry
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # Compute commitment loss
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # Preserve gradients
        z_q = z + (z_q - z).detach()

        # Reshape back: (B, D, H, W, C) -> (B, C, D, H, W)
        z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()

        # Handle index shape
        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3], z_q.shape[4])

        # Compute perplexity
        with torch.no_grad():
            min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_e, device=z.device)
            min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)
            e_mean = torch.mean(min_encodings, dim=0)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(
        self,
        indices: torch.Tensor,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """
        Get quantized latents from indices.

        Args:
            indices: Codebook indices
            shape: Optional shape hint (B, D, H, W, C)

        Returns:
            Quantized latents
        """
        if shape is not None:
            b, d, h, w, c = shape
            indices = rearrange(indices, '(b d h w) -> b d h w', b=b, d=d, h=h, w=w)
            z_q = self.embedding(indices)
            z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()
        else:
            z_q = self.embedding(indices)

        return z_q


class VectorQuantizer2(nn.Module, QuantizerInterface):
    """
    Improved VectorQuantizer with better efficiency.

    Avoids costly matrix multiplications and supports index remapping.
    """

    def __init__(
        self,
        n_embed: int,
        embed_dim: int,
        beta: float = 0.25,
        remap: str | None = None,
        sane_index_shape: bool = False,
        legacy: bool = True,
    ):
        super().__init__()
        self.n_e = n_embed
        self.e_dim = embed_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            loaded = torch.load(self.remap)
            self.register_buffer("used", loaded.long())
            # pyright doesn't understand registered buffers - use type: ignore
            self.re_embed: int = int(self.used.shape[0])  # type: ignore[union-attr]
            print(f"Remapping {self.n_e} to {self.re_embed} indices.")
        else:
            self.re_embed = n_embed

        self.sane_index_shape = sane_index_shape

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Quantize latent codes."""
        z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened,
                         rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # Compute loss with correct beta term
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                torch.mean((z_q - z.detach()) ** 2)

        # Preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3], z_q.shape[4])

        return z_q, loss, (None, None, min_encoding_indices)

    def get_codebook_entry(
        self,
        indices: torch.Tensor,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Get quantized latents from indices."""
        if shape is not None:
            b, d, h, w, c = shape
            # Reshape indices to (B, D, H, W) first
            indices_view = indices.view(b, d, h, w)
            z_q = self.embedding(indices_view)  # (B, D, H, W, C)
            z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()
        else:
            z_q = self.embedding(indices)
        return z_q


class EMAVectorQuantizer(nn.Module, QuantizerInterface):
    """
    Vector Quantizer with EMA codebook updates.

    Uses exponential moving average for more stable codebook learning.
    """

    def __init__(
        self,
        n_embed: int,
        embed_dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        remap: str | None = None,
    ):
        super().__init__()
        self.n_e = n_embed
        self.e_dim = embed_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        # EMA codebook
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())

        self.remap = remap
        if self.remap is not None:
            loaded = torch.load(self.remap)
            self.register_buffer("used", loaded.long())
            # pyright doesn't understand registered buffers - use type: ignore
            self.re_embed: int = int(self.used.shape[0])  # type: ignore[union-attr]
        else:
            self.re_embed = n_embed

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Quantize with EMA updates."""
        z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        z_flattened = z.reshape(-1, self.e_dim)

        # Compute distances
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)

        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)

        # One-hot encodings for EMA
        encodings = F.one_hot(encoding_indices, self.n_e).type(z.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # EMA updates
        if self.training:
            # Get buffers - pyright needs explicit typing
            cs: torch.Tensor = self.cluster_size  # type: ignore[assignment]
            ea: torch.Tensor = self.embed_avg  # type: ignore[assignment]

            # Update cluster size using in-place operations
            cs.data.mul_(self.decay).add_(encodings.sum(0), alpha=1 - self.decay)

            # Update embed average
            embed_sum = encodings.transpose(0, 1) @ z_flattened
            ea.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # Update embeddings
            n = cs.sum()
            smoothed_cs = (cs + self.eps) / (n + self.n_e * self.eps) * n
            embed_normalized = ea / smoothed_cs.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        # Commitment loss
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        # Preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()

        return z_q, loss, (perplexity, encodings, encoding_indices)

    def get_codebook_entry(
        self,
        indices: torch.Tensor,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Get quantized latents from indices."""
        if shape is not None:
            b, d, h, w, c = shape
            indices = rearrange(indices, '(b d h w) -> b d h w', b=b, d=d, h=h, w=w)
            z_q = self.embedding(indices)
            z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()
        else:
            z_q = self.embedding(indices)
        return z_q
