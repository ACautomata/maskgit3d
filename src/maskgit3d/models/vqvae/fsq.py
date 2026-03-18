"""Finite Scalar Quantization (FSQ) for 3D VQVAE.

Based on: Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple", ICLR 2024
Reference: https://arxiv.org/abs/2309.15505
Code adapted from: https://github.com/duchenzhuang/FSQ-pytorch
"""

import math

import torch
import torch.nn as nn


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight-through estimator for gradient flow."""
    return z + (z.round() - z).detach()


class FSQ(nn.Module):
    """Finite Scalar Quantization module.

    Quantizes each dimension independently to a finite set of levels.
    No learned parameters - pure deterministic quantization.
    """

    def __init__(self, levels: list[int]):
        super().__init__()
        levels_tensor = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", levels_tensor)

        basis_tensor = torch.cumprod(torch.tensor([1, *levels[:-1]], dtype=torch.int32), dim=0)
        self.register_buffer("_basis", basis_tensor)

        self.dim = len(levels)
        self.n_codes = math.prod(levels)

        self.register_buffer(
            "implicit_codebook",
            self.indices_to_codes(torch.arange(self.n_codes)),
        )

    @property
    def levels_tensor(self) -> torch.Tensor:
        levels = self._buffers.get("_levels")
        if not isinstance(levels, torch.Tensor):
            raise RuntimeError("FSQ levels buffer is not initialized")
        return levels

    @property
    def basis_tensor(self) -> torch.Tensor:
        basis = self._buffers.get("_basis")
        if not isinstance(basis, torch.Tensor):
            raise RuntimeError("FSQ basis buffer is not initialized")
        return basis

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize continuous values to discrete levels."""
        levels = self.levels_tensor.float()
        half_l = (levels - 1) / 2
        offset = torch.where(
            levels.remainder(2) == 0, torch.full_like(levels, 0.5), torch.zeros_like(levels)
        )
        shift = half_l - offset

        z_bounded = z.tanh() * (half_l - 1e-3 * half_l) + offset
        z_quantized = round_ste(z_bounded) - offset

        return z_quantized / shift

    def codes_to_indices(self, z_q: torch.Tensor) -> torch.Tensor:
        """Convert quantized codes to flat indices."""
        levels = self.levels_tensor.float()
        half_l = (levels - 1) / 2
        offset = torch.where(
            levels.remainder(2) == 0, torch.full_like(levels, 0.5), torch.zeros_like(levels)
        )
        shift = half_l - offset

        codes = (z_q * shift + offset).round().long()
        return (codes * self.basis_tensor.float()).sum(dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert flat indices back to quantized codes."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self.basis_tensor) % self.levels_tensor

        levels = self.levels_tensor.float()
        half_l = (levels - 1) / 2
        offset = torch.where(
            levels.remainder(2) == 0, torch.full_like(levels, 0.5), torch.zeros_like(levels)
        )
        shift = half_l - offset

        return (codes_non_centered.float() - offset) / shift

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize input tensor."""
        assert z.shape[-1] == self.dim, f"Expected last dim {self.dim}, got {z.shape[-1]}"

        z_q = self.quantize(z)
        indices = self.codes_to_indices(z_q)

        return z_q, indices


class FSQQuantizer(nn.Module):
    """FSQ wrapper matching VectorQuantizer's interface exactly.

    Drop-in replacement for VectorQuantizer with:
    - Same input/output shapes: (B, C, D, H, W)
    - Same method signatures: forward(), decode_from_indices()
    - Same property: num_embeddings

    Key difference: FSQ has NO VQ loss (returns tensor(0.0))
    """

    def __init__(self, levels: list[int], embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.levels = list(levels)
        fsq_dim = len(levels)

        self.fsq = FSQ(levels=levels)

        self.project_in = nn.Linear(embedding_dim, fsq_dim)
        self.project_out = nn.Linear(fsq_dim, embedding_dim)

    @property
    def num_embeddings(self) -> int:
        """Codebook size. Used by MaskGIT for vocabulary size."""
        return self.fsq.n_codes

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize latent tensor.

        Args:
            z: Input tensor (B, C, D, H, W) where C == embedding_dim

        Returns:
            z_q: Quantized tensor (B, C, D, H, W)
            vq_loss: Always tensor(0.0) - FSQ has no quantization loss
            indices: Codebook indices (B, D, H, W)
        """
        z = z.permute(0, 2, 3, 4, 1).contiguous()

        z_projected = self.project_in(z)

        z_fsq, indices = self.fsq(z_projected)

        z_q = self.project_out(z_fsq)

        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        vq_loss = torch.tensor(0.0, device=z_q.device, requires_grad=False)

        return z_q, vq_loss, indices

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from codebook indices.

        Args:
            indices: Codebook indices (B, D, H, W)

        Returns:
            z_q: Quantized latents (B, C, D, H, W)
        """
        z_fsq = self.fsq.indices_to_codes(indices)

        z_q = self.project_out(z_fsq)

        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q
