"""
Base VQModel class for shared functionality.

This module provides the abstract base class that encapsulates common
logic for all VQGAN/VQVAE models.
"""

from abc import abstractmethod

import torch
import torch.nn as nn

from maskgit3d.domain.interfaces import QuantizerInterface, VQModelInterface


class BaseVQModel(nn.Module, VQModelInterface):  # type: ignore[misc]
    """
    Abstract base class for VQGAN/VQVAE models.

    Provides common functionality including:
    - Forward pass (encode-decode)
    - Codebook index decoding
    - Checkpoint saving/loading
    - Device management
    """

    # Type annotation for quantize module - subclasses will override
    quantize: QuantizerInterface

    def __init__(
        self,
        in_channels: int,
        codebook_size: int,
        embed_dim: int,
        latent_channels: int,
    ):
        """
        Initialize base VQ model.

        Args:
            in_channels: Number of input channels
            codebook_size: Number of codebook entries
            embed_dim: Codebook embedding dimension
            latent_channels: Latent space channels
        """
        super().__init__()
        self._in_channels = in_channels
        self._codebook_size = codebook_size
        self._embed_dim = embed_dim
        self._latent_channels = latent_channels

    @abstractmethod
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Encode volumes to quantized latent codes.

        Args:
            x: Input volumes [B, C, D, H, W]

        Returns:
            Tuple of (quantized, commitment_loss, info)
        """
        pass

    @abstractmethod
    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latents to volumes.

        Args:
            quant: Quantized latents [B, C, D, H, W]

        Returns:
            Reconstructed volumes [B, C, D', H', W']
        """
        pass

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
        # Get actual latent shape from property
        _, D, H, W = self.latent_shape
        N_per_sample = D * H * W

        # Handle different input shapes
        if code.dim() == 4:  # [B, D, H, W]
            B = code.shape[0]
            code_flat = code.view(B, -1).view(-1)  # Flatten to (B*D*H*W,)
        elif code.dim() == 2:  # [B, N]
            B = code.shape[0]
            code_flat = code.view(-1)  # Flatten to (B*N,)
        elif code.dim() == 1:  # [N] or [B*N] - flattened
            total_tokens = code.shape[0]
            B = total_tokens // N_per_sample
            code_flat = code
        else:
            raise ValueError(f"Invalid code shape: {code.shape}. Expected 1D, 2D or 4D tensor.")

        # Get quantized latents from indices
        quant_b = self.quantize.get_codebook_entry(code_flat, shape=(B, D, H, W, self._embed_dim))

        # Decode
        dec = self.decode(quant_b)
        return dec

    def forward_with_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full encode-decode forward pass with quantization loss.

        Args:
            x: Input volumes [B, C, D, H, W]

        Returns:
            Tuple of (reconstructed_volumes, quantization_loss)
        """
        quant, diff, _ = self.encode(x)
        dec = self.decode(quant)
        return dec, diff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only reconstruction (ModelInterface contract).

        Args:
            x: Input volumes [B, C, D, H, W]

        Returns:
            Reconstructed volumes [B, C, D', H', W']
        """
        dec, _ = self.forward_with_loss(x)
        return dec

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
    @abstractmethod
    def latent_shape(self) -> tuple[int, int, int, int]:
        """Get the shape of latent representations (C, D, H, W)."""
        pass
