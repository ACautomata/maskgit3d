"""
MaskGIT Model implementation.

Complete MaskGIT model combining:
- VQGAN tokenizer (Encoder + Quantizer + Decoder)
- Bidirectional Transformer for masked token prediction
"""

import torch
import torch.nn as nn

from maskgit3d.domain.interfaces import MaskGITModelInterface, VQModelInterface
from maskgit3d.infrastructure.maskgit.sampling import MaskGITSampler
from maskgit3d.infrastructure.maskgit.transformer import MaskGITTransformer


class MaskGITModel(MaskGITModelInterface):
    """
    Complete MaskGIT model for 3D medical image generation.

    Combines:
    - VQVAE tokenizer: encodes images to discrete tokens
    - Transformer: bidirectional model for token prediction
    """

    def __init__(
        self,
        vqgan: VQModelInterface,
        transformer: MaskGITTransformer,
        mask_ratio: float = 0.5,
    ):
        """
        Args:
            vqgan: VQVAE model for encoding/decoding
            transformer: Transformer for masked token prediction
            mask_ratio: (DEPRECATED - not used) Mask ratio is now dynamically sampled during training
        """
        super().__init__()

        self.vqgan = vqgan
        self.transformer = transformer
        self._codebook_size = vqgan.codebook_size
        self._latent_shape = (1,) + vqgan.latent_shape[1:]

        self.sampler = MaskGITSampler(num_iterations=12)

    @property
    def embed_dim(self) -> int:
        """Embedding dimension (from latent shape)."""
        return self._latent_shape[1]

    @property
    def device(self) -> torch.device:
        """Get device of the model."""
        return next(self.parameters()).device

    @property
    def num_tokens(self) -> int:
        """Total number of tokens (codebook_size + 1 for mask)."""
        return self.codebook_size + 1

    @property
    def codebook_size(self) -> int:
        """Get configured codebook size."""
        return self._codebook_size

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        """Shape of latent representations (B, D, H, W)."""
        return self._latent_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (reconstruction).

        Args:
            x: Input volumes [B, C, D, H, W]

        Returns:
            Reconstructed volumes [B, C, D, H, W]
        """
        # Encode to tokens
        tokens = self.encode_tokens(x)
        # Decode tokens
        return self.decode_tokens(tokens)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to discrete latent tokens.

        Args:
            x: Input images [B, C, D, H, W]

        Returns:
            Token indices [B, D, H, W]
        """
        # Encode through VQGAN
        z, _, info = self.vqgan.encode(x)
        # Get quantized latents
        quant, _, _ = self.vqgan.quantize(z)  # type: ignore[operator]
        # Get indices
        indices = info[2]  # Codebook indices
        return indices

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens to images.

        Args:
            tokens: Token indices [B, D, H, W] or [B, N] or [N] or [B*N]

        Returns:
            Reconstructed images [B, C, D', H', W']
        """
        # Use VQGAN's decode_code method which handles all shape logic
        return self.vqgan.decode_code(tokens)

    def generate(
        self,
        num_tokens: int | None = None,
        shape: tuple[int, ...] | None = None,
        temperature: float = 1.0,
        num_iterations: int = 12,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate images from random tokens using iterative decoding.

        Args:
            num_tokens: Total number of tokens (D * H * W)
            shape: Shape of token grid (B, D, H, W)
            temperature: Sampling temperature
            num_iterations: Number of decoding iterations

        Returns:
            Generated images
        """
        device = self.device

        # Determine shape
        if shape is None:
            latent_shape = self._latent_shape
            B = 1
            shape = (B,) + latent_shape[1:]
        elif len(shape) != 4:
            raise ValueError("shape must be a 4D tuple: (B, D, H, W)")

        B, D, H, W = shape

        # Create sampler with specified iterations
        sampler = MaskGITSampler(
            num_iterations=num_iterations,
            temperature=temperature,
        )

        # Generate tokens
        tokens = sampler.sample(
            model=self.transformer,
            shape=shape,
            device=device,
        )

        # Decode tokens to images
        return self.decode_tokens(tokens)

    def compute_maskgit_loss(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute MaskGIT cross-entropy loss and detached scalar metrics."""
        # Encode to tokens
        tokens = self.encode_tokens(x)

        # Handle different token shapes dynamically
        if tokens.dim() == 1:
            # Flattened tokens [N] - reshape assuming batch size
            B = x.shape[0]
            tokens = tokens.view(B, -1)

        if tokens.dim() == 4:
            B, D, H, W = tokens.shape
            tokens_flat = tokens.view(B, -1)
        elif tokens.dim() == 3:
            B, D, N = tokens.shape
            tokens_flat = tokens.view(B, -1)
        elif tokens.dim() == 2:
            B, N = tokens.shape
            tokens_flat = tokens
        else:
            raise ValueError(f"Unexpected tokens shape: {tokens.shape}")

        n_total = tokens_flat.shape[1]

        # Random masking
        mask = torch.rand(B, n_total, device=tokens.device) < mask_ratio

        # Ensure at least one token masked per sample
        for i in range(B):
            if not mask[i].any():
                mask[i, torch.randint(0, n_total, (1,), device=tokens.device)] = True

        # Get predictions
        logits = self.transformer.forward(tokens_flat, mask_indices=mask)

        # Compute loss only on masked positions
        masked_logits = logits[mask]
        masked_targets = tokens_flat[mask]

        loss = nn.functional.cross_entropy(masked_logits, masked_targets)

        # Get accuracy on masked positions
        with torch.no_grad():
            preds = masked_logits.argmax(dim=-1)
            acc = (preds == masked_targets).float().mean()

        metrics = {
            "loss": loss.item(),
            "mask_acc": acc.item(),
            "mask_ratio": mask.float().mean().item(),
        }
        return loss, metrics

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "vqgan": self.vqgan.state_dict(),
                "transformer": self.transformer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if "vqgan" in checkpoint:
            self.vqgan.load_state_dict(checkpoint["vqgan"])
        if "transformer" in checkpoint:
            self.transformer.load_state_dict(checkpoint["transformer"])


class MaskGITModelConfig:
    """Configuration for MaskGIT model."""

    @staticmethod
    def create_config(
        image_size: int = 64,
        in_channels: int = 1,
        codebook_size: int = 1024,
        embed_dim: int = 256,
        latent_channels: int = 256,
        transformer_hidden: int = 768,
        transformer_layers: int = 12,
        transformer_heads: int = 12,
        mask_ratio: float = 0.5,
    ) -> dict:
        """Create configuration dictionary."""
        return {
            "in_channels": in_channels,
            "codebook_size": codebook_size,
            "embed_dim": embed_dim,
            "latent_channels": latent_channels,
            "resolution": image_size,
            "channel_multipliers": (1, 1, 2, 2, 4),
            "transformer_hidden": transformer_hidden,
            "transformer_layers": transformer_layers,
            "transformer_heads": transformer_heads,
            "mask_ratio": mask_ratio,
        }
