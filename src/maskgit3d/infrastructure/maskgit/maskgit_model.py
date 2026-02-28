"""
MaskGIT Model implementation.

Complete MaskGIT model combining:
- VQGAN tokenizer (Encoder + Quantizer + Decoder)
- Bidirectional Transformer for masked token prediction
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from maskgit3d.domain.interfaces import MaskGITModelInterface
from maskgit3d.infrastructure.vqgan.vqgan_model_3d import VQModel3D
from maskgit3d.infrastructure.maskgit.transformer import MaskGITTransformer
from maskgit3d.infrastructure.maskgit.sampling import MaskGITSampler


class MaskGITModel(nn.Module, MaskGITModelInterface):
    """
    Complete MaskGIT model for 3D medical image generation.

    Combines:
    - VQGAN tokenizer: encodes images to discrete tokens
    - Transformer: bidirectional model for token prediction
    """

    def __init__(
        self,
        vqgan: Optional[VQModel3D] = None,
        transformer: Optional[MaskGITTransformer] = None,
        # VQGAN config
        in_channels: int = 1,
        codebook_size: int = 1024,
        embed_dim: int = 256,
        latent_channels: int = 256,
        resolution: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 1, 2, 2, 4),
        # Transformer config
        transformer_hidden: int = 768,
        transformer_layers: int = 12,
        transformer_heads: int = 12,
        # Training config
        mask_ratio: float = 0.5,
    ):
        """
        Args:
            vqgan: Optional pre-configured VQModel
            transformer: Optional pre-configured Transformer
            in_channels: Number of input channels
            codebook_size: Size of VQ codebook
            embed_dim: Embedding dimension for codebook
            latent_channels: Latent space channels
            resolution: Input resolution
            channel_multipliers: Channel multipliers for encoder/decoder
            transformer_hidden: Hidden dimension for transformer
            transformer_layers: Number of transformer layers
            transformer_heads: Number of attention heads
            mask_ratio: Ratio of tokens to mask during training
        """
        super().__init__()

        self.in_channels = in_channels
        self._codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        # Build VQGAN if not provided
        if vqgan is None:
            self.vqgan = self._build_vqgan(
                in_channels=in_channels,
                codebook_size=codebook_size,
                embed_dim=embed_dim,
                latent_channels=latent_channels,
                resolution=resolution,
                channel_multipliers=channel_multipliers,
            )
        else:
            self.vqgan = vqgan

        # Build Transformer if not provided
        # vocab_size = codebook_size + 1 (for mask token)
        if transformer is None:
            self.transformer = MaskGITTransformer(
                vocab_size=codebook_size + 1,
                hidden_size=transformer_hidden,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
            )
        else:
            self.transformer = transformer

        # Sampler for inference
        self.sampler = MaskGITSampler(num_iterations=12)

        # Latent shape (computed from VQGAN)
        self._latent_shape = self._compute_latent_shape(resolution, channel_multipliers)

    def _build_vqgan(
        self,
        in_channels: int,
        codebook_size: int,
        embed_dim: int,
        latent_channels: int,
        resolution: int,
        channel_multipliers: Tuple[int, ...],
    ) -> "VQModel3D":
        """Build 3D VQGAN model."""
        from maskgit3d.infrastructure.vqgan.vqgan_model_3d import VQModel3D

        return VQModel3D(
            in_channels=in_channels,
            codebook_size=codebook_size,
            embed_dim=embed_dim,
            latent_channels=latent_channels,
            resolution=resolution,
            channel_multipliers=channel_multipliers,
        )

    def _compute_latent_shape(
        self,
        resolution: int,
        channel_multipliers: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        """Compute latent shape after VQGAN encoding."""
        # Each downsampling step halves the resolution
        num_downsamples = len(channel_multipliers) - 1
        latent_res = resolution // (2**num_downsamples)
        return (1, latent_res, latent_res, latent_res)

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
    def latent_shape(self) -> Tuple[int, int, int, int]:
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
        quant, _, _ = self.vqgan.quantize(z)
        # Get indices
        indices = info[2]  # Codebook indices
        return indices

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens to images.

        Args:
            tokens: Token indices [B, D, H, W] or [B, N]

        Returns:
            Reconstructed images [B, C, D', H', W']
        """
        # Handle different token shapes
        if tokens.dim() == 4:  # [B, D, H, W]
            B, D, H, W = tokens.shape
            tokens_flat = tokens.view(B, -1)
            shape = (B, D, H, W)
        else:
            tokens_flat = tokens
            B = tokens_flat.shape[0]
            _, D, H, W = self._latent_shape
            shape = (B, D, H, W)

        # Get quantized latents from indices
        quant = self.vqgan.quantize.get_codebook_entry(tokens_flat, shape)

        # Decode through VQGAN
        return self.vqgan.decode(quant)

    def generate(
        self,
        num_tokens: Optional[int] = None,
        shape: Optional[Tuple[int, ...]] = None,
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

    def train_step(
        self,
        x: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Execute one training step with masked token prediction.

        Args:
            x: Input volumes [B, C, D, H, W]

        Returns:
            Dictionary of training metrics
        """
        _, metrics = self.compute_maskgit_loss(x=x, mask_ratio=self.mask_ratio)
        return metrics

    def compute_maskgit_loss(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MaskGIT cross-entropy loss and detached scalar metrics."""
        # Encode to tokens
        tokens = self.encode_tokens(x)
        B, D, H, W = tokens.shape
        tokens_flat = tokens.view(B, -1)

        # Random masking
        mask = torch.rand(B, D * H * W, device=tokens.device) < mask_ratio

        # Ensure at least one token masked per sample
        for i in range(B):
            if not mask[i].any():
                mask[i, torch.randint(0, D * H * W, (1,), device=tokens.device)] = True

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
        checkpoint = torch.load(path, map_location=self.device)
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
