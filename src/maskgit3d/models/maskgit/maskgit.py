"""MaskGIT model implementation.

Complete MaskGIT model combining:
- VQVAE tokenizer (Encoder + Quantizer + Decoder)
- Bidirectional Transformer for masked token prediction
"""

from typing import Any

import torch
import torch.nn as nn
from monai.inferers.inferer import SlidingWindowInferer

from ..vqvae import VQVAE
from .sampling import MaskGITSampler
from .scheduling import TrainingMaskScheduler
from .transformer import MaskGITTransformer


class MaskGIT(nn.Module):
    """MaskGIT model for 3D medical image generation.

    Combines:
    - VQVAE tokenizer: encodes images to discrete tokens
    - Transformer: bidirectional model for token prediction
    - Sliding window inference support for large images

    Args:
        vqvae: VQVAE model for encoding/decoding
        hidden_size: Transformer hidden dimension (default: 768)
        num_layers: Number of transformer layers (default: 12)
        num_heads: Number of attention heads (default: 12)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        dropout: Dropout probability (default: 0.1)
        gamma_type: Mask scheduling gamma type (default: "cosine")
        num_iterations: Number of decoding iterations (default: 12)
        temperature: Sampling temperature (default: 1.0)
        sliding_window: Sliding window configuration for encoding/decoding large images
    """

    def __init__(
        self,
        vqvae: VQVAE,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        gamma_type: str = "cosine",
        num_iterations: int = 12,
        temperature: float = 1.0,
        sliding_window: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.vqvae = vqvae
        self._codebook_size = vqvae.quantizer.num_embeddings

        # Mask token is the last token in vocab (codebook_size)
        self.mask_token_id = self._codebook_size
        vocab_size = self._codebook_size + 1

        # Build transformer
        self.transformer = MaskGITTransformer(
            vocab_size=vocab_size,
            mask_token_id=self.mask_token_id,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Mask scheduler for training
        self.mask_scheduler = TrainingMaskScheduler(gamma_type=gamma_type)

        # Sampler for inference
        self.sampler = MaskGITSampler(
            num_iterations=num_iterations,
            temperature=temperature,
        )

        # Sliding window configuration
        self.sliding_window_cfg = sliding_window or {}
        self._sliding_window_inferer: SlidingWindowInferer | None = None
        self._downsampling_factor = 16

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def num_tokens(self) -> int:
        return self.codebook_size + 1

    def _get_sliding_window_inferer(self) -> SlidingWindowInferer | None:
        if not self.sliding_window_cfg.get("enabled", False):
            return None
        if self._sliding_window_inferer is None:
            roi_size = tuple(self.sliding_window_cfg.get("roi_size", [64, 64, 64]))
            overlap = self.sliding_window_cfg.get("overlap", 0.25)
            mode = self.sliding_window_cfg.get("mode", "gaussian")
            sigma_scale = self.sliding_window_cfg.get("sigma_scale", 0.125)
            sw_batch_size = self.sliding_window_cfg.get("sw_batch_size", 1)
            self._sliding_window_inferer = SlidingWindowInferer(
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
                mode=mode,
                sigma_scale=sigma_scale,
                padding_mode="constant",
                cval=0.0,
            )
        return self._sliding_window_inferer

    def _pad_to_divisible(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        k = self._downsampling_factor
        d_new = ((D + k - 1) // k) * k
        h_new = ((H + k - 1) // k) * k
        w_new = ((W + k - 1) // k) * k
        pad_d = d_new - D
        pad_h = h_new - H
        pad_w = w_new - W
        pad = (
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
            pad_d // 2,
            pad_d - pad_d // 2,
        )
        return nn.functional.pad(x, pad, mode="constant", value=-1.0)

    def _to_transformer_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert VQVAE tokens to Transformer tokens."""
        tokens = tokens.long()
        if torch.any(tokens < 0) or torch.any(tokens >= self.codebook_size):
            raise ValueError(
                f"VQVAE token indices out of range: expected [0, {self.codebook_size - 1}], "
                f"got [{tokens.min().item()}, {tokens.max().item()}]"
            )
        # Shift by 1 and wrap: VQVAE [0, codebook_size-1] -> Transformer [1, codebook_size-1, 0]
        return (tokens + 1) % self.codebook_size

    def _to_vq_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.long()
        if bool(torch.any(tokens == self.mask_token_id)):
            raise ValueError("Cannot decode tokens that still contain mask token id")
        if bool(torch.any(tokens < 0)) or bool(torch.any(tokens >= self.codebook_size)):
            raise ValueError(
                f"Token indices out of range for decode: expected [0, {self.codebook_size - 1}]"
            )
        return (tokens - 1) % self.codebook_size

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to discrete latent tokens.

        Args:
            x: Input images [B, C, D, H, W]

        Returns:
            Token indices [B, D', H', W']
        """
        _, _, indices = self.vqvae.encode(x)
        return self._to_transformer_tokens(indices)

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens to images.

        Args:
            tokens: Token indices [B, D, H, W] or [B, N] or [N] or [B*N]

        Returns:
            Reconstructed images [B, C, D', H', W']
        """
        return self.vqvae.decode_from_indices(self._to_vq_tokens(tokens))

    def encode_images_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to discrete tokens with sliding window support.

        Args:
            x: Input images [B, C, D, H, W]

        Returns:
            Token indices [B, D', H', W']
        """
        inferer = self._get_sliding_window_inferer()

        if inferer is None:
            return self.encode_tokens(x)

        original_shape = x.shape[2:]
        x_padded = self._pad_to_divisible(x)

        def encode_fn(patch: torch.Tensor) -> torch.Tensor:
            z_q, _, indices = self.vqvae.encode(patch)
            return indices.float().unsqueeze(1)

        indices_padded = inferer(x_padded, encode_fn)

        latent_d = original_shape[0] // self._downsampling_factor
        latent_h = original_shape[1] // self._downsampling_factor
        latent_w = original_shape[2] // self._downsampling_factor

        B = x.shape[0]
        indices = indices_padded[:B, 0, :latent_d, :latent_h, :latent_w]  # type: ignore[index]

        indices = indices.long()
        return self._to_transformer_tokens(indices)

    def decode_tokens_to_latent(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens to latent/images with sliding window support.

        Args:
            tokens: Token indices [B, D, H, W]

        Returns:
            Decoded latent/images [B, C, D*16, H*16, W*16]
        """
        if tokens.dim() == 2:
            B, N = tokens.shape
            D = H = W = int(round(N ** (1 / 3)))
            tokens = tokens.view(B, D, H, W)
        elif tokens.dim() != 4:
            raise ValueError(f"Expected tokens to be 4D [B, D, H, W], got {tokens.shape}")

        vq_tokens = self._to_vq_tokens(tokens)
        inferer = self._get_sliding_window_inferer()

        if inferer is None:
            z_q = self.vqvae.quantizer.decode_from_indices(vq_tokens)
            return self.vqvae.decode(z_q)

        z_q = self.vqvae.quantizer.decode_from_indices(vq_tokens)

        def decode_fn(patch: torch.Tensor) -> torch.Tensor:
            return self.vqvae.decode(patch)

        result = inferer(z_q, decode_fn)
        assert isinstance(result, torch.Tensor)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (reconstruction).

        Args:
            x: Input volumes [B, C, D, H, W]

        Returns:
            Reconstructed volumes [B, C, D, H, W]
        """
        tokens = self.encode_tokens(x)
        return self.decode_tokens(tokens)

    def compute_loss(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        """Compute MaskGIT cross-entropy loss and detached scalar metrics.

        Args:
            x: Input images [B, C, D, H, W]
            mask_ratio: Optional mask ratio (if None, samples from scheduler)

        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        # Encode to tokens
        tokens = self.encode_tokens(x)

        # Handle different token shapes
        if tokens.dim() == 1:
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

        # Sample mask ratio if not provided
        if mask_ratio is None:
            mask_ratio = self.mask_scheduler.sample_mask_ratio()

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

        actual_mask_ratio: float = float(mask.float().mean().item())
        metrics: dict[str, float] = {
            "loss": loss.item(),
            "mask_acc": acc.item(),
            "mask_ratio": actual_mask_ratio,
        }
        return loss, metrics

    def generate(
        self,
        shape: tuple[int, ...] | None = None,
        temperature: float = 1.0,
        num_iterations: int = 12,
        **kwargs,
    ) -> torch.Tensor:
        """Generate images from random tokens using iterative decoding.

        Args:
            shape: Shape of token grid (B, D, H, W)
            temperature: Sampling temperature
            num_iterations: Number of decoding iterations

        Returns:
            Generated images
        """
        device = next(self.parameters()).device

        # Determine shape
        if shape is None:
            # Default shape based on VQVAE latent shape
            B = 1
            shape = (B, 4, 4, 4)  # Default latent shape
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
