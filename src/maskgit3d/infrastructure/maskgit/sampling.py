"""
MaskGIT iterative sampling logic.

Implements the MaskGIT decoding algorithm:
1. Start with all tokens masked
2. Iteratively predict and unmask a fraction of tokens
3. Use confidence-based scheduling to determine which tokens to unmask
"""

import math

import torch
import torch.nn.functional as F

from maskgit3d.infrastructure.maskgit.transformer import MaskGITTransformer
from maskgit3d.infrastructure.vqgan import VQVAE


class MaskGITSampler:
    """
    Iterative decoder for MaskGIT.

    Uses a masking schedule to progressively reveal tokens during generation.
    """

    def __init__(
        self,
        num_iterations: int = 12,
        temperature: float = 1.0,
        mask_type: str = "random",  # "random" or "confidence"
    ):
        """
        Args:
            num_iterations: Number of decoding iterations
            temperature: Sampling temperature for softmax
            mask_type: Strategy for selecting tokens to unmask
        """
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.mask_type = mask_type

        # Precompute masking schedule (fraction to unmask each iteration)
        self.schedule = self._get_schedule(num_iterations)

    def _get_schedule(self, num_iterations: int) -> torch.Tensor:
        """
        Generate masking schedule.

        Uses cosine schedule from MaskGIT paper:
        Each iteration reveals more tokens.
        """
        steps = torch.arange(num_iterations + 1)
        # Cosine schedule: starts high, decreases
        schedule = (1 - torch.cos(steps.float() * math.pi / num_iterations)) / 2
        # Convert to reveal ratio per iteration
        reveal_ratios = schedule[1:] - schedule[:-1]
        return reveal_ratios

    def _resolve_mask_token_id(self, model: MaskGITTransformer) -> int:
        mask_token_id = getattr(model, "mask_token_id", None)
        if isinstance(mask_token_id, int):
            return mask_token_id

        vocab_size = getattr(model, "vocab_size", None)
        if isinstance(vocab_size, int):
            return vocab_size - 1

        codebook_size = getattr(model, "codebook_size", None)
        if isinstance(codebook_size, int):
            return codebook_size

        return 0

    def sample(
        self,
        model: MaskGITTransformer,
        shape: tuple[int, int, int, int],
        device: torch.device,
        vqgan_model: VQVAE | None = None,
        cond_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate samples using iterative decoding.

        Args:
            model: MaskGIT Transformer model
            shape: Shape of token grid (B, D, H, W)
            device: Device to use
            vqgan_model: Optional VQGAN model for reconstruction
            cond_tokens: Optional conditioning tokens

        Returns:
            Generated token indices [B, D, H, W]
        """
        B, D, H, W = shape
        N = D * H * W  # Total tokens

        # Initialize with all masked tokens
        tokens = torch.zeros(B, N, dtype=torch.long, device=device)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)  # All masked

        mask_token_id = self._resolve_mask_token_id(model)

        # Iterative decoding
        for iteration in range(self.num_iterations):
            # Set masked positions to mask token
            tokens_input = tokens.clone()
            tokens_input[mask] = mask_token_id

            # Get predictions
            with torch.no_grad():
                logits = model.encode(tokens_input, return_logits=True)

            if 0 <= mask_token_id < logits.shape[-1]:
                logits[..., mask_token_id] = -float("inf")

            # Apply temperature
            if self.temperature > 0:
                logits = logits / self.temperature

            # Get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample or use argmax for masked positions
            if iteration < self.num_iterations - 1:
                # Sample from distribution
                pred_tokens = torch.multinomial(probs.view(B * N, -1), 1).view(B, N)
            else:
                # Final iteration: use argmax
                pred_tokens = logits.argmax(dim=-1)

            # Determine which tokens to reveal this iteration
            remaining_masked = mask.sum(dim=1)
            if iteration == self.num_iterations - 1:
                num_to_reveal = int(remaining_masked.max().item())
            else:
                num_to_reveal = int(N * self.schedule[iteration].item())
            if num_to_reveal > 0:
                # Get confidence scores (probability of predicted token)
                confidence = probs.gather(2, pred_tokens.unsqueeze(2)).squeeze(2)

                # Handle different mask types
                if self.mask_type == "confidence":
                    # Reveal most confident predictions
                    reveal_mask = self._get_confidence_based_mask(confidence, mask, num_to_reveal)
                else:
                    # Random reveal
                    reveal_mask = self._get_random_mask(mask, num_to_reveal)

                # Update tokens with predictions
                tokens[reveal_mask] = pred_tokens[reveal_mask]
                mask = mask & ~reveal_mask  # Remove revealed from mask

        return tokens.view(B, D, H, W)

    def _get_random_mask(
        self,
        current_mask: torch.Tensor,
        num_to_reveal: int,
    ) -> torch.Tensor:
        """Randomly select tokens to reveal."""
        B, N = current_mask.shape
        reveal_mask = torch.zeros_like(current_mask)

        for i in range(B):
            masked_indices = torch.where(current_mask[i])[0]
            if len(masked_indices) > 0:
                num = min(num_to_reveal, len(masked_indices))
                selected = masked_indices[
                    torch.randperm(len(masked_indices), device=masked_indices.device)[:num]
                ]
                reveal_mask[i, selected] = True

        return reveal_mask

    def _get_confidence_based_mask(
        self,
        confidence: torch.Tensor,
        current_mask: torch.Tensor,
        num_to_reveal: int,
    ) -> torch.Tensor:
        """
        Select most confident predictions to reveal.

        This is the MaskGIT approach - reveal high confidence predictions first.
        """
        B, N = confidence.shape
        reveal_mask = torch.zeros_like(current_mask)

        for i in range(B):
            # Get confidence only for currently masked positions
            masked_conf = confidence[i].clone()
            masked_conf[~current_mask[i]] = -float("inf")  # Exclude already revealed

            # Get top-k indices
            if num_to_reveal > 0:
                _, top_indices = torch.topk(masked_conf, min(num_to_reveal, N))
                reveal_mask[i, top_indices] = True

        return reveal_mask


class MaskGITSamplerWithVQGAN:
    """
    Complete sampler that includes VQGAN decoding.

    Generates tokens with MaskGIT Transformer, then decodes with VQGAN.
    """

    def __init__(
        self,
        num_iterations: int = 12,
        temperature: float = 1.0,
    ):
        self.maskgit_sampler = MaskGITSampler(
            num_iterations=num_iterations,
            temperature=temperature,
        )

    def sample(
        self,
        maskgit_model: MaskGITTransformer,
        vqgan_model: VQVAE,
        shape: tuple[int, int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate images using MaskGIT + VQGAN.

        Args:
            maskgit_model: MaskGIT Transformer model
            vqgan_model: VQGAN model for decoding tokens
            shape: Shape of token grid (B, D, H, W)
            device: Device to use

        Returns:
            Generated volumes [B, C, D, H, W]
        """
        # Generate tokens
        tokens = self.maskgit_sampler.sample(
            model=maskgit_model,
            shape=shape,
            device=device,
        )

        # Decode tokens to volumes
        with torch.no_grad():
            volumes = vqgan_model.decode_code(tokens)

        return volumes


def create_mask_schedule(
    num_iterations: int,
    mode: str = "cosine",
) -> torch.Tensor:
    """
    Create masking schedule for iterative decoding.

    Args:
        num_iterations: Number of iterations
        mode: Schedule type ("cosine", "linear", "sqrt")

    Returns:
        Tensor of reveal ratios for each iteration
    """
    if mode == "cosine":
        steps = torch.arange(num_iterations + 1)
        schedule = (1 - torch.cos(steps.float() * math.pi / num_iterations)) / 2
        reveal_ratios = schedule[1:] - schedule[:-1]
    elif mode == "linear":
        reveal_ratios = torch.ones(num_iterations) / num_iterations
    elif mode == "sqrt":
        steps = torch.arange(1, num_iterations + 1)
        reveal_ratios = 1 / torch.sqrt(steps.float())
        reveal_ratios = reveal_ratios / reveal_ratios.sum()
    else:
        raise ValueError(f"Unknown schedule mode: {mode}")

    return reveal_ratios
