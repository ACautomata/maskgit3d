"""Sequence builder for constructing in-context sequences from multi-modal latents."""

from __future__ import annotations

import torch
import torch.nn as nn


class InContextSequenceBuilder(nn.Module):
    """Builds in-context sequences from multiple modality latents.

    Constructs sequences for transformer input with the following format:
        [CLS] [MOD_LABEL_0] [LATENT_0_FLAT...] [MOD_LABEL_1] [LATENT_1_FLAT...]
        ... [MOD_TARGET_LABEL] [TARGET_LATENT_FLAT...] [SEP]

    Token ID allocation:
        - Real vocab (VQVAE codebook): 0 to vocab_size - 1
        - CLS token: vocab_size
        - SEP token: vocab_size + 1
        - Modality labels: vocab_size + 2 + modality_id

    Args:
        num_modalities: Number of distinct modalities to support.
        latent_spatial_size: Spatial dimensions of latent tensors (D', H', W').
        vocab_size: Size of the VQVAE codebook.
        cls_token_id: Optional custom CLS token ID. Defaults to vocab_size.
        sep_token_id: Optional custom SEP token ID. Defaults to vocab_size + 1.

    Attributes:
        num_modalities: Number of modalities.
        latent_spatial_size: Spatial dimensions of latents.
        vocab_size: Codebook size.
        cls_token_id: Token ID for CLS token.
        sep_token_id: Token ID for SEP token.
    """

    def __init__(
        self,
        num_modalities: int,
        latent_spatial_size: tuple[int, int, int],
        vocab_size: int,
        cls_token_id: int | None = None,
        sep_token_id: int | None = None,
    ) -> None:
        super().__init__()

        self.num_modalities = num_modalities
        self.latent_spatial_size = latent_spatial_size
        self.vocab_size = vocab_size

        # Create special token IDs if not provided
        self.cls_token_id = cls_token_id if cls_token_id is not None else vocab_size
        self.sep_token_id = sep_token_id if sep_token_id is not None else vocab_size + 1

        # Latent size (flattened)
        self._latent_size = latent_spatial_size[0] * latent_spatial_size[1] * latent_spatial_size[2]

    def get_modality_label_id(self, modality_id: int) -> int:
        """Get the token ID for a modality label.

        Args:
            modality_id: The modality ID (0-indexed).

        Returns:
            Token ID for the modality label.
        """
        return self.vocab_size + 2 + modality_id

    def build(
        self,
        context_latents: list[torch.Tensor],
        target_latent: torch.Tensor,
        context_modality_ids: list[int],
        target_modality_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build an in-context sequence from context and target latents.

        Args:
            context_latents: List of context latent tensors, each of shape
                [B, D', H', W'] containing token indices.
            target_latent: Target latent tensor of shape [B, D', H', W'].
            context_modality_ids: List of modality IDs for each context latent.
            target_modality_id: Modality ID for the target latent.

        Returns:
            Tuple of:
                - sequence: [B, L] token sequence.
                - target_mask: [B, L] bool tensor, True for target latent positions.
                - attention_mask: [B, L] float tensor, all 1s (no padding).
        """
        batch_size = target_latent.shape[0]
        device = target_latent.device

        # Calculate sequence length
        # CLS + (MOD_LABEL + LATENT) * num_context + MOD_TARGET + TARGET + SEP
        num_context = len(context_latents)
        seq_len = (
            1  # CLS
            + num_context * (1 + self._latent_size)  # Context: MOD + latent
            + 1  # MOD_TARGET
            + self._latent_size  # Target latent
            + 1  # SEP
        )

        # Initialize tensors
        sequence = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        target_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float, device=device)

        # Build sequence
        pos = 0

        # CLS token
        sequence[:, pos] = self.cls_token_id
        pos += 1

        # Context latents
        for latent, mod_id in zip(context_latents, context_modality_ids, strict=True):
            # Modality label
            sequence[:, pos] = self.get_modality_label_id(mod_id)
            pos += 1

            # Flatten and add latent tokens
            # latent shape: [B, D', H', W'] -> [B, D'*H'*W']
            flat_latent = latent.reshape(batch_size, -1)
            sequence[:, pos : pos + self._latent_size] = flat_latent
            pos += self._latent_size

        # Target modality label
        sequence[:, pos] = self.get_modality_label_id(target_modality_id)
        pos += 1

        # Target latent
        flat_target = target_latent.reshape(batch_size, -1)
        sequence[:, pos : pos + self._latent_size] = flat_target
        # Set target mask for target latent positions
        target_mask[:, pos : pos + self._latent_size] = True
        pos += self._latent_size

        # SEP token
        sequence[:, pos] = self.sep_token_id

        return sequence, target_mask, attention_mask

    def build_sample(
        self,
        context_latents: list[torch.Tensor],
        target_latent: torch.Tensor,
        context_modality_ids: list[int],
        target_modality_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a single sample's sequence (unbatched, 1D output).

        Used for any2one training where each sample has different context counts.

        Args:
            context_latents: List of context latent tensors, each of shape [D', H', W'].
            target_latent: Target latent tensor of shape [D', H', W'].
            context_modality_ids: List of modality IDs for each context latent.
            target_modality_id: Modality ID for the target latent.

        Returns:
            Tuple of:
                - sequence: [L] token sequence (1D).
                - target_mask: [L] bool tensor, True for target latent positions.
        """
        device = target_latent.device
        num_context = len(context_latents)
        seq_len = 1 + num_context * (1 + self._latent_size) + 1 + self._latent_size + 1

        sequence = torch.zeros(seq_len, dtype=torch.long, device=device)
        target_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

        pos = 0
        sequence[pos] = self.cls_token_id
        pos += 1

        for latent, mod_id in zip(context_latents, context_modality_ids, strict=True):
            sequence[pos] = self.get_modality_label_id(mod_id)
            pos += 1
            flat_latent = latent.reshape(-1)
            sequence[pos : pos + self._latent_size] = flat_latent
            pos += self._latent_size

        sequence[pos] = self.get_modality_label_id(target_modality_id)
        pos += 1

        flat_target = target_latent.reshape(-1)
        sequence[pos : pos + self._latent_size] = flat_target
        target_mask[pos : pos + self._latent_size] = True
        pos += self._latent_size

        sequence[pos] = self.sep_token_id

        return sequence, target_mask
