from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MaskWeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with per-token mask weights for BERT-style masked language modeling.

    This loss function computes cross-entropy loss with:
    - Support for padding tokens (ignored via -100 labels)
    - Per-token weighting via mask_weights
    - Normalization by sum of weights (not count)

    Args:
        ignore_index: Index to ignore in loss computation. Default: -100.

    Example:
        >>> loss_fn = MaskWeightedCrossEntropyLoss()
        >>> logits = torch.randn(2, 3, 5)  # [B, L, V]
        >>> labels = torch.tensor([[1, 2, -100], [0, 1, 2]])
        >>> mask_weights = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        >>> loss = loss_fn(logits, labels, mask_weights)
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,  # [B, L, V] or [B*L, V]
        labels: torch.Tensor,  # [B, L] or [B*L] - -100 for ignored positions
        mask_weights: torch.Tensor,  # [B, L] or [B*L] - weight per token
    ) -> torch.Tensor:
        """Compute mask-weighted cross-entropy loss.

        Args:
            logits: Predicted logits. Shape [B, L, V] or [B*L, V] where
                B = batch size, L = sequence length, V = vocabulary size.
            labels: Target labels. Shape [B, L] or [B*L]. Use -100 for ignored positions.
            mask_weights: Weight per token. Shape [B, L] or [B*L]. Use 0 for padding/ignored.

        Returns:
            Scalar loss value, normalized by sum of weights.
        """
        # Flatten all inputs to 1D for cross_entropy
        if logits.dim() == 3:
            # [B, L, V] -> [B*L, V]
            B, L, V = logits.shape
            logits = logits.reshape(B * L, V)
            labels = labels.reshape(B * L)
            mask_weights = mask_weights.reshape(B * L)
        elif labels.dim() == 2:
            # Logits already 2D, flatten labels and weights to match
            labels = labels.reshape(-1)
            mask_weights = mask_weights.reshape(-1)

        # Compute per-token loss with no reduction
        loss_per_token = F.cross_entropy(
            logits,
            labels,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # Apply mask weights
        weighted_loss = loss_per_token * mask_weights

        # Normalize by sum of weights (handle edge case where all weights are 0)
        weight_sum = mask_weights.sum()
        if weight_sum == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=False)

        return weighted_loss.sum() / weight_sum
