"""Variable-length transformer for in-context learning.

A transformer that handles variable-length sequences with proper attention
and padding masks. Unlike the base MaskGITTransformer, this version accepts
and uses attention masks to handle padded sequences.
"""

import torch
import torch.nn as nn

from src.maskgit3d.models.maskgit.transformer import (
    SinusoidalPositionalEncoding,
    TransformerBlock,
)


class VariableLengthMaskGITTransformer(nn.Module):
    """Bidirectional Transformer for variable-length sequences.

    Handles padded sequences by accepting an attention mask that indicates
    which positions are real tokens (1) vs padding (0).

    The attention mask is converted to a 2D mask for MultiheadAttention,
    ensuring that padding positions do not affect the output of real tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
    ):
        """Initialize the variable-length transformer.

        Args:
            vocab_size: Size of vocabulary (codebook_size + 1 for mask)
            mask_token_id: Token index reserved for mask token
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.token_embed.weight, std=0.02)

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size, max_len=max_seq_len)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _create_attention_mask(
        self,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convert 1D padding mask to 3D attention mask for MultiheadAttention.

        Args:
            attention_mask: [B, L] where 1 = real token, 0 = padding

        Returns:
            3D attention mask [B*num_heads, L, L] where 0 = can attend, -inf = cannot attend
        """
        B, L = attention_mask.shape

        # For self-attention with bidirectional mask:
        # Position i can attend to position j if j is a real token
        # attn_mask[b, i, j] = 0 if attention_mask[b, j] == 1, else -inf

        # Expand [B, L] -> [B, L, L]
        # attention_mask[b, j] determines if position j can be attended to
        key_mask = attention_mask.unsqueeze(1).expand(B, L, L)

        # Create mask: 0 where can attend, -inf where cannot
        attn_mask = torch.zeros_like(key_mask, dtype=torch.float)
        attn_mask = attn_mask.masked_fill(key_mask == 0, float("-inf"))

        # Expand for num_heads: [B, L, L] -> [B*num_heads, L, L]
        attn_mask = attn_mask.unsqueeze(1).expand(B, self.num_heads, L, L)
        attn_mask = attn_mask.reshape(B * self.num_heads, L, L)

        return attn_mask

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with attention mask for padding and mask indices.

        Args:
            tokens: Token IDs [B, L] (padded with any token, e.g., 0)
            attention_mask: [B, L] where 1 = real token, 0 = padding
            mask_indices: [B, L] bool, True for masked positions to predict

        Returns:
            Logits [B, L, vocab_size] for all positions
        """
        # Create input with mask tokens at masked positions
        input_tokens = tokens.clone()
        input_tokens[mask_indices] = self.mask_token_id

        # Get embeddings
        x = self.token_embed(input_tokens)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create 2D attention mask from padding mask
        attn_mask = self._create_attention_mask(attention_mask)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=attn_mask)

        # Output projection
        x = self.norm(x)
        logits: torch.Tensor = self.head(x)

        return logits

    def predict_masked(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_ratio: float = 0.15,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask tokens and predict them (for training).

        Only masks real tokens (where attention_mask == 1), never padding.

        Args:
            tokens: Token IDs [B, L] (padded)
            attention_mask: [B, L] where 1 = real, 0 = padding
            mask_ratio: Ratio of real tokens to mask

        Returns:
            Tuple of:
            - masked_logits: Predictions for masked positions [num_masked, vocab_size]
            - masked_targets: Ground truth tokens [num_masked]
            - mask: Boolean mask indicating masked positions [B, L]
        """
        B, L = tokens.shape
        device = tokens.device

        # Only mask real tokens, not padding
        real_token_mask = attention_mask.bool()

        # Random masking only within real tokens
        random_values = torch.rand(B, L, device=device)
        mask = (random_values < mask_ratio) & real_token_mask

        # Ensure at least one token is masked per sample (only for samples with real tokens)
        num_real_per_sample = real_token_mask.sum(dim=1)
        any_masked = mask.any(dim=1)
        needs_mask = ~any_masked & (num_real_per_sample > 0)

        if needs_mask.any():
            # For samples that need at least one mask, pick a random real token
            for i in range(B):
                if needs_mask[i]:
                    real_indices = torch.where(real_token_mask[i])[0]
                    if len(real_indices) > 0:
                        random_idx = real_indices[
                            torch.randint(len(real_indices), (1,), device=device)
                        ]
                        mask[i, random_idx] = True

        # Get predictions
        logits = self.forward(tokens, attention_mask, mask)

        # Extract predictions and targets for masked positions
        masked_logits = logits[mask]  # [num_masked, vocab_size]
        masked_targets = tokens[mask]  # [num_masked]

        return masked_logits, masked_targets, mask
