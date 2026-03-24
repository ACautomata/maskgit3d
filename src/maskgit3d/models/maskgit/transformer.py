"""MaskGIT Transformer implementation.

Bidirectional Transformer for masked token prediction.
Uses BERT-style training where tokens are randomly masked and
the model learns to predict them.
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for variable-length sequences.

    Computes positional embeddings on-the-fly using sine and cosine functions.
    Supports arbitrary sequence lengths without pre-allocated parameters.
    """

    pe: torch.Tensor  # type: ignore[assignment]

    def __init__(self, embed_dim: int, max_len: int = 8192):
        """Initialize sinusoidal positional encoding.

        Args:
            embed_dim: Embedding dimension (must match hidden_size)
            max_len: Maximum sequence length to precompute (for caching)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input.

        Args:
            x: Input tokens [B, N, C] where N is sequence length

        Returns:
            Tokens with positional information added
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            # Extend positional encoding dynamically if needed
            return self._forward_extended(x, seq_len)
        return x + self.pe[:, :seq_len, :]

    def _forward_extended(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Compute positional encoding for sequences longer than max_len."""
        position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float, device=x.device)
            * (-math.log(10000.0) / self.embed_dim)
        )
        pe = torch.zeros(seq_len, self.embed_dim, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(0)


class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention and feedforward."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Layer norm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feedforward
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through transformer block.

        Args:
            x: Input [B, N, C]
            mask: Optional attention mask

        Returns:
            Output [B, N, C]
        """
        # Self-attention with residual
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, attn_mask=mask)[0]

        # Feedforward with residual
        x = x + self.mlp(self.norm2(x))

        return x


class MaskGITTransformer(nn.Module):
    """Bidirectional Transformer for MaskGIT.

    During training: Predicts masked tokens using bidirectional attention
    During inference: Uses iterative decoding with masking schedule
    """

    def __init__(
        self,
        vocab_size: int,  # codebook_size + 1 (for mask token)
        mask_token_id: int | None = None,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """Args:
        vocab_size: Size of vocabulary (codebook_size + 1 for mask)
        mask_token_id: Token index reserved for mask token
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        dropout: Dropout probability
        """
        super().__init__()
        self.vocab_size = vocab_size
        if mask_token_id is None:
            mask_token_id = vocab_size - 1
        if mask_token_id < 0 or mask_token_id >= vocab_size:
            raise ValueError(
                f"mask_token_id={mask_token_id} out of range for vocab_size={vocab_size}"
            )
        self.mask_token_id = mask_token_id
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.token_embed.weight, std=0.02)

        # Mask token (learnable, used during inference)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)

        # Sinusoidal positional encoding (supports arbitrary sequence lengths)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

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

    def encode(
        self,
        tokens: torch.Tensor,
        return_logits: bool = True,
    ) -> torch.Tensor:
        """Encode tokens through transformer.

        Args:
            tokens: Token indices [B, N]
            return_logits: Whether to return logits or embeddings

        Returns:
            Logits [B, N, vocab_size] or embeddings [B, N, hidden_size]
        """
        # Get token embeddings
        x = self.token_embed(tokens)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if return_logits:
            logits_out: torch.Tensor = self.head(x)
            return logits_out
        output: torch.Tensor = x
        return output

    def forward(
        self,
        tokens: torch.Tensor,
        mask_indices: torch.Tensor | None = None,
        mask_token_only: bool = True,
    ) -> torch.Tensor:
        """Forward pass with optional masking.

        Args:
            tokens: Token indices [B, N]
            mask_indices: Boolean mask indicating which positions to predict
            mask_token_only: If True, only compute loss on masked positions

        Returns:
            Logits [B, N, vocab_size]
        """
        # If no mask specified, return all logits
        if mask_indices is None:
            return self.encode(tokens, return_logits=True)

        # Create input with mask tokens
        input_tokens = tokens.clone()
        input_tokens[mask_indices] = self.mask_token_id

        # Get embeddings
        x = self.token_embed(input_tokens)
        x = self.pos_encoding(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits: torch.Tensor = self.head(x)

        return logits

    def predict_masked(
        self,
        tokens: torch.Tensor,
        mask_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask tokens and predict them (for training).

        Args:
            tokens: Token indices [B, N]
            mask_ratio: Ratio of tokens to mask

        Returns:
            Tuple of (logits, targets, mask)
            - logits: Predictions for masked positions [num_masked, vocab_size]
            - targets: Ground truth tokens for masked positions [num_masked]
            - mask: Boolean mask indicating masked positions [B, N]
        """
        B, N = tokens.shape
        device = tokens.device

        # Random masking
        mask = torch.rand(B, N, device=device) < mask_ratio

        # Ensure at least one token is masked per sample (vectorized)
        any_masked = mask.any(dim=1)
        if not any_masked.all():
            needs_mask = ~any_masked
            random_indices = torch.randint(0, N, (int(needs_mask.sum().item()),), device=device)
            mask[needs_mask, random_indices] = True

        # Get predictions
        logits = self.forward(tokens, mask_indices=mask)

        # Extract predictions and targets for masked positions
        masked_logits = logits[mask]  # [num_masked, vocab_size]
        targets = tokens[mask]  # [num_masked]

        return masked_logits, targets, mask
