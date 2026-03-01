from typing import Optional
"""
MaskGIT Transformer implementation.

Bidirectional Transformer for masked token prediction.
Uses BERT-style training where tokens are randomly masked and
the model learns to predict them.
"""
import torch
import torch.nn as nn


class PositionalEncoding3D(nn.Module):
    """
    3D Positional encoding for volumetric tokens.

    Creates learnable embeddings for (D, H, W) positions.
    """

    def __init__(self, num_tokens: int, embed_dim: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to input.

        Args:
            x: Input tokens [B, N, C] where N = D * H * W

        Returns:
            Tokens with positional information added
        """
        B = x.shape[0]
        # Repeat positional embeddings for each sample in batch
        pos_emb = self.pos_embed.repeat(B, 1, 1)
        return x + pos_emb


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
        """
        Forward pass through transformer block.

        Args:
            x: Input [B, N, C]
            mask: Optional attention mask

        Returns:
            Output [B, N, C]
        """
        # Self-attention with residual
        x = x + self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            attn_mask=mask,
        )[0]

        # Feedforward with residual
        x = x + self.mlp(self.norm2(x))

        return x


class MaskGITTransformer(nn.Module):
    """
    Bidirectional Transformer for MaskGIT.

    During training: Predicts masked tokens using bidirectional attention
    During inference: Uses iterative decoding with masking schedule
    """

    def __init__(
        self,
        vocab_size: int,  # codebook_size + 1 (for mask token)
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ):
        """
        Args:
            vocab_size: Size of vocabulary (codebook_size + 1 for mask)
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Token embedding (includes mask token at index 0)
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.token_embed.weight, std=0.02)

        # Mask token (learnable, used during inference)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)

        # Positional encoding (will be initialized based on actual sequence)
        self.pos_encoding: Optional[PositionalEncoding3D] = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _init_pos_encoding(self, seq_len: int, device: torch.device):
        """Initialize positional encoding if needed."""
        pos_enc: Optional[PositionalEncoding3D] = self.pos_encoding  # type: ignore[assignment]
        if pos_enc is None or pos_enc.shape[1] < seq_len:  # type: ignore[union-attr]
            pos_enc = PositionalEncoding3D(seq_len, self.hidden_size)
            self.pos_encoding = pos_enc.to(device)
        # Assert for type checker - pos_encoding is guaranteed non-None after this
        assert self.pos_encoding is not None

    def encode(
        self,
        tokens: torch.Tensor,
        return_logits: bool = True,
    ) -> torch.Tensor:
        """
        Encode tokens through transformer.

        Args:
            tokens: Token indices [B, N]
            return_logits: Whether to return logits or embeddings

        Returns:
            Logits [B, N, vocab_size] or embeddings [B, N, hidden_size]
        """
        B, N = tokens.shape
        device = tokens.device

        # Initialize pos encoding if needed
        self._init_pos_encoding(N, device)

        # Get token embeddings
        x = self.token_embed(tokens)

        # Add positional encoding
        assert self.pos_encoding is not None
        x = self.pos_encoding(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if return_logits:
            return self.head(x)
        return x

    def forward(
        self,
        tokens: torch.Tensor,
        mask_indices: torch.Tensor | None = None,
        mask_token_only: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with optional masking.

        Args:
            tokens: Token indices [B, N]
            mask_indices: Boolean mask indicating which positions to predict
            mask_token_only: If True, only compute loss on masked positions

        Returns:
            Logits [B, N, vocab_size]
        """
        B, N = tokens.shape
        device = tokens.device

        # If no mask specified, return all logits
        if mask_indices is None:
            return self.encode(tokens, return_logits=True)

        # Initialize pos encoding if needed
        self._init_pos_encoding(N, device)

        # Create input with mask tokens
        input_tokens = tokens.clone()

        # Replace masked positions with mask token
        mask_token_id = 0  # Assume mask token is at index 0
        input_tokens[mask_indices] = mask_token_id

        # Get embeddings
        x = self.token_embed(input_tokens)
        assert self.pos_encoding is not None
        x = self.pos_encoding(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.head(x)

        return logits

    def predict_masked(
        self,
        tokens: torch.Tensor,
        mask_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask tokens and predict them (for training).

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

        # Ensure at least one token is masked per sample
        for i in range(B):
            if not mask[i].any():
                mask[i, torch.randint(0, N, (1,))] = True

        # Get predictions
        logits = self.forward(tokens, mask_indices=mask)

        # Extract predictions and targets for masked positions
        masked_logits = logits[mask]  # [num_masked, vocab_size]
        targets = tokens[mask]  # [num_masked]

        return masked_logits, targets, mask


class MaskGITTransformerConfig:
    """Configuration class for MaskGITTransformer."""

    # Base config (similar to BERT-base)
    BASE = {
        "vocab_size": 1025,  # codebook_size + 1
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
    }

    # Large config (similar to BERT-large)
    LARGE = {
        "vocab_size": 1025,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
    }

    # Small config for faster training
    SMALL = {
        "vocab_size": 1025,
        "hidden_size": 384,
        "num_layers": 6,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
    }

    @classmethod
    def from_name(cls, name: str) -> dict:
        """Get config by name."""
        if name == "base":
            return cls.BASE.copy()
        elif name == "large":
            return cls.LARGE.copy()
        elif name == "small":
            return cls.SMALL.copy()
        else:
            raise ValueError(f"Unknown config: {name}. Available: base, large, small")
