"""Tests for VariableLengthMaskGITTransformer."""

import torch

from maskgit3d.models.incontext import VariableLengthMaskGITTransformer


class TestVariableLengthMaskGITTransformer:
    """Tests for VariableLengthMaskGITTransformer."""

    def test_init(self):
        """Test initialization with default parameters."""
        model = VariableLengthMaskGITTransformer(
            vocab_size=1025,
            mask_token_id=1024,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
        )
        assert model.vocab_size == 1025
        assert model.mask_token_id == 1024
        assert model.hidden_size == 256
        assert model.num_layers == 4
        assert model.num_heads == 4

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        model = VariableLengthMaskGITTransformer(
            vocab_size=512,
            mask_token_id=511,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
            mlp_ratio=2.0,
            dropout=0.2,
            max_seq_len=4096,
        )
        assert model.vocab_size == 512
        assert model.mask_token_id == 511
        assert model.hidden_size == 128
        assert model.num_layers == 2
        assert model.num_heads == 2
        assert model.max_seq_len == 4096

    def test_forward_with_padding(self):
        """Test forward pass with padded sequences."""
        model = VariableLengthMaskGITTransformer(
            vocab_size=1025,
            mask_token_id=1024,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
        model.eval()

        batch_size = 2
        seq_len = 16
        tokens = torch.randint(0, 1024, (batch_size, seq_len))

        # Create attention mask with padding
        # First sequence: 10 real tokens, 6 padding
        # Second sequence: all real tokens
        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[0, :10] = 1  # 10 real tokens
        attention_mask[1, :] = 1  # all real

        # Mask some positions
        mask_indices = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask_indices[0, 2] = True
        mask_indices[1, 5] = True

        with torch.no_grad():
            logits = model.forward(tokens, attention_mask, mask_indices)

        assert logits.shape == (batch_size, seq_len, 1025)

    def test_attention_mask_creation(self):
        """Test that 1D attention mask is correctly converted to 3D."""
        model = VariableLengthMaskGITTransformer(
            vocab_size=1025,
            mask_token_id=1024,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )

        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )

        attn_mask = model._create_attention_mask(attention_mask)

        # Shape should be [B*num_heads, L, L]
        assert attn_mask.shape == (2 * 2, 5, 5)

        # Check that padding positions are -inf
        # First sequence: positions 3, 4 are padding -> -inf
        assert attn_mask[0, 0, 3] == float("-inf")
        assert attn_mask[0, 0, 4] == float("-inf")

        # Real positions should be 0
        assert attn_mask[0, 0, 0] == 0
        assert attn_mask[0, 0, 1] == 0
        assert attn_mask[0, 0, 2] == 0

        # Second sequence: all real -> all 0 (starts at index B*num_heads//2)
        assert (attn_mask[2, 0, :] == 0).all()

    def test_predict_masked_variable_lengths(self):
        """Test predict_masked with variable-length sequences."""
        model = VariableLengthMaskGITTransformer(
            vocab_size=1025,
            mask_token_id=1024,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
        model.eval()

        batch_size = 2
        seq_len = 16
        tokens = torch.randint(0, 1024, (batch_size, seq_len))

        # Variable lengths: first has 10 real tokens, second has 16
        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[0, :10] = 1
        attention_mask[1, :] = 1

        with torch.no_grad():
            masked_logits, targets, mask = model.predict_masked(
                tokens, attention_mask, mask_ratio=0.15
            )

        # Mask should have same shape as tokens
        assert mask.shape == (batch_size, seq_len)

        # Logits should be for masked positions only
        assert masked_logits.shape[0] == targets.shape[0]
        assert masked_logits.shape[1] == 1025

        # First sequence should only have masks in first 10 positions
        assert not mask[0, 10:].any()

        # At least one token masked per sequence
        assert mask[0, :10].sum() >= 1
        assert mask[1].sum() >= 1

    def test_padding_does_not_affect_output(self):
        """Test that padding positions don't affect real token outputs."""
        model = VariableLengthMaskGITTransformer(
            vocab_size=1025,
            mask_token_id=1024,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
        model.eval()

        # Same real tokens, different padding
        tokens1 = torch.randint(0, 1024, (1, 16))
        tokens1[0, :10] = torch.arange(10) + 1  # First 10 are deterministic

        tokens2 = tokens1.clone()
        tokens2[0, 10:] = torch.randint(100, 200, (6,))  # Different padding values

        attention_mask = torch.zeros(1, 16)
        attention_mask[0, :10] = 1  # Only first 10 are real

        mask_indices = torch.zeros(1, 16, dtype=torch.bool)
        mask_indices[0, 2] = True  # Mask position 2

        with torch.no_grad():
            logits1 = model.forward(tokens1, attention_mask, mask_indices)
            logits2 = model.forward(tokens2, attention_mask, mask_indices)

        # Outputs for real positions should be identical
        # (padding values should not affect attention)
        assert torch.allclose(logits1[0, :10], logits2[0, :10], atol=1e-5)

    def test_mask_handling(self):
        """Test that masked positions are replaced with mask token."""
        model = VariableLengthMaskGITTransformer(
            vocab_size=1025,
            mask_token_id=1024,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )

        batch_size = 1
        seq_len = 8
        tokens = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.ones(batch_size, seq_len)

        # Mask specific positions
        mask_indices = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask_indices[0, 1] = True
        mask_indices[0, 3] = True
        mask_indices[0, 5] = True

        logits = model.forward(tokens, attention_mask, mask_indices)

        assert logits.shape == (batch_size, seq_len, 1025)

    def test_predict_masked_only_masks_real_tokens(self):
        """Test that predict_masked only masks real tokens, never padding."""
        model = VariableLengthMaskGITTransformer(
            vocab_size=1025,
            mask_token_id=1024,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
        model.eval()

        batch_size = 4
        seq_len = 20
        tokens = torch.randint(0, 1024, (batch_size, seq_len))

        # Variable lengths
        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[0, :5] = 1  # 5 real tokens
        attention_mask[1, :10] = 1  # 10 real tokens
        attention_mask[2, :15] = 1  # 15 real tokens
        attention_mask[3, :] = 1  # all real tokens

        with torch.no_grad():
            _, _, mask = model.predict_masked(tokens, attention_mask, mask_ratio=0.5)

        # Check that no padding positions are masked
        for i in range(batch_size):
            real_positions = attention_mask[i].bool()
            padding_positions = ~real_positions

            # No padding should be masked
            assert not mask[i][padding_positions].any()

    def test_predict_masked_ensures_at_least_one_mask(self):
        """Test that predict_masked ensures at least one token is masked per sample."""
        model = VariableLengthMaskGITTransformer(
            vocab_size=1025,
            mask_token_id=1024,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
        model.eval()

        # Small sequence with low mask ratio might not mask anything by chance
        # Run multiple times to test the safeguard
        for _ in range(10):
            tokens = torch.randint(0, 1024, (4, 8))
            attention_mask = torch.ones(4, 8)  # All real tokens

            with torch.no_grad():
                _, _, mask = model.predict_masked(tokens, attention_mask, mask_ratio=0.01)

            # Each sample should have at least one masked token
            assert mask.sum(dim=1).min() >= 1

    def test_forward_all_padding_in_batch(self):
        """Test forward when some samples are all padding."""
        model = VariableLengthMaskGITTransformer(
            vocab_size=1025,
            mask_token_id=1024,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
        model.eval()

        batch_size = 2
        seq_len = 8
        tokens = torch.randint(0, 1024, (batch_size, seq_len))

        # First sample: all padding
        # Second sample: all real
        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[1, :] = 1

        mask_indices = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask_indices[1, 3] = True  # Only mask in the real sample

        with torch.no_grad():
            logits = model.forward(tokens, attention_mask, mask_indices)

        assert logits.shape == (batch_size, seq_len, 1025)

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        model = VariableLengthMaskGITTransformer(
            vocab_size=1025,
            mask_token_id=1024,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
        model.eval()

        # Test various sequence lengths
        for seq_len in [4, 16, 32, 64]:
            tokens = torch.randint(0, 1024, (2, seq_len))
            attention_mask = torch.ones(2, seq_len)
            attention_mask[0, seq_len // 2 :] = 0  # Half padding for first sample

            mask_indices = torch.zeros(2, seq_len, dtype=torch.bool)
            mask_indices[0, 1] = True
            mask_indices[1, seq_len - 1] = True

            with torch.no_grad():
                logits = model.forward(tokens, attention_mask, mask_indices)

            assert logits.shape == (2, seq_len, 1025)
