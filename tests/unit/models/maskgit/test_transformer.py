"""Tests for MaskGIT Transformer."""

import pytest
import torch

from maskgit3d.models.maskgit import MaskGITTransformer


class TestMaskGITTransformer:
    """Tests for MaskGITTransformer."""

    def test_init(self):
        model = MaskGITTransformer(
            vocab_size=1025,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
        )
        assert model.vocab_size == 1025
        assert model.hidden_size == 256
        assert model.num_layers == 4
        assert model.num_heads == 4

    def test_forward_basic(self):
        model = MaskGITTransformer(
            vocab_size=1025,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
        model.eval()

        batch_size = 2
        seq_len = 16
        tokens = torch.randint(0, 1024, (batch_size, seq_len))

        with torch.no_grad():
            logits = model.encode(tokens, return_logits=True)

        assert logits.shape == (batch_size, seq_len, 1025)

    def test_forward_with_mask(self):
        model = MaskGITTransformer(
            vocab_size=1025,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
        model.eval()

        batch_size = 2
        seq_len = 16
        tokens = torch.randint(0, 1024, (batch_size, seq_len))
        mask = torch.rand(batch_size, seq_len) < 0.5

        with torch.no_grad():
            logits = model.forward(tokens, mask_indices=mask)

        assert logits.shape == (batch_size, seq_len, 1025)

    def test_predict_masked(self):
        model = MaskGITTransformer(
            vocab_size=1025,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
        model.eval()

        batch_size = 2
        seq_len = 16
        tokens = torch.randint(0, 1024, (batch_size, seq_len))

        with torch.no_grad():
            masked_logits, targets, mask = model.predict_masked(tokens, mask_ratio=0.5)

        assert mask.shape == (batch_size, seq_len)
        assert masked_logits.shape[0] == targets.shape[0]
        assert masked_logits.shape[1] == 1025
        assert (mask.sum(dim=1) > 0).all()

    def test_mask_token_id_default(self):
        model = MaskGITTransformer(
            vocab_size=1025,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
        assert model.mask_token_id == 1024

    def test_mask_token_id_custom(self):
        model = MaskGITTransformer(
            vocab_size=1025,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
            mask_token_id=500,
        )
        assert model.mask_token_id == 500

    def test_mask_token_id_invalid(self):
        with pytest.raises(ValueError, match="out of range"):
            MaskGITTransformer(
                vocab_size=1025,
                hidden_size=128,
                num_layers=2,
                num_heads=2,
                mask_token_id=2000,
            )
