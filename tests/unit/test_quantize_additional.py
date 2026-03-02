"""Additional tests for quantize module to improve coverage."""

import pytest
import torch
import torch.nn as nn

from maskgit3d.infrastructure.vqgan.quantize import (
    EMAVectorQuantizer,
    VectorQuantizer,
    VectorQuantizer2,
)


class TestVectorQuantizerAdditional:
    """Additional tests for VectorQuantizer."""


    def test_get_codebook_entry_reshaped(self):
        """Test get_codebook_entry with reshaping."""
        quantizer = VectorQuantizer(n_embed=512, embed_dim=256)
        indices = torch.randint(0, 512, (128,))  # Flattened indices
        shape = (2, 4, 4, 4, 256)  # (B, D, H, W, C)

        z_q = quantizer.get_codebook_entry(indices, shape=shape)

        assert z_q.shape == (2, 256, 4, 4, 4)


class TestVectorQuantizer2Additional:
    """Additional tests for VectorQuantizer2."""

    def test_forward_non_legacy(self):
        """Test forward with legacy=False."""
        quantizer = VectorQuantizer2(n_embed=512, embed_dim=256, legacy=False)
        z = torch.randn(2, 256, 4, 4, 4)

        z_q, loss, info = quantizer(z)

        assert z_q.shape == z.shape
        assert isinstance(loss, torch.Tensor)

    def test_forward_sane_index_shape(self):
        """Test forward with sane_index_shape=True."""
        quantizer = VectorQuantizer2(n_embed=512, embed_dim=256, sane_index_shape=True)
        z = torch.randn(2, 256, 4, 4, 4)

        z_q, loss, info = quantizer(z)

        assert z_q.shape == z.shape
        assert info[2].shape == (2, 4, 4, 4)

    def test_get_codebook_entry_with_reshape(self):
        """Test get_codebook_entry with reshape."""
        quantizer = VectorQuantizer2(n_embed=512, embed_dim=256)
        indices = torch.randint(0, 512, (128,))
        shape = (2, 4, 4, 4, 256)

        z_q = quantizer.get_codebook_entry(indices, shape=shape)

        assert z_q.shape == (2, 256, 4, 4, 4)


class TestEMAVectorQuantizerAdditional:
    """Additional tests for EMAVectorQuantizer."""

    def test_forward_not_training(self):
        """Test forward in eval mode."""
        quantizer = EMAVectorQuantizer(n_embed=512, embed_dim=256)
        quantizer.eval()
        z = torch.randn(2, 256, 4, 4, 4)

        # Store initial embed_avg
        initial_embed_avg = quantizer.embed_avg.data.clone()

        z_q, loss, info = quantizer(z)

        assert z_q.shape == z.shape
        # EMA should not update in eval mode
        assert torch.allclose(quantizer.embed_avg, initial_embed_avg)

    def test_get_codebook_entry_reshaped(self):
        """Test get_codebook_entry with reshaping."""
        quantizer = EMAVectorQuantizer(n_embed=512, embed_dim=256)
        indices = torch.randint(0, 512, (128,))
        shape = (2, 4, 4, 4, 256)

        z_q = quantizer.get_codebook_entry(indices, shape=shape)

        assert z_q.shape == (2, 256, 4, 4, 4)
