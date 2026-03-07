"""Tests for VectorQuantizer."""

import pytest
import torch
from src.maskgit3d.models.vqvae.quantizer import VectorQuantizer


def test_vector_quantizer_forward():
    """Test VectorQuantizer forward pass returns correct shapes."""
    num_embeddings = 8192
    embedding_dim = 256
    batch_size = 2
    spatial_dims = (4, 4, 4)

    quantizer = VectorQuantizer(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
    )

    # Input: (B, C, D, H, W)
    z = torch.randn(batch_size, embedding_dim, *spatial_dims)

    z_q, vq_loss, indices = quantizer(z)

    # Check shapes
    assert z_q.shape == z.shape, f"z_q shape mismatch: {z_q.shape} vs {z.shape}"
    assert indices.shape == (batch_size, *spatial_dims), f"indices shape mismatch"
    assert vq_loss.dim() == 0, "vq_loss should be scalar"


def test_vector_quantizer_straight_through_estimator():
    """Test that gradients flow through straight-through estimator."""
    quantizer = VectorQuantizer(num_embeddings=100, embedding_dim=16)
    z = torch.randn(1, 16, 2, 2, 2, requires_grad=True)

    z_q, vq_loss, _ = quantizer(z)
    loss = z_q.sum() + vq_loss
    loss.backward()

    assert z.grad is not None, "Gradient should flow through quantizer"


def test_vector_quantizer_decode_from_indices():
    """Test decoding from indices."""
    quantizer = VectorQuantizer(num_embeddings=100, embedding_dim=16)

    # Create some indices
    indices = torch.randint(0, 100, (2, 4, 4, 4))

    # Decode
    z_q = quantizer.decode_from_indices(indices)

    assert z_q.shape == (2, 16, 4, 4, 4), f"Decoded shape mismatch: {z_q.shape}"
