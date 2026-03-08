"""Tests for MaskGIT generate method."""

import pytest
import torch

from src.maskgit3d.models.maskgit import MaskGIT
from src.maskgit3d.models.vqvae import VQVAE


@pytest.fixture
def vqvae():
    model = VQVAE(in_channels=1, out_channels=1, latent_channels=64, num_embeddings=100, embedding_dim=64)
    model.eval()
    model.requires_grad_(False)
    return model


def test_maskgit_generate_basic(vqvae):
    maskgit = MaskGIT(vqvae=vqvae, hidden_size=128, num_layers=2, num_heads=4)
    maskgit.eval()

    with torch.no_grad():
        output = maskgit.generate(shape=(1, 2, 2, 2), num_iterations=8)

    assert output is not None
    assert output.shape[0] == 1


def test_maskgit_generate_different_shapes(vqvae):
    maskgit = MaskGIT(vqvae=vqvae, hidden_size=128, num_layers=2, num_heads=4)
    maskgit.eval()

    shapes = [(1, 2, 2, 2), (2, 2, 2, 2), (1, 3, 3, 3)]

    for shape in shapes:
        with torch.no_grad():
            output = maskgit.generate(shape=shape, num_iterations=8)
        assert output is not None


def test_maskgit_generate_with_temperature(vqvae):
    maskgit = MaskGIT(vqvae=vqvae, hidden_size=128, num_layers=2, num_heads=4)
    maskgit.eval()

    with torch.no_grad():
        output = maskgit.generate(shape=(1, 2, 2, 2), num_iterations=8, temperature=1.5)

    assert output is not None
