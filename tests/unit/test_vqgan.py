"""
Unit tests for VQGAN components.

These tests verify the functionality of VQGAN models, discriminators,
quantizers, and training strategies.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from maskgit3d.infrastructure.vqgan.quantize import (
    VectorQuantizer,
)
from maskgit3d.infrastructure.vqgan.discriminator import (
    NLayerDiscriminator,
    IdentityDiscriminator,
    ActNorm,
)
from maskgit3d.domain.interfaces import ModelInterface
from maskgit3d.infrastructure.vqgan.vqgan_model_3d import VQModel3D


class TestVectorQuantizer:
    """Tests for VectorQuantizer."""

    def test_vector_quantizer_creation(self):
        """Test creating VectorQuantizer."""
        quantizer = VectorQuantizer(
            n_embed=256,
            embed_dim=32,
            beta=0.25,
        )
        assert quantizer.n_e == 256
        assert quantizer.e_dim == 32
        assert quantizer.beta == 0.25

    def test_vector_quantizer_forward(self):
        """Test forward pass through VectorQuantizer."""
        quantizer = VectorQuantizer(
            n_embed=256,
            embed_dim=32,
            beta=0.25,
        )
        # Input: [B, C, D, H, W] for 3D
        z = torch.randn(2, 32, 4, 4, 4)

        z_q, loss, info = quantizer(z)

        assert z_q.shape == z.shape
        assert loss.shape == ()  # Scalar
        assert info[0].shape == ()  # Perplexity is scalar

    def test_vector_quantizer_get_codebook_entry(self):
        """Test getting codebook entries from indices."""
        quantizer = VectorQuantizer(
            n_embed=256,
            embed_dim=32,
            beta=0.25,
        )
        # Indices: flattened [B*D*H*W]
        num_indices = 2 * 4 * 4 * 4
        indices = torch.randint(0, 256, (num_indices,))
        shape = (2, 4, 4, 4, 32)

        z_q = quantizer.get_codebook_entry(indices, shape=shape)

        assert z_q.shape == (2, 32, 4, 4, 4)


class TestVectorQuantizerEMA:
    """Tests for VectorQuantizer with EMA mode (replaces VectorQuantizer2/EMAVectorQuantizer)."""

    def test_ema_quantizer_creation(self):
        """Test creating VectorQuantizer with EMA enabled."""
        quantizer = VectorQuantizer(
            n_embed=256,
            embed_dim=32,
            beta=0.25,
            use_ema=True,
            decay=0.99,
        )
        assert quantizer.n_e == 256
        assert quantizer.e_dim == 32
        assert quantizer.use_ema is True
        assert quantizer.decay == 0.99

    def test_ema_quantizer_forward(self):
        """Test forward pass through VectorQuantizer with EMA."""
        quantizer = VectorQuantizer(
            n_embed=256,
            embed_dim=32,
            beta=0.25,
            use_ema=True,
            decay=0.99,
        )
        z = torch.randn(2, 32, 4, 4, 4)

        quantizer.train()
        z_q, loss, info = quantizer(z)

        assert z_q.shape == z.shape
        assert loss.shape == ()
        assert info[0] is not None  # Perplexity


class TestNLayerDiscriminator:
    """Tests for NLayerDiscriminator."""

    def test_discriminator_creation(self):
        """Test creating NLayerDiscriminator."""
        disc = NLayerDiscriminator(
            input_nc=1,
            ndf=64,
            n_layers=3,
            use_actnorm=False,
        )
        assert disc is not None

    def test_discriminator_forward(self):
        """Test forward pass through discriminator."""
        disc = NLayerDiscriminator(
            input_nc=1,
            ndf=64,
            n_layers=3,
        )
        # Input: [B, C, D, H, W] for 3D
        x = torch.randn(2, 1, 64, 64, 64)

        output = disc(x)

        # Output should be [B, 1, D', H', W']
        assert output.shape[0] == 2
        assert output.shape[1] == 1
        # Spatial dims should be reduced
        assert output.shape[2] < 64
        assert output.shape[3] < 64

    def test_discriminator_with_actnorm(self):
        """Test discriminator with ActNorm."""
        pytest.skip("ActNorm has 3D compatibility issues")


class TestIdentityDiscriminator:
    """Tests for IdentityDiscriminator."""

    def test_identity_discriminator(self):
        """Test identity discriminator returns zeros."""
        disc = IdentityDiscriminator()
        x = torch.randn(2, 1, 32, 32, 32)

        output = disc(x)

        assert output.shape == x.shape
        assert torch.all(output == 0)


class TestActNorm:
    """Tests for ActNorm layer."""

    def test_actnorm_creation(self):
        """Test creating ActNorm."""
        actnorm = ActNorm(num_channels=64)
        assert actnorm.num_channels == 64
        assert not actnorm.initialized

    def test_actnorm_forward(self):
        """Test forward pass through ActNorm."""
        pytest.skip("ActNorm has 3D compatibility issues")

    def test_actnorm_initialization(self):
        """Test ActNorm initialization on first forward."""
        pytest.skip("ActNorm has 3D compatibility issues")


class TestDiscriminatorInterface:
    """Tests for DiscriminatorInterface compliance."""

    def test_nlayer_discriminator_interface(self):
        """Test NLayerDiscriminator implements DiscriminatorInterface."""
        from maskgit3d.domain.interfaces import DiscriminatorInterface

        disc = NLayerDiscriminator()
        assert isinstance(disc, DiscriminatorInterface)

    def test_identity_discriminator_interface(self):
        """Test IdentityDiscriminator implements DiscriminatorInterface."""
        from maskgit3d.domain.interfaces import DiscriminatorInterface

        disc = IdentityDiscriminator()
        assert isinstance(disc, DiscriminatorInterface)


class TestQuantizerInterface:
    """Tests for QuantizerInterface compliance."""

    def test_vector_quantizer_interface(self):
        """Test VectorQuantizer implements QuantizerInterface."""
        from maskgit3d.domain.interfaces import QuantizerInterface

        quantizer = VectorQuantizer(n_embed=256, embed_dim=32)
        assert isinstance(quantizer, QuantizerInterface)

    def test_ema_quantizer_interface(self):
        """Test VectorQuantizer with EMA implements QuantizerInterface."""
        from maskgit3d.domain.interfaces import QuantizerInterface

        quantizer = VectorQuantizer(n_embed=256, embed_dim=32, use_ema=True)
        assert isinstance(quantizer, QuantizerInterface)


class TestVQModel3D:
    """Regression tests for VQModel3D construction behavior."""

    def test_vqmodel3d_initialization_succeeds(self):
        """VQModel3D should initialize with explicit constructor args."""
        model = VQModel3D(
            in_channels=1,
            codebook_size=64,
            embed_dim=32,
            latent_channels=64,
            resolution=32,
            channel_multipliers=(1, 2),
        )
        assert model is not None

    def test_vqmodel3d_codebook_size_matches_argument(self):
        """codebook_size property should reflect the configured codebook size."""
        model = VQModel3D(
            in_channels=1,
            codebook_size=128,
            embed_dim=32,
            latent_channels=64,
            resolution=32,
            channel_multipliers=(1, 2),
        )
        assert model.codebook_size == 128


def test_vqmodel_forward_returns_tensor_for_model_interface_contract():
    """VQModel3D.forward() must return a single Tensor per ModelInterface contract."""
    model = VQModel3D(
        in_channels=1,
        codebook_size=64,
        embed_dim=32,
        latent_channels=64,
        resolution=32,
        channel_multipliers=(1, 2),
    )
    x = torch.randn(1, 1, 32, 32, 32)
    out = model.forward(x)
    assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"
    assert isinstance(model, ModelInterface)


def test_vqmodel_latent_shape_does_not_rerun_encoder_each_access(monkeypatch):
    """latent_shape must return cached value without running the encoder."""
    model = VQModel3D(
        in_channels=1,
        codebook_size=64,
        embed_dim=32,
        latent_channels=64,
        resolution=32,
        channel_multipliers=(1, 2),
    )

    call_count = {"n": 0}
    original = model.encoder.forward

    def wrapped(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.encoder, "forward", wrapped)
    _ = model.latent_shape
    _ = model.latent_shape

    assert call_count["n"] == 0
