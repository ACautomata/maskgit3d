"""Unit tests for PerceptualLoss."""
import torch

from src.maskgit3d.losses.perceptual_loss import PerceptualLoss


def test_perceptual_loss_init_default_params() -> None:
    """Test initialization with default parameters."""
    loss = PerceptualLoss()
    assert loss is not None


def test_perceptual_loss_forward_3d_input() -> None:
    """Test forward pass with 3D input."""
    loss = PerceptualLoss(network="alex", spatial_dims=3)
    pred = torch.randn(2, 1, 32, 32, 32)
    target = torch.randn(2, 1, 32, 32, 32)

    result = loss(pred, target)

    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0  # Scalar loss


def test_perceptual_loss_forward_2d_input() -> None:
    """Test forward pass with 2D input."""
    loss = PerceptualLoss(network="alex", spatial_dims=2)
    pred = torch.randn(2, 1, 32, 32)
    target = torch.randn(2, 1, 32, 32)

    result = loss(pred, target)

    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0


def test_perceptual_loss_disabled_returns_zero() -> None:
    """Test that disabled loss returns zero."""
    loss = PerceptualLoss(enabled=False)
    pred = torch.randn(2, 1, 32, 32, 32)
    target = torch.randn(2, 1, 32, 32, 32)

    result = loss(pred, target)

    assert result.item() == 0.0


def test_perceptual_loss_different_networks() -> None:
    """Test initialization with different network backbones."""
    for network in ["alex", "vgg"]:
        loss = PerceptualLoss(network=network, spatial_dims=3)
        assert loss is not None
