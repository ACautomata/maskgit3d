"""Loss functions."""

from .gan_loss import GANLoss
from .perceptual_loss import PerceptualLoss
from .vq_loss import VQLoss

__all__ = ["GANLoss", "PerceptualLoss", "VQLoss"]
