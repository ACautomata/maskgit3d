"""Loss functions."""

from .gan_loss import GANLoss
from .vq_loss import VQLoss

__all__ = ["GANLoss", "VQLoss"]
