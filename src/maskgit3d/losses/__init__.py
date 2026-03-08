"""Loss functions."""

from .perceptual_loss import PerceptualLoss
from .vq_perceptual_loss import VQPerceptualLoss

__all__ = ["PerceptualLoss", "VQPerceptualLoss"]
