"""Loss functions."""

from .mask_weighted_ce import MaskWeightedCrossEntropyLoss
from .perceptual_loss import PerceptualLoss
from .vq_perceptual_loss import VQPerceptualLoss

__all__ = ["MaskWeightedCrossEntropyLoss", "PerceptualLoss", "VQPerceptualLoss"]
