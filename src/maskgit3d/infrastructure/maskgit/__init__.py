"""
MaskGIT infrastructure module.

This module provides:
- MaskGITTransformer: Bidirectional Transformer for masked token prediction
- MaskGITSampler: Iterative decoding during inference
- MaskGITModel: Complete MaskGIT model combining VQGAN and Transformer
- TrainingMaskScheduler: Dynamic mask ratio scheduling for training
"""

from maskgit3d.infrastructure.maskgit.maskgit_model import MaskGITModel
from maskgit3d.infrastructure.maskgit.sampling import MaskGITSampler
from maskgit3d.infrastructure.maskgit.scheduling import TrainingMaskScheduler
from maskgit3d.infrastructure.maskgit.transformer import MaskGITTransformer

__all__ = [
    "MaskGITTransformer",
    "MaskGITSampler",
    "MaskGITModel",
    "TrainingMaskScheduler",
]
