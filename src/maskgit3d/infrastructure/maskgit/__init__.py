"""
MaskGIT infrastructure module.

This module provides:
- MaskGITTransformer: Bidirectional Transformer for masked token prediction
- MaskGITSampler: Iterative decoding during inference
- MaskGITModel: Complete MaskGIT model combining VQGAN and Transformer
"""
from maskgit3d.infrastructure.maskgit.maskgit_model import MaskGITModel
from maskgit3d.infrastructure.maskgit.sampling import MaskGITSampler
from maskgit3d.infrastructure.maskgit.transformer import MaskGITTransformer

__all__ = [
    "MaskGITTransformer",
    "MaskGITSampler",
    "MaskGITModel",
]
