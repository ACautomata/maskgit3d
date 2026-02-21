"""
VQGAN Infrastructure.

This module provides 3D VQGAN implementations including:
- 3D Encoder/Decoder architectures
- Vector quantization
- Discriminator
- Complete VQModel3D
"""
from maskgit3d.infrastructure.vqgan.quantize import (
    VectorQuantizer,
    VectorQuantizer2,
    EMAVectorQuantizer,
)
from maskgit3d.infrastructure.vqgan.discriminator import (
    NLayerDiscriminator,
    IdentityDiscriminator,
)
from maskgit3d.infrastructure.vqgan.encoder_decoder_3d import (
    Encoder3d,
    Decoder3d,
    get_encoder_decoder_config_3d,
)
from maskgit3d.infrastructure.vqgan.vqgan_model_3d import (
    VQModel3D,
)

__all__ = [
    # Quantizers
    "VectorQuantizer",
    "VectorQuantizer2",
    "EMAVectorQuantizer",
    # Discriminator
    "NLayerDiscriminator",
    "IdentityDiscriminator",
    # Encoder/Decoder (3D)
    "Encoder3d",
    "Decoder3d",
    "get_encoder_decoder_config_3d",
    # Model (3D)
    "VQModel3D",
]
