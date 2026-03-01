"""
VQGAN Infrastructure.

This module provides 3D VQGAN implementations including:
- 3D Encoder/Decoder architectures
- Vector quantization
- Discriminator
- Complete VQModel3D
- MAISI-based VQGAN model
"""
from maskgit3d.infrastructure.vqgan.discriminator import (
    IdentityDiscriminator,
    NLayerDiscriminator,
)
from maskgit3d.infrastructure.vqgan.encoder_decoder_3d import (
    Decoder3d,
    Encoder3d,
    get_encoder_decoder_config_3d,
)
from maskgit3d.infrastructure.vqgan.maisi_vq_model import (
    MaisiVQModel3D,
    get_maisi_vq_config,
)
from maskgit3d.infrastructure.vqgan.quantize import (
    EMAVectorQuantizer,
    VectorQuantizer,
    VectorQuantizer2,
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
    # MAISI Model (3D)
    "MaisiVQModel3D",
    "get_maisi_vq_config",
]
