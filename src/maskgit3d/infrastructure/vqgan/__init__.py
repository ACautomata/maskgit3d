"""
VQVAE Infrastructure.

This module provides 3D VQVAE implementations including:
- 3D Encoder/Decoder architectures
- Vector quantization
- Discriminator
- Complete VQVAE model
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
from maskgit3d.infrastructure.vqgan.quantize import (
    EMAVectorQuantizer,
    VectorQuantizer,
    VectorQuantizer2,
)
from maskgit3d.infrastructure.vqgan.vqvae import (
    VQVAE,
    MaisiVQModel3D,
    get_maisi_vq_config,
    get_vqvae_config,
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
    # Model (VQVAE)
    "VQVAE",
    # Backward compatibility aliases
    "MaisiVQModel3D",
    "get_maisi_vq_config",
    "get_vqvae_config",
]
