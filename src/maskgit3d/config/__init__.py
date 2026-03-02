"""Configuration layer - Dependency injection modules for maskgit3d."""

from maskgit3d.config.modules import (
    DataModule,
    InferenceModule,
    MaisiVQModelModule,
    MaisiVQModule,
    MaskGITModelModule,
    MaskGITModule,
    ModelModule,
    SystemModule,
    TrainingModule,
    VQGANModelModule,
    VQGANModule,
    create_fabric_pipeline,
    create_maisi_vq_module,
    create_maskgit_module,
    create_vqgan_module,
)

__all__ = [
    "DataModule",
    "InferenceModule",
    "MaisiVQModelModule",
    "MaisiVQModule",
    "MaskGITModelModule",
    "MaskGITModule",
    "ModelModule",
    "SystemModule",
    "TrainingModule",
    "VQGANModelModule",
    "VQGANModule",
    "create_fabric_pipeline",
    "create_maisi_vq_module",
    "create_maskgit_module",
    "create_vqgan_module",
]
