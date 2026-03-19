from __future__ import annotations

from typing import Literal, cast

from omegaconf import DictConfig

from ..config.schemas import MaskGITModelConfig, VQVAEModelConfig
from ..models.maskgit import MaskGIT
from ..models.vqvae import VQVAE


def create_vqvae_model(model_config: DictConfig) -> VQVAE:
    config = VQVAEModelConfig.from_conf(model_config)
    quantizer_type: Literal["vq", "fsq"] = cast(Literal["vq", "fsq"], config.quantizer_type)
    return VQVAE(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        latent_channels=config.latent_channels,
        num_embeddings=config.num_embeddings,
        embedding_dim=config.embedding_dim,
        num_channels=config.num_channels,
        num_res_blocks=config.num_res_blocks,
        attention_levels=config.attention_levels,
        commitment_cost=config.commitment_cost,
        quantizer_type=quantizer_type,
        fsq_levels=config.fsq_levels,
        num_splits=config.num_splits,
        dim_split=config.dim_split,
    )


def create_maskgit_model(model_config: DictConfig, vqvae: VQVAE) -> MaskGIT:
    config = MaskGITModelConfig.from_conf(model_config)
    vqvae.eval()
    vqvae.requires_grad_(False)
    return MaskGIT(
        vqvae=vqvae,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        gamma_type=config.gamma_type,
        num_iterations=config.num_iterations,
        temperature=config.temperature,
    )
