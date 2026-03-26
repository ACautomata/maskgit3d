"""Tests for VQVAE model."""

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from maskgit3d.models.vqvae import VQVAE

CONFIG_DIR = str(Path(__file__).resolve().parents[4] / "src/maskgit3d/conf")


def test_vqvae_forward_pass():
    vqvae = VQVAE(
        in_channels=1,
        out_channels=1,
        latent_channels=256,
        num_embeddings=8192,
        embedding_dim=256,
    )

    x = torch.randn(2, 1, 32, 32, 32)
    x_recon, vq_loss = vqvae(x)

    assert x_recon.shape[0] == 2
    assert x_recon.shape[1] == 1
    assert x_recon.shape[2] == 32
    assert vq_loss.dim() == 0


def test_vqvae_encode_decode():
    vqvae = VQVAE(
        in_channels=1,
        out_channels=1,
        latent_channels=256,
        num_embeddings=100,
        embedding_dim=16,
    )

    x = torch.randn(1, 1, 16, 16, 16)
    z_q, vq_loss, indices, z_e = vqvae.encode(x)
    x_recon = vqvae.decode(z_q)

    assert z_q.shape[1] == 16
    assert z_e.shape[1] == 16
    assert indices.shape[0] == 1
    assert x_recon.shape == x.shape


def test_vqvae_decoder_attention_uses_declared_order() -> None:
    vqvae = VQVAE(
        in_channels=1,
        out_channels=1,
        latent_channels=64,
        num_embeddings=64,
        embedding_dim=64,
        num_channels=(32, 64, 64, 64),
        num_res_blocks=(1, 1, 1, 1),
        attention_levels=(False, False, False, True),
    )
    decoder_attention_shapes: list[tuple[int, int, int]] = []
    hooks = []

    for block in vqvae.decoder.decoder.blocks:
        if block.__class__.__name__ != "SpatialAttentionBlock":
            continue

        def capture_shape(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            del module, output
            spatial_shape = inputs[0].shape[2:5]
            decoder_attention_shapes.append(
                (int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2]))
            )

        hooks.append(block.register_forward_hook(capture_shape))

    x = torch.randn(1, 1, 32, 32, 32)
    try:
        vqvae(x)
    finally:
        for hook in hooks:
            hook.remove()

    assert decoder_attention_shapes
    assert decoder_attention_shapes == [(4, 4, 4)]


def test_default_vqvae_config_has_single_decoder_attention_block() -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="train")

    vqvae = instantiate(cfg.model)
    decoder_attention_blocks = [
        block
        for block in vqvae.decoder.decoder.blocks
        if block.__class__.__name__ == "SpatialAttentionBlock"
    ]

    assert len(decoder_attention_blocks) == 1
