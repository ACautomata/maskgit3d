from omegaconf import OmegaConf

from maskgit3d.models.maskgit import MaskGIT
from maskgit3d.models.vqvae import VQVAE
from maskgit3d.runtime.model_factory import create_maskgit_model, create_vqvae_model


def test_create_vqvae_model_builds_vqvae_from_config() -> None:
    config = OmegaConf.create(
        {
            "_target_": "maskgit3d.models.vqvae.VQVAE",
            "in_channels": 1,
            "out_channels": 1,
            "latent_channels": 64,
            "num_embeddings": 32,
            "embedding_dim": 16,
            "num_channels": [32, 64],
            "num_res_blocks": [1, 1],
            "attention_levels": [False, True],
            "commitment_cost": 0.125,
            "quantizer_type": "vq",
            "fsq_levels": [8, 8, 8, 5, 5, 5],
            "num_splits": 1,
            "dim_split": 0,
        }
    )

    model = create_vqvae_model(config)

    assert isinstance(model, VQVAE)
    assert model.in_channels == 1
    assert model.quant_conv.in_channels == 64
    assert model.quant_conv.out_channels == 16
    assert model.quantizer.num_embeddings == 32


def test_create_maskgit_model_freezes_vqvae_and_uses_config() -> None:
    vqvae = VQVAE(num_channels=(64, 128), num_res_blocks=(1, 1), attention_levels=(False, False))
    config = OmegaConf.create(
        {
            "_target_": "maskgit3d.models.maskgit.MaskGIT",
            "hidden_size": 96,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "gamma_type": "cosine",
            "num_iterations": 4,
            "temperature": 0.7,
        }
    )

    model = create_maskgit_model(config, vqvae)

    assert isinstance(model, MaskGIT)
    assert model.vqvae is vqvae
    assert model.hidden_size == 96
    assert model.sampler.num_iterations == 4
    assert model.sampler.temperature == 0.7
    assert not model.vqvae.training
    assert all(not parameter.requires_grad for parameter in model.vqvae.parameters())
