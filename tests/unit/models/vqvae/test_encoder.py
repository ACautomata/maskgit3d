import torch

from src.maskgit3d.models.vqvae.encoder import Encoder


def test_encoder_forward_reduces_spatial_shape() -> None:
    encoder = Encoder(
        spatial_dims=3,
        in_channels=1,
        out_channels=256,
        num_channels=(64, 128, 256),
    )

    x = torch.randn(2, 1, 32, 32, 32)
    z = encoder(x)

    assert z.shape[0] == 2
    assert z.shape[1] == 256
    assert z.shape[2] == 8
    assert z.shape[3] == 8
    assert z.shape[4] == 8
