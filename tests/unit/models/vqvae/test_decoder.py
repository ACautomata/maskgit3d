import importlib

import torch


def test_decoder_forward_expands_spatial_dims_by_16x() -> None:
    decoder_module = importlib.import_module("maskgit3d.models.vqvae.decoder")
    decoder_cls = decoder_module.Decoder

    in_channels = 256
    out_channels = 1

    decoder = decoder_cls(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
    )

    z = torch.randn(2, in_channels, 2, 2, 2)
    x_recon = decoder(z)

    assert x_recon.shape == (2, out_channels, 32, 32, 32)
    assert x_recon.shape[2] // z.shape[2] == 16
    assert x_recon.shape[3] // z.shape[3] == 16
    assert x_recon.shape[4] // z.shape[4] == 16
