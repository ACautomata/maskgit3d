import torch

from maskgit3d.models.discriminator.patch_discriminator import PatchDiscriminator3D


def test_patch_discriminator_forward() -> None:
    discriminator = PatchDiscriminator3D(
        in_channels=1,
        ndf=64,
        n_layers=3,
    )

    x = torch.randn(2, 1, 32, 32, 32)
    output = discriminator(x)

    assert isinstance(output, list), "Output should be a list"
    assert len(output) > 0, "Output should not be empty"
    assert all(isinstance(item, tuple) for item in output), "Each output should be a tuple"
    assert all(len(item) == 2 for item in output), "Each tuple should contain 2 elements"
