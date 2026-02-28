"""
PatchGAN Discriminator for VQGAN.

This discriminator is based on the Pix2Pix architecture and is used
for adversarial training of the VQVAE.
"""
import functools
from typing import Optional
import torch
import torch.nn as nn

from maskgit3d.domain.interfaces import DiscriminatorInterface


def weights_init(m):
    """Initialize network weights."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ActNorm(nn.Module):
    """Activation Normalization layer."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.initialized = False

    def initialize(self, x: torch.Tensor):
        """Initialize statistics from first batch."""
        with torch.no_grad():
            bias = -x.flatten(1).mean(1)
            var = ((x.flatten(1) - bias.unsqueeze(1)) ** 2).mean(1)
            weight = 1.0 / (torch.sqrt(var) + 1e-6)
            self.bias.data.copy_(bias)
            self.weight.data.copy_(weight)
        self.initialized = True

    def forward(self, x: torch.Tensor):
        if not self.initialized:
            self.initialize(x)
        return self.weight.unsqueeze(0).unsqueeze(2) * x + self.bias.unsqueeze(0).unsqueeze(2)


class NLayerDiscriminator(nn.Module, DiscriminatorInterface):
    """
    PatchGAN Discriminator.

    This discriminator evaluates patches of the image rather than
    the entire image, which allows it to capture high-frequency
    details better.
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_actnorm: bool = False,
    ):
        """
        Args:
            input_nc: Number of input channels
            ndf: Number of filters in the last layer
            n_layers: Number of convolutional layers
            use_actnorm: Whether to use ActNorm instead of BatchNorm
        """
        super().__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            norm_layer = ActNorm

        use_bias = norm_layer != nn.BatchNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=kw, stride=2, padding=padw, bias=use_bias
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=1, padding=padw, bias=use_bias
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Output: 1 channel prediction map
        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.main = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run discriminator forward pass.

        Args:
            x: Input volumes [B, C, D, H, W]

        Returns:
            Discrimination logits [B, 1, D', H', W']
        """
        return self.main(x)


class IdentityDiscriminator(nn.Module, DiscriminatorInterface):
    """
    Identity discriminator that always returns zeros.

    Useful for training VQVAE without adversarial loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
