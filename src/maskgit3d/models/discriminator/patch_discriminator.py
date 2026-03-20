from __future__ import annotations

import torch
import torch.nn as nn


class PatchDiscriminator3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: str = "instance",
    ) -> None:
        super().__init__()
        use_bias = norm_layer == "instance"

        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        mult = 1
        for layer_idx in range(1, n_layers):
            prev_mult = mult
            mult = min(2**layer_idx, 8)
            layers.extend(
                [
                    nn.Conv3d(
                        ndf * prev_mult,
                        ndf * mult,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=use_bias,
                    ),
                    self._get_norm_layer(ndf * mult, norm_layer),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        prev_mult = mult
        mult = min(2**n_layers, 8)
        layers.extend(
            [
                nn.Conv3d(
                    ndf * prev_mult,
                    ndf * mult,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                ),
                self._get_norm_layer(ndf * mult, norm_layer),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(ndf * mult, 1, kernel_size=4, stride=1, padding=1),
            ]
        )

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_norm_layer(self, num_features: int, norm_type: str) -> nn.Module:
        if norm_type == "instance":
            return nn.InstanceNorm3d(num_features, affine=False)
        if norm_type == "batch":
            return nn.BatchNorm3d(num_features)
        raise ValueError(f"Unknown norm layer: {norm_type}")

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, None]]:
        output = self.model(x)
        return [(output, None)]
