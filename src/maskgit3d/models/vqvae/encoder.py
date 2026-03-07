from collections.abc import Sequence

import torch
import torch.nn as nn
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import MaisiEncoder


class Encoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_channels: Sequence[int] = (64, 128, 256),
        num_res_blocks: Sequence[int] = (2, 2, 2),
        attention_levels: Sequence[bool] = (False, False, False),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_nonlocal_attn: bool = False,
        num_splits: int = 1,
        dim_split: int = 1,
        norm_float16: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = MaisiEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channels=num_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_nonlocal_attn,
            num_splits=num_splits,
            dim_split=dim_split,
            norm_float16=norm_float16,
            use_flash_attention=use_flash_attention,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
