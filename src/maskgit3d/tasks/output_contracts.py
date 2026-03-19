from typing import TypedDict

import torch


class VQVAEStepOutputRequired(TypedDict):
    x_real: torch.Tensor
    x_recon: torch.Tensor
    vq_loss: torch.Tensor


class VQVAEStepOutput(VQVAEStepOutputRequired, total=False):
    last_layer: torch.nn.Parameter | None
    inference_time: float
    use_sliding_window: bool


class MaskGITStepOutputRequired(TypedDict):
    x_real: torch.Tensor
    generated_images: torch.Tensor


class MaskGITStepOutput(MaskGITStepOutputRequired, total=False):
    masked_logits: torch.Tensor
    masked_targets: torch.Tensor
    mask_ratio: float
    input_shape: torch.Size
    token_shape: torch.Size
