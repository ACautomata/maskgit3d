from typing import TypedDict

import torch


class VQVAEStepOutputRequired(TypedDict):
    x_real: torch.Tensor
    x_recon: torch.Tensor
    vq_loss: torch.Tensor
    last_layer: torch.nn.Parameter | None


class VQVAEStepOutput(VQVAEStepOutputRequired, total=False):
    inference_time: float
    use_sliding_window: bool


class MaskGITLogData(TypedDict, total=False):
    correct: int
    total: int
    mask_ratio: float
    mask_acc: float


class MaskGITStepOutputRequired(TypedDict):
    loss: torch.Tensor


class MaskGITStepOutput(MaskGITStepOutputRequired, total=False):
    log_data: MaskGITLogData
