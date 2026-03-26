from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class InContextSample:
    context_images: list[torch.Tensor]
    context_modality_ids: list[int]
    target_image: torch.Tensor
    target_modality_id: int
    mask_ratio: float | None = None


@dataclass
class PreparedInContextBatch:
    sequences: torch.Tensor
    attention_mask: torch.Tensor
    target_mask: torch.Tensor
    labels: torch.Tensor
    mask_ratios: torch.Tensor | None = None
    raw_samples: list[InContextSample] | None = None
