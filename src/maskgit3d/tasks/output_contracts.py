from dataclasses import dataclass, field
from typing import TypedDict

import torch


# =============================================================================
# NEW OUTPUT CONTRACTS (Phase 1 - Refactor)
# =============================================================================


class VQVAETrainingStepOutput(TypedDict):
    """Output from VQVAETask.training_step().

    Returns a dictionary containing:
        - loss: Scalar loss tensor for backward pass
        - loss_g: Generator loss (detached)
        - loss_d: Discriminator loss (detached)
        - nll_loss, rec_loss, p_loss, g_loss, vq_loss: Loss components (detached)
        - disc_loss: Discriminator loss (detached)
    """

    loss: torch.Tensor
    loss_g: torch.Tensor
    loss_d: torch.Tensor
    nll_loss: torch.Tensor
    rec_loss: torch.Tensor
    p_loss: torch.Tensor
    g_loss: torch.Tensor
    vq_loss: torch.Tensor
    disc_loss: torch.Tensor


class VQVAEEvalStepOutput(TypedDict):
    """Output from VQVAETask validation/test/predict steps.

    Returns only raw data without any computed metrics.
    """

    x_real: torch.Tensor
    x_recon: torch.Tensor


class MaskGITTrainingStepOutput(TypedDict):
    """Output from MaskGITTask.training_step().

    Returns a dictionary containing:
        - loss: Scalar loss tensor for backward pass
    """

    loss: torch.Tensor


class MaskGITEvalStepOutput(TypedDict):
    """Output from MaskGITTask validation/test steps.

    Returns raw data and predictions without computed metrics.
    """

    x_real: torch.Tensor
    generated_images: torch.Tensor
    masked_logits: torch.Tensor
    masked_targets: torch.Tensor


# =============================================================================
# LEGACY DATACLASSES (kept for backwards compatibility during transition)
# =============================================================================


@dataclass
class VQVAETrainingOutput:
    """Output dataclass for VQVAE training step callback payload."""

    x_real: torch.Tensor
    x_recon: torch.Tensor
    vq_loss: torch.Tensor
    last_layer: torch.nn.Parameter | None = None


@dataclass
class VQVAEValidationOutput:
    """Output dataclass for VQVAE validation/test step."""

    x_real: torch.Tensor
    x_recon: torch.Tensor
    vq_loss: torch.Tensor
    inference_time: float | None = field(default=None)
    use_sliding_window: bool | None = field(default=None)


@dataclass
class MaskGITTrainingOutput:
    """Output dataclass for MaskGIT training step callback payload."""

    tokens: torch.Tensor
    masked_logits: torch.Tensor
    masked_targets: torch.Tensor
    mask_ratio: float


@dataclass
class MaskGITValidationOutput:
    """Output dataclass for MaskGIT validation/test step."""

    x_real: torch.Tensor
    generated_images: torch.Tensor
    masked_logits: torch.Tensor
    masked_targets: torch.Tensor
    mask_ratio: float
    token_shape: torch.Size | None = field(default=None)


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
