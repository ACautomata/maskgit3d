from typing import TypedDict

import torch


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
