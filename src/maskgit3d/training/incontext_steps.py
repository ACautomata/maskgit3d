"""Training steps for InContextMaskGIT model.

This module provides the training orchestration logic for the InContextMaskGIT
model, handling training, validation, and prediction steps.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from typing_extensions import TypedDict

from ..models.incontext.types import InContextSample


class InContextTrainingStepOutput(TypedDict):
    """Output from InContextTrainingSteps.training_step()."""

    loss: torch.Tensor


class InContextEvalStepOutput(TypedDict):
    """Output from InContextTrainingSteps validation/predict steps."""

    context_images: list[torch.Tensor]
    target_image: torch.Tensor
    generated_image: torch.Tensor


class InContextValidationStepOutput(InContextEvalStepOutput):
    """Output from InContextTrainingSteps.validation_step()."""

    loss: torch.Tensor


class InContextPredictStepOutput(InContextEvalStepOutput):
    """Output from InContextTrainingSteps.predict_step()."""

    pass


class InContextTrainingSteps:
    """Orchestrates training steps for InContextMaskGIT model.

    This class handles the training, validation, and prediction workflows
    for multi-modal in-context learning with 3D medical images.

    Args:
        model: InContextMaskGIT model instance.
        log_fn: Optional logging function for metrics.

    Attributes:
        model: The InContextMaskGIT model.
        log_fn: Optional logging function.
    """

    def __init__(
        self,
        model: Any,
        log_fn: Callable[..., None] | None = None,
    ) -> None:
        self.model = model
        self.log_fn = log_fn

    def training_step_any2one(
        self,
        samples: list[InContextSample],
    ) -> InContextTrainingStepOutput:
        """Training step for any2one batches with variable context per sample."""
        prepared = self.model.prepare_batch(samples)
        loss, _metrics = self.model.compute_loss_from_prepared(prepared)
        return {"loss": loss}

    def validation_step_any2one(
        self,
        samples: list[InContextSample],
    ) -> InContextValidationStepOutput:
        """Validation step for any2one batches."""
        prepared = self.model.prepare_batch(samples)
        loss, _metrics = self.model.compute_loss_from_prepared(prepared)

        target_image = samples[0].target_image
        B, C, D, H, W = target_image.shape

        with torch.no_grad():
            context_images = [s.context_images for s in samples]
            context_modality_ids = samples[0].context_modality_ids
            target_modality_id = samples[0].target_modality_id

            first_context = samples[0].context_images
            generated_image = self.model.generate(
                context_images=first_context,
                context_modality_ids=context_modality_ids,
                target_modality_id=target_modality_id,
                target_shape=(B, D, H, W),
            )

        return {
            "context_images": first_context,
            "target_image": target_image.detach().cpu(),
            "generated_image": generated_image.detach().cpu(),
            "loss": loss.detach(),
        }

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        encode_context_fn: Callable,
        encode_target_fn: Callable,
    ) -> InContextTrainingStepOutput:
        context_images = batch["context_images"]
        context_modality_ids = batch["context_modality_ids"]
        target_image = batch["target_image"]
        target_modality_id = batch["target_modality_id"]

        encoded_context = encode_context_fn(context_images, context_modality_ids)
        encoded_target = encode_target_fn(target_image, target_modality_id)

        loss, _metrics = self.model.compute_loss(
            context_images=encoded_context,
            context_modality_ids=context_modality_ids,
            target_image=encoded_target,
            target_modality_id=target_modality_id,
        )

        return {"loss": loss}

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        encode_context_fn: Callable,
        encode_target_fn: Callable,
    ) -> InContextValidationStepOutput:
        context_images = batch["context_images"]
        context_modality_ids = batch["context_modality_ids"]
        target_image = batch["target_image"]
        target_modality_id = batch["target_modality_id"]

        encoded_context = encode_context_fn(context_images, context_modality_ids)
        encoded_target = encode_target_fn(target_image, target_modality_id)

        loss, _metrics = self.model.compute_loss(
            context_images=encoded_context,
            context_modality_ids=context_modality_ids,
            target_image=encoded_target,
            target_modality_id=target_modality_id,
        )

        B, C, D, H, W = target_image.shape
        with torch.no_grad():
            generated_image = self.model.generate(
                context_images=encoded_context,
                context_modality_ids=context_modality_ids,
                target_modality_id=target_modality_id,
                target_shape=(B, D, H, W),
            )

        return {
            "context_images": [t.detach().cpu() for t in context_images],
            "target_image": target_image.detach().cpu(),
            "generated_image": generated_image.detach().cpu(),
            "loss": loss.detach(),
        }

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        encode_context_fn: Callable,
        encode_target_fn: Callable,
    ) -> InContextPredictStepOutput:
        context_images = batch["context_images"]
        context_modality_ids = batch["context_modality_ids"]
        target_image = batch["target_image"]
        target_modality_id = batch["target_modality_id"]

        encoded_context = encode_context_fn(context_images, context_modality_ids)

        B, C, D, H, W = target_image.shape
        with torch.no_grad():
            generated_image = self.model.generate(
                context_images=encoded_context,
                context_modality_ids=context_modality_ids,
                target_modality_id=target_modality_id,
                target_shape=(B, D, H, W),
            )

        return {
            "context_images": [t.detach().cpu() for t in context_images],
            "target_image": target_image.detach().cpu(),
            "generated_image": generated_image.detach().cpu(),
        }
