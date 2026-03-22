from __future__ import annotations

from typing import Any, Callable, cast

import torch
from omegaconf import OmegaConf

from ..tasks.output_contracts import MaskGITEvalStepOutput, MaskGITTrainingStepOutput


class MaskGITTrainingSteps:
    def __init__(
        self,
        maskgit: Any,
        log_fn: Callable[..., None] | None = None,
    ) -> None:
        self.maskgit = maskgit
        self.log_fn = log_fn

    def extract_input_tensor(
        self, batch: torch.Tensor | tuple[Any, ...] | list[Any]
    ) -> torch.Tensor:
        if isinstance(batch, tuple | list):
            return cast(torch.Tensor, batch[0])
        return batch

    def _get_masked_predictions_for_training(
        self, tokens: torch.Tensor, mask_ratio: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Compute masked predictions and loss for training.

        Returns:
            Tuple of (loss, masked_logits, masked_targets, mask_ratio)
        """
        import torch.nn.functional as F

        batch_size = tokens.shape[0]
        tokens_flat = tokens.view(batch_size, -1)
        effective_mask_ratio = (
            self.maskgit.mask_scheduler.sample_mask_ratio() if mask_ratio is None else mask_ratio
        )
        masked_logits, masked_targets, _ = self.maskgit.transformer.predict_masked(
            tokens_flat,
            mask_ratio=effective_mask_ratio,
        )
        loss = F.cross_entropy(masked_logits, masked_targets)
        return loss, masked_logits.detach(), masked_targets.detach(), float(effective_mask_ratio)

    def _get_masked_predictions_for_eval(
        self, tokens: torch.Tensor, mask_ratio: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Get masked predictions for evaluation without computing loss.

        Returns:
            Tuple of (masked_logits, masked_targets, mask_ratio)
        """
        batch_size = tokens.shape[0]
        tokens_flat = tokens.view(batch_size, -1)
        effective_mask_ratio = (
            self.maskgit.mask_scheduler.sample_mask_ratio() if mask_ratio is None else mask_ratio
        )
        masked_logits, masked_targets, _ = self.maskgit.transformer.predict_masked(
            tokens_flat,
            mask_ratio=effective_mask_ratio,
        )
        return masked_logits.detach(), masked_targets.detach(), float(effective_mask_ratio)

    def training_step(
        self,
        batch: torch.Tensor | tuple[Any, ...] | list[Any],
        encode_images_to_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> MaskGITTrainingStepOutput:
        x = self.extract_input_tensor(batch)
        tokens = encode_images_to_tokens_fn(x)
        loss, _, _, _ = self._get_masked_predictions_for_training(tokens)
        return {"loss": loss}

    def compute_masked_loss(
        self, tokens: torch.Tensor, mask_ratio: float | None = None
    ) -> tuple[torch.Tensor, dict[str, int | float]]:
        """Compute masked loss and return raw data for testing.

        This method is used for testing/debugging and returns loss
        plus raw data for metrics computation.

        Args:
            tokens: Token indices tensor
            mask_ratio: Optional explicit mask ratio

        Returns:
            Tuple of (loss, raw_data dict with correct/total/mask_ratio)
        """
        loss, masked_logits, masked_targets, effective_mask_ratio = (
            self._get_masked_predictions_for_training(tokens, mask_ratio=mask_ratio)
        )

        with torch.no_grad():
            predictions = masked_logits.argmax(dim=-1)
            correct = (predictions == masked_targets).sum().item()
            total = masked_targets.numel()

        raw_data: dict[str, int | float] = {
            "correct": correct,
            "total": total,
            "mask_ratio": effective_mask_ratio,
        }
        return loss, raw_data

    def validation_step(
        self,
        batch: torch.Tensor | tuple[Any, ...] | list[Any],
        encode_images_to_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> MaskGITEvalStepOutput:
        x = self.extract_input_tensor(batch)
        tokens = encode_images_to_tokens_fn(x)
        masked_logits, masked_targets, _ = self._get_masked_predictions_for_eval(tokens)

        with torch.no_grad():
            generated_images = self.maskgit.generate(
                shape=tokens.shape,
                temperature=1.0,
                num_iterations=12,
            )
        return {
            "x_real": x.detach().cpu(),
            "generated_images": generated_images.detach().cpu(),
            "masked_logits": masked_logits.cpu(),
            "masked_targets": masked_targets.cpu(),
        }

    def test_step(
        self,
        batch: torch.Tensor | tuple[Any, ...] | list[Any],
        encode_images_to_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> MaskGITEvalStepOutput:
        return self.validation_step(
            batch=batch,
            encode_images_to_tokens_fn=encode_images_to_tokens_fn,
        )

    def predict_step(
        self,
        batch: torch.Tensor | tuple[Any, ...] | list[Any],
        encode_images_to_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        x = self.extract_input_tensor(batch)
        tokens = encode_images_to_tokens_fn(x)
        with torch.no_grad():
            generated_images = self.maskgit.generate(
                shape=tokens.shape,
                temperature=1.0,
                num_iterations=12,
            )
        return {
            "x_real": x.detach().cpu(),
            "generated_images": generated_images.detach().cpu(),
        }
