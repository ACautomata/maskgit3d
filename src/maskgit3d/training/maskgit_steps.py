from __future__ import annotations

from typing import Any, Callable, cast

import torch
import torch.nn.functional as F
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

    def compute_masked_loss(
        self, tokens: torch.Tensor, mask_ratio: float | None = None
    ) -> tuple[torch.Tensor, dict[str, int | float]]:
        loss, callback_payload = self._compute_masked_predictions(tokens, mask_ratio=mask_ratio)
        masked_logits = cast(torch.Tensor, callback_payload["masked_logits"])
        masked_targets = cast(torch.Tensor, callback_payload["masked_targets"])

        with torch.no_grad():
            predictions = masked_logits.argmax(dim=-1)
            correct = (predictions == masked_targets).sum().item()
            total = masked_targets.numel()

        raw_data: dict[str, int | float] = {
            "correct": correct,
            "total": total,
            "mask_ratio": float(callback_payload["mask_ratio"]),
        }
        return loss, raw_data

    def _compute_masked_predictions(
        self, tokens: torch.Tensor, mask_ratio: float | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
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
        return loss, {
            "tokens": tokens.detach(),
            "masked_logits": masked_logits.detach(),
            "masked_targets": masked_targets.detach(),
            "mask_ratio": float(effective_mask_ratio),
        }

    def training_step(
        self,
        batch: torch.Tensor | tuple[Any, ...] | list[Any],
        encode_images_to_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> MaskGITTrainingStepOutput:
        x = self.extract_input_tensor(batch)
        tokens = encode_images_to_tokens_fn(x)
        loss, raw_data = self.compute_masked_loss(tokens)
        return {
            "loss": loss,
            "mask_ratio": raw_data["mask_ratio"],
        }

    def validation_step(
        self,
        batch: torch.Tensor | tuple[Any, ...] | list[Any],
        encode_images_to_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> MaskGITEvalStepOutput:
        x = self.extract_input_tensor(batch)
        tokens = encode_images_to_tokens_fn(x)
        _, callback_payload = self._compute_masked_predictions(tokens)

        with torch.no_grad():
            generated_images = self.maskgit.generate(
                shape=tokens.shape,
                temperature=1.0,
                num_iterations=12,
            )
        return {
            "x_real": x.detach().cpu(),
            "generated_images": generated_images.detach().cpu(),
            "masked_logits": cast(torch.Tensor, callback_payload["masked_logits"]).cpu(),
            "masked_targets": cast(torch.Tensor, callback_payload["masked_targets"]).cpu(),
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
