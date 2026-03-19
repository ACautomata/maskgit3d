from __future__ import annotations

from typing import Any, Callable, cast

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


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

        with torch.no_grad():
            predictions = masked_logits.argmax(dim=-1)
            correct = (predictions == masked_targets).sum().item()
            total = masked_targets.numel()

        raw_data: dict[str, int | float] = {
            "correct": correct,
            "total": total,
            "mask_ratio": float(effective_mask_ratio),
        }
        return loss, raw_data

    def training_step(
        self,
        batch: torch.Tensor | tuple[Any, ...] | list[Any],
        encode_images_to_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> dict[str, Any]:
        x = self.extract_input_tensor(batch)
        tokens = encode_images_to_tokens_fn(x)
        loss, raw_data = self.compute_masked_loss(tokens)
        return {"loss": loss, "log_data": raw_data}

    def validation_step(
        self,
        batch: torch.Tensor | tuple[Any, ...] | list[Any],
        encode_images_to_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
        log_fn: Callable[..., None] | None = None,
    ) -> dict[str, Any]:
        logger = log_fn or self.log_fn
        if logger is None:
            raise ValueError("validation_step requires log_fn.")

        x = self.extract_input_tensor(batch)
        tokens = encode_images_to_tokens_fn(x)
        loss, raw_data = self.compute_masked_loss(tokens)
        batch_size = x.shape[0]

        logger(
            "val_loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        total = raw_data["total"]
        if total > 0:
            logger(
                "val_mask_acc",
                raw_data["correct"] / total,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
        logger(
            "val_mask_ratio",
            raw_data["mask_ratio"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )

        with torch.no_grad():
            generated_images = self.maskgit.generate(
                shape=tokens.shape,
                temperature=1.0,
                num_iterations=12,
            )
        return {"generated_images": generated_images.detach().cpu()}

    def test_step(
        self,
        batch: torch.Tensor | tuple[Any, ...] | list[Any],
        encode_images_to_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> dict[str, Any]:
        x = self.extract_input_tensor(batch)
        with torch.no_grad():
            tokens_shape = encode_images_to_tokens_fn(x).shape
            generated_images = self.maskgit.generate(
                shape=tokens_shape,
                temperature=1.0,
                num_iterations=12,
            )
        return {
            "generated_images": generated_images,
            "input_shape": x.shape,
            "token_shape": tokens_shape,
        }

    def create_optimizers(
        self,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        optimizer_config: Any = None,
    ) -> dict[str, Any]:
        from ..runtime.optimizer_factory import create_optimizer
        from ..runtime.scheduler_factory import create_scheduler

        if optimizer_config is not None:
            optimizer = create_optimizer(self.maskgit.parameters(), optimizer_config)
        else:
            optimizer = torch.optim.AdamW(
                self.maskgit.transformer.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        scheduler_config = OmegaConf.create({"warmup_steps": warmup_steps})
        scheduler = create_scheduler(optimizer, scheduler_config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
