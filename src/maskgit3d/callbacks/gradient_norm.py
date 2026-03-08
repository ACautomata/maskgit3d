"""Gradient norm monitoring callback."""

import logging
from typing import Any, Literal

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class GradientNormCallback(Callback):
    """Monitor and log gradient norms during training.

    This callback computes and logs gradient norms (total or per-layer)
    at specified intervals. Useful for debugging training instability
    and monitoring gradient flow.

    Args:
        mode: Which gradients to track. "total" logs only the total norm,
            "per_layer" logs norms for each parameter, "both" logs both.
        log_every_n_steps: Log gradient norms every N steps (batches).
        norm_type: Type of norm to compute (default: 2 for L2 norm).
        only_trainable: Only compute norms for trainable parameters.

    Example:
        >>> callback = GradientNormCallback(mode="per_layer", log_every_n_steps=100)
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        mode: Literal["total", "per_layer", "both"] = "total",
        log_every_n_steps: int = 1,
        norm_type: float = 2.0,
        only_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.log_every_n_steps = log_every_n_steps
        self.norm_type = norm_type
        self.only_trainable = only_trainable

        self._step_count = 0

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log gradient norms at specified intervals."""
        self._step_count += 1

        if self._step_count % self.log_every_n_steps != 0:
            return

        if trainer is None or trainer.logger is None:
            return

        has_gradients = any(
            p.grad is not None
            for p in pl_module.parameters()
            if (not self.only_trainable or p.requires_grad)
        )

        if not has_gradients:
            return

        if self.mode in ("total", "both"):
            self._log_total_norm(trainer, pl_module)

        if self.mode in ("per_layer", "both"):
            self._log_per_layer_norms(trainer, pl_module)

    def _log_total_norm(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log total gradient norm."""
        total_norm = self._compute_total_norm(pl_module)

        if torch.isfinite(total_norm) and trainer.logger is not None:
            trainer.logger.log_metrics(
                {"total_norm:gradients": total_norm.item()},
                step=trainer.global_step,
            )

    def _compute_total_norm(self, pl_module: LightningModule) -> torch.Tensor:
        """Compute total gradient norm across all parameters."""
        parameters = self._get_parameters(pl_module)

        if not parameters:
            return torch.tensor(0.0)

        grads = [p.grad.detach().norm(self.norm_type) for p in parameters if p.grad is not None]

        if not grads:
            return torch.tensor(0.0)

        total_norm: torch.Tensor = torch.stack(grads).norm(self.norm_type)
        return total_norm

    def _log_per_layer_norms(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log gradient norms for each layer/parameter."""
        norms = {}

        for name, param in pl_module.named_parameters():
            if self.only_trainable and not param.requires_grad:
                continue

            if param.grad is not None:
                grad_norm = param.grad.detach().norm(self.norm_type)

                if torch.isfinite(grad_norm):
                    clean_name = name.replace(".", "_")
                    norms[f"{clean_name}:gradients"] = grad_norm.item()

        if norms and trainer.logger is not None:
            trainer.logger.log_metrics(norms, step=trainer.global_step)

    def _get_parameters(self, pl_module: LightningModule):
        """Get parameters to compute norms for."""
        if self.only_trainable:
            return [p for p in pl_module.parameters() if p.requires_grad]
        return list(pl_module.parameters())

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset step counter at epoch start."""
        self._step_count = 0

    def state_dict(self) -> dict:
        """Return state dict for checkpointing."""
        return {
            "step_count": self._step_count,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self._step_count = state_dict.get("step_count", 0)
