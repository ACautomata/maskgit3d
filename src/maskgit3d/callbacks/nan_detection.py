"""NaN/Inf detection callback for monitoring training stability."""

import logging
from typing import Any, Literal

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class NaNDetectionCallback(Callback):
    """Detect NaN/Inf values in losses and gradients during training.

    This callback monitors training for numerical instability and aborts
    training when NaN/Inf values are detected.

    For manual optimization (e.g., GANs with multiple optimizers), this callback
    hooks into on_before_optimizer_step to check gradients BEFORE the optimizer
    step is taken, preventing corrupted updates.

    Args:
        action: Action to take when NaN/Inf is detected. Only "abort" is supported
            which stops training immediately.
        check_loss: Whether to check loss values for NaN/Inf.
        check_gradients: Whether to check gradients for NaN/Inf.
        log_every_n_nan: Log a warning every N NaN occurrences to avoid spam.

    Example:
        >>> callback = NaNDetectionCallback()
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        action: Literal["abort"] = "abort",
        check_loss: bool = True,
        check_gradients: bool = True,
        log_every_n_nan: int = 1,
    ) -> None:
        super().__init__()
        self.action = action
        self.check_loss = check_loss
        self.check_gradients = check_gradients
        self.log_every_n_nan = log_every_n_nan

        self._nan_count = 0
        self._nan_count_epoch = 0
        self._batch_idx = 0

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset epoch-level counters."""
        self._nan_count_epoch = 0
        self._batch_idx = 0

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Track batch index."""
        self._batch_idx = batch_idx

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Check for NaN/Inf in loss after each batch."""
        # Check loss for NaN/Inf (for automatic optimization)
        if self.check_loss:
            loss = self._extract_loss(outputs)
            if loss is not None and self._is_invalid(loss):
                self._handle_nan(
                    "loss", batch_idx, loss.item() if torch.isfinite(loss).all() else float("nan")
                )

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Check gradients BEFORE optimizer step to prevent corrupted updates.

        This is critical for manual optimization scenarios where optimizer.step()
        is called explicitly in the training_step.
        """
        if not self.check_gradients:
            return

        # Check all parameters that this optimizer handles
        nan_found = False
        for group in optimizer.param_groups:
            for param in group.get("params", []):
                if param.grad is not None and self._is_invalid(param.grad):
                    # Try to find parameter name
                    param_name = self._get_param_name(pl_module, param)
                    self._handle_nan(f"gradient[{param_name}]", self._batch_idx)
                    nan_found = True
                    break
            if nan_found:
                break

    def _get_param_name(self, pl_module: LightningModule, param: torch.nn.Parameter) -> str:
        """Get the name of a parameter."""
        for name, p in pl_module.named_parameters():
            if p is param:
                return name
        return "unknown"

    def _extract_loss(
        self,
        outputs: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Extract loss tensor from outputs."""
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        return None

    def _is_invalid(self, tensor: torch.Tensor) -> bool:
        """Check if tensor contains NaN or Inf values."""
        return bool(torch.isnan(tensor).any() or torch.isinf(tensor).any())

    def _handle_nan(self, source: str, batch_idx: int, value: float | None = None) -> None:
        """Handle NaN/Inf detection based on configured action."""
        self._nan_count += 1
        self._nan_count_epoch += 1

        value_str = f" (value: {value})" if value is not None else ""
        message = (
            f"NaN/Inf detected in {source} at batch {batch_idx}{value_str}. "
            f"Total NaN count: {self._nan_count}, Epoch NaN count: {self._nan_count_epoch}"
        )

        if self._nan_count % self.log_every_n_nan == 0:
            logger.warning(message)

        # Log to trainer if logger exists
        if (
            hasattr(self, "_trainer")
            and self._trainer is not None
            and self._trainer.logger is not None
        ):
            self._trainer.logger.log_metrics(
                {"nan_detected:train": 1.0, "nan_count:train": self._nan_count},
                step=self._trainer.global_step,
            )

        raise RuntimeError(f"Training aborted due to NaN/Inf in {source}. {message}")

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Store reference to trainer for logging."""
        self._trainer = trainer

    def state_dict(self) -> dict:
        """Return state dict for checkpointing."""
        return {
            "nan_count": self._nan_count,
            "action": self.action,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self._nan_count = state_dict.get("nan_count", 0)
        self.action = state_dict.get("action", self.action)
