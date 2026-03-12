"""Training stability monitoring callback for comprehensive health tracking."""

import logging
from collections import deque
from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class TrainingStabilityCallback(Callback):
    """Monitor training stability with comprehensive health metrics.

    This callback tracks multiple indicators of training stability:
    - Loss spikes and trends (moving average, variance)
    - Gradient explosion/vanishing detection
    - Training health metrics over time
    - Alerts when instability is detected

    Args:
        loss_window_size: Window size for moving average calculations.
        loss_spike_threshold: Multiplier for loss spike detection.
            A spike is flagged when loss > moving_avg * threshold.
        grad_explosion_threshold: Maximum allowed gradient norm.
        grad_vanishing_threshold: Minimum allowed gradient norm.
        log_every_n_steps: How often to log metrics to the logger.
        alert_on_spike: Whether to log warnings on loss spikes.
        stop_on_explosion: Whether to stop training on gradient explosion.
        check_gradients: Whether to check gradient norms.

    Example:
        >>> callback = TrainingStabilityCallback(
        ...     loss_window_size=100,
        ...     loss_spike_threshold=2.0,
        ...     log_every_n_steps=10,
        ... )
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        loss_window_size: int = 100,
        loss_spike_threshold: float = 2.0,
        grad_explosion_threshold: float = 1000.0,
        grad_vanishing_threshold: float = 1e-7,
        log_every_n_steps: int = 10,
        alert_on_spike: bool = True,
        stop_on_explosion: bool = False,
        check_gradients: bool = True,
    ) -> None:
        super().__init__()
        self.loss_window_size = loss_window_size
        self.loss_spike_threshold = loss_spike_threshold
        self.grad_explosion_threshold = grad_explosion_threshold
        self.grad_vanishing_threshold = grad_vanishing_threshold
        self.log_every_n_steps = log_every_n_steps
        self.alert_on_spike = alert_on_spike
        self.stop_on_explosion = stop_on_explosion
        self.check_gradients = check_gradients

        # Loss tracking
        self._loss_buffer: deque[float] = deque(maxlen=loss_window_size)
        self._loss_improvements: deque[bool] = deque(maxlen=loss_window_size)
        self._prev_loss: float | None = None

        # Gradient tracking
        self._grad_norm_buffer: deque[float] = deque(maxlen=loss_window_size)

        # Epoch tracking
        self._epoch_start_loss: float | None = None
        self._epoch_end_loss: float | None = None

        # Step counter
        self._step_count = 0

        # Spike/explosion counters for reporting
        self._spike_count = 0
        self._explosion_count = 0
        self._vanishing_count = 0

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize tracking state at training start."""
        self._reset_buffers()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset epoch-level tracking."""
        self._epoch_start_loss = None
        self._epoch_end_loss = None
        self._step_count = 0

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Monitor loss and gradients after each batch."""
        self._step_count += 1

        # Extract and process loss
        loss = self._extract_loss(outputs)
        if loss is not None:
            loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
            self._process_loss(trainer, loss_value)

        # Log metrics at specified interval
        if self._step_count % self.log_every_n_steps == 0:
            self._log_metrics(trainer)

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Check gradients before optimizer step."""
        if not self.check_gradients:
            return

        total_norm = self._compute_gradient_norm(pl_module)
        if total_norm is not None:
            self._grad_norm_buffer.append(total_norm)
            self._check_gradient_health(trainer, total_norm)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log epoch-level statistics."""
        if trainer.logger is None:
            return

        metrics: dict[str, float] = {}

        # Epoch loss delta
        if self._epoch_start_loss is not None and self._epoch_end_loss is not None:
            delta = self._epoch_end_loss - self._epoch_start_loss
            metrics["epoch_loss_delta:stability"] = delta

        # Total counts
        metrics["spike_count:stability"] = float(self._spike_count)
        metrics["explosion_count:stability"] = float(self._explosion_count)
        metrics["vanishing_count:stability"] = float(self._vanishing_count)

        if metrics:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

    def _reset_buffers(self) -> None:
        """Reset all tracking buffers."""
        self._loss_buffer.clear()
        self._loss_improvements.clear()
        self._grad_norm_buffer.clear()
        self._prev_loss = None
        self._epoch_start_loss = None
        self._epoch_end_loss = None
        self._step_count = 0
        self._spike_count = 0
        self._explosion_count = 0
        self._vanishing_count = 0

    def _extract_loss(self, outputs: Any) -> torch.Tensor | float | None:
        """Extract loss from training step outputs."""
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, dict):
            if "loss" in outputs:
                return outputs["loss"]
            # Try common loss keys
            for key in ["train_loss", "total_loss", "step_loss"]:
                if key in outputs:
                    return outputs[key]
        return None

    def _process_loss(self, trainer: Trainer, loss_value: float) -> None:
        """Process loss value and check for spikes."""
        # Track epoch start/end loss
        if self._epoch_start_loss is None:
            self._epoch_start_loss = loss_value
        self._epoch_end_loss = loss_value

        # Track loss improvement
        if self._prev_loss is not None:
            improved = loss_value < self._prev_loss
            self._loss_improvements.append(improved)
        self._prev_loss = loss_value

        # Add to buffer
        self._loss_buffer.append(loss_value)

        # Check for spike
        if len(self._loss_buffer) >= 10:  # Need enough samples
            moving_avg = sum(self._loss_buffer) / len(self._loss_buffer)
            if loss_value > moving_avg * self.loss_spike_threshold:
                self._spike_count += 1
                if self.alert_on_spike:
                    logger.warning(
                        "Loss spike detected at step %d: %.4f (moving avg: %.4f, ratio: %.2f)",
                        trainer.global_step,
                        loss_value,
                        moving_avg,
                        loss_value / moving_avg if moving_avg > 0 else float("inf"),
                    )

    def _compute_gradient_norm(self, pl_module: LightningModule) -> float | None:
        """Compute total gradient norm."""
        parameters = [p for p in pl_module.parameters() if p.requires_grad]
        if not parameters:
            return None

        grads = [p.grad.detach().norm(2.0) for p in parameters if p.grad is not None]
        if not grads:
            return None

        total_norm: torch.Tensor = torch.stack(grads).norm(2.0)
        return total_norm.item()

    def _check_gradient_health(self, trainer: Trainer, grad_norm: float) -> None:
        """Check gradient norm for explosion or vanishing."""
        if grad_norm > self.grad_explosion_threshold:
            self._explosion_count += 1
            logger.warning(
                "Gradient explosion detected at step %d: norm=%.4f (threshold=%.4f)",
                trainer.global_step,
                grad_norm,
                self.grad_explosion_threshold,
            )
            if self.stop_on_explosion:
                raise RuntimeError(
                    f"Training stopped due to gradient explosion. "
                    f"Gradient norm {grad_norm:.4f} exceeded threshold {self.grad_explosion_threshold}"
                )

        if grad_norm < self.grad_vanishing_threshold:
            self._vanishing_count += 1
            logger.warning(
                "Gradient vanishing detected at step %d: norm=%.4e (threshold=%.4e)",
                trainer.global_step,
                grad_norm,
                self.grad_vanishing_threshold,
            )

    def _log_metrics(self, trainer: Trainer) -> None:
        """Log stability metrics to the trainer logger."""
        if trainer.logger is None:
            return

        metrics: dict[str, float] = {}

        # Loss statistics
        if self._loss_buffer:
            loss_list = list(self._loss_buffer)
            loss_mean = sum(loss_list) / len(loss_list)
            loss_var = sum((x - loss_mean) ** 2 for x in loss_list) / len(loss_list)
            metrics["loss_mean:stability"] = loss_mean
            metrics["loss_std:stability"] = loss_var**0.5
            metrics["loss_min:stability"] = min(loss_list)
            metrics["loss_max:stability"] = max(loss_list)

        # Loss improvement rate
        if self._loss_improvements:
            improvement_rate = sum(self._loss_improvements) / len(self._loss_improvements)
            metrics["loss_improvement_rate:stability"] = improvement_rate

        # Gradient statistics
        if self._grad_norm_buffer:
            grad_list = list(self._grad_norm_buffer)
            grad_mean = sum(grad_list) / len(grad_list)
            grad_var = sum((x - grad_mean) ** 2 for x in grad_list) / len(grad_list)
            metrics["grad_norm_mean:stability"] = grad_mean
            metrics["grad_norm_std:stability"] = grad_var**0.5
            metrics["grad_norm_min:stability"] = min(grad_list)
            metrics["grad_norm_max:stability"] = max(grad_list)

        # Spike/explosion counts
        metrics["spike_count:stability"] = float(self._spike_count)
        metrics["explosion_count:stability"] = float(self._explosion_count)
        metrics["vanishing_count:stability"] = float(self._vanishing_count)

        if metrics:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

    def state_dict(self) -> dict[str, Any]:
        """Return state dict for checkpointing."""
        return {
            "loss_buffer": list(self._loss_buffer),
            "loss_improvements": list(self._loss_improvements),
            "grad_norm_buffer": list(self._grad_norm_buffer),
            "prev_loss": self._prev_loss,
            "spike_count": self._spike_count,
            "explosion_count": self._explosion_count,
            "vanishing_count": self._vanishing_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self._loss_buffer = deque(state_dict.get("loss_buffer", []), maxlen=self.loss_window_size)
        self._loss_improvements = deque(
            state_dict.get("loss_improvements", []), maxlen=self.loss_window_size
        )
        self._grad_norm_buffer = deque(
            state_dict.get("grad_norm_buffer", []), maxlen=self.loss_window_size
        )
        self._prev_loss = state_dict.get("prev_loss")
        self._spike_count = state_dict.get("spike_count", 0)
        self._explosion_count = state_dict.get("explosion_count", 0)
        self._vanishing_count = state_dict.get("vanishing_count", 0)
