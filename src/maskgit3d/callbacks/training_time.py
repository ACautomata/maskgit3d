"""Training time tracking callback."""

import logging
import time
from collections import deque
from typing import Any

from lightning.pytorch import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class TrainingTimeCallback(Callback):
    """Track and log training time metrics.

    This callback tracks time spent in training, validation, and provides
    estimates for time to completion (ETC). Useful for monitoring long
    training runs and resource planning.

    Args:
        log_every_n_epochs: Log time metrics every N epochs (0 = every epoch).
        estimate_time_to_completion: Whether to estimate and log time to completion.
        max_epoch_history: Maximum number of epoch times to keep for ETC calculation.
            Uses bounded deque to prevent memory leaks during long training runs.

    Example:
        >>> callback = TrainingTimeCallback(log_every_n_epochs=1)
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        estimate_time_to_completion: bool = True,
        max_epoch_history: int = 100,
    ) -> None:
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.estimate_time_to_completion = estimate_time_to_completion
        self.max_epoch_history = max_epoch_history

        # Timing state
        self._epoch_start_time: float | None = None
        self._train_start_time: float | None = None
        self._validation_start_time: float | None = None

        # Aggregated metrics
        self._total_train_time = 0.0
        self._total_validation_time = 0.0
        # Use bounded deque to prevent unbounded memory growth during long training runs
        self._epoch_times: deque[float] = deque(maxlen=max_epoch_history)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Record training start time."""
        self._train_start_time = time.time()
        logger.info("Training started")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Record epoch start time."""
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log epoch training time."""
        if self._epoch_start_time is None or trainer.logger is None:
            return

        epoch_time = time.time() - self._epoch_start_time
        self._epoch_times.append(epoch_time)
        self._total_train_time += epoch_time

        # Log every N epochs (or every epoch if log_every_n_epochs == 1)
        current_epoch = trainer.current_epoch
        if self.log_every_n_epochs <= 1 or current_epoch % self.log_every_n_epochs == 0:
            metrics: dict[str, Any] = {
                "epoch_train_seconds:time": epoch_time,
                "epoch_train_minutes:time": epoch_time / 60.0,
                "total_train_seconds:time": self._total_train_time,
                "total_train_minutes:time": self._total_train_time / 60.0,
            }

            # Estimate time to completion
            if self.estimate_time_to_completion:
                etc_seconds = self._estimate_etc(trainer)
                if etc_seconds is not None:
                    metrics["etc_seconds:time"] = etc_seconds
                    metrics["etc_minutes:time"] = etc_seconds / 60.0
                    metrics["etc_hours:time"] = etc_seconds / 3600.0

            trainer.logger.log_metrics(metrics, step=trainer.global_step)

            # Log to console
            time_str = self._format_time(epoch_time)
            total_str = self._format_time(self._total_train_time)
            logger.info(
                f"Epoch {current_epoch} completed in {time_str}. Total training time: {total_str}"
            )

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Record validation start time."""
        self._validation_start_time = time.time()

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log validation time."""
        if self._validation_start_time is None or trainer.logger is None:
            return

        validation_time = time.time() - self._validation_start_time
        self._total_validation_time += validation_time

        trainer.logger.log_metrics(
            {
                "validation_seconds:time": validation_time,
                "validation_minutes:time": validation_time / 60.0,
                "total_validation_seconds:time": self._total_validation_time,
                "total_validation_minutes:time": self._total_validation_time / 60.0,
            },
            step=trainer.global_step,
        )

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log total training statistics."""
        if self._train_start_time is None:
            return

        total_time = time.time() - self._train_start_time

        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "total_elapsed_seconds:time": total_time,
                    "total_elapsed_minutes:time": total_time / 60.0,
                    "total_elapsed_hours:time": total_time / 3600.0,
                },
                step=trainer.global_step,
            )

        # Log final summary
        train_str = self._format_time(self._total_train_time)
        val_str = self._format_time(self._total_validation_time)
        total_str = self._format_time(total_time)

        logger.info("=" * 60)
        logger.info("Training completed!")
        logger.info(f"  Total time:          {total_str}")
        logger.info(f"  Training time:       {train_str}")
        logger.info(f"  Validation time:     {val_str}")
        logger.info(f"  Average epoch time:  {self._format_time(self._get_avg_epoch_time())}")
        logger.info(f"  Epochs completed:    {len(self._epoch_times)}")
        logger.info("=" * 60)

    def _estimate_etc(self, trainer: Trainer) -> float | None:
        """Estimate time to completion in seconds."""
        if not self._epoch_times:
            return None

        max_epochs = trainer.max_epochs
        current_epoch = trainer.current_epoch

        if max_epochs is None or max_epochs <= 0:
            return None

        epochs_remaining = max_epochs - current_epoch - 1
        if epochs_remaining <= 0:
            return 0.0

        avg_epoch_time = self._get_avg_epoch_time()
        etc_seconds = epochs_remaining * avg_epoch_time

        return etc_seconds

    def _get_avg_epoch_time(self) -> float:
        """Get average epoch time from recent epochs."""
        if not self._epoch_times:
            return 0.0

        # Use last 5 epochs or all available
        epoch_list = list(self._epoch_times)
        recent_times = epoch_list[-5:]
        return sum(recent_times) / len(recent_times)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"

    def state_dict(self) -> dict[str, Any]:
        """Return state dict for checkpointing."""
        return {
            "total_train_time": self._total_train_time,
            "total_validation_time": self._total_validation_time,
            "epoch_times": list(self._epoch_times),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self._total_train_time = state_dict.get("total_train_time", 0.0)
        self._total_validation_time = state_dict.get("total_validation_time", 0.0)
        epoch_times = state_dict.get("epoch_times", [])
        self._epoch_times = deque(epoch_times, maxlen=self.max_epoch_history)
