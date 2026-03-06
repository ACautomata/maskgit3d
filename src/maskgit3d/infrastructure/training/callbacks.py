"""
Training callbacks for maskgit-3d training framework.

Provides callbacks compatible with PyTorch Lightning's Callback interface.
Supports both custom implementations and Lightning's built-in callbacks.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback as _LightningCallback
from lightning.pytorch.callbacks import \
    EarlyStopping as _LightningEarlyStopping
from lightning.pytorch.callbacks import \
    ModelCheckpoint as _LightningModelCheckpoint
from PIL import Image

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]
    TENSORBOARD_AVAILABLE = False

if TYPE_CHECKING:
    from lightning import Fabric
    from lightning.pytorch import Trainer

logger = logging.getLogger(__name__)


class Callback(_LightningCallback):
    """
    Base callback class compatible with PyTorch Lightning's Callback.

    This provides a unified interface that works both with FabricTrainingPipeline
    and Lightning's Trainer. All callbacks in maskgit-3d extend this class.

    The callback receives Fabric/Trainer and model/optimizer through the hook
    methods, allowing consistent behavior across different training pipelines.
    """

    def __init__(self) -> None:
        super().__init__()
        self._fabric: Fabric | None = None
        self._trainer: Trainer | None = None

    @property
    def trainer(self) -> Trainer | None:
        """Get the Lightning Trainer if available."""
        return self._trainer

    @property
    def fabric(self) -> Fabric | None:
        """Get the Fabric instance if available."""
        return self._fabric

    def on_fit_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        """Called at the very beginning of fit."""
        self._trainer = trainer

    def on_train_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        """Called at the end of training."""
        pass

    def on_validation_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        """Called at the beginning of validation."""
        pass

    def on_validation_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        """Called at the end of validation."""
        pass


class ModelCheckpoint(Callback):
    """
    Save model checkpoints based on monitored metric.

    This is a custom implementation compatible with Lightning's Callback interface.
    For full Lightning compatibility, consider using :class:`LightningModelCheckpoint`
    which wraps Lightning's built-in ModelCheckpoint.

    Args:
        monitor: Metric name to monitor (default: "val_loss")
        mode: "min" or "max" (default: "min")
        save_top_k: Number of best checkpoints to save (default: 3, -1 for all)
        filename: Checkpoint filename format (default: "{epoch:02d}-{monitor:.4f}")
        save_last: Whether to always save "last.ckpt" (default: True)
        dirpath: Directory to save checkpoints (default: "./checkpoints")
        save_best: Whether to save best checkpoint (default: True)
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        filename: str | None = None,
        save_last: bool = True,
        dirpath: str = "./checkpoints",
        save_best: bool | None = None,
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k if save_best is not False else 0
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.save_last = save_last

        if filename is None:
            filename = "{epoch:02d}-{" + monitor + ":.4f}"
        self.filename = filename

        self._current_epoch = 0
        self._best_scores: list[tuple[float, Path]] = []

    def on_fit_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        """Initialize state at the start of training."""
        super().on_fit_start(trainer, pl_module)
        self._current_epoch = 0
        self._best_scores = []

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: torch.nn.Module,
    ) -> None:
        """Save checkpoints after validation epoch."""
        # Get metrics from trainer
        metrics = trainer.callback_metrics
        score = metrics.get(self.monitor)

        if score is None:
            return

        # Convert to float if tensor
        if isinstance(score, torch.Tensor):
            score = score.item()

        if self.save_last:
            self._save_checkpoint(pl_module, "last", trainer)

        if self.save_top_k != 0:
            self._update_best_checkpoints(pl_module, score, trainer)

        self._current_epoch += 1

    def _update_best_checkpoints(
        self,
        pl_module: torch.nn.Module,
        score: float,
        trainer: Trainer,
    ) -> None:
        is_better = False
        if not self._best_scores:
            is_better = True
        elif self.mode == "min":
            is_better = score < self._best_scores[-1][0]
        else:
            is_better = score > self._best_scores[-1][0]

        if is_better or len(self._best_scores) < self.save_top_k or self.save_top_k == -1:
            filename = self.filename.format(
                epoch=self._current_epoch,
                **{self.monitor: score},
            )

            self._save_checkpoint(pl_module, filename, trainer)
            filepath = self.dirpath / f"{filename}.ckpt"
            self._best_scores.append((score, filepath))
            self._best_scores.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

            if self.save_top_k > 0:
                while len(self._best_scores) > self.save_top_k:
                    _, old_path = self._best_scores.pop()
                    if old_path.exists():
                        old_path.unlink()

    def _save_checkpoint(
        self,
        pl_module: torch.nn.Module,
        filename: str,
        trainer: Trainer,
    ) -> None:
        filepath = self.dirpath / f"{filename}.ckpt"

        checkpoint = {
            "epoch": self._current_epoch,
            "state_dict": pl_module.state_dict(),
            "monitor": self.monitor,
            self.monitor: self._best_scores[-1][0] if self._best_scores else None,
        }

        # Add optimizer state if available
        if hasattr(trainer, "optimizers") and trainer.optimizers:
            optimizer = trainer.optimizers[0]
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    @property
    def current_epoch(self) -> int:
        """Current epoch number."""
        return self._current_epoch

    @property
    def best_scores(self) -> list:
        """List of best scores."""
        return [s[0] for s in self._best_scores]

    @property
    def best_model_path(self) -> str:
        """Path to best model checkpoint."""
        if self._best_scores:
            return str(self._best_scores[0][1])
        return ""

    @property
    def best_model_score(self) -> torch.Tensor | None:
        """Best score value."""
        if self._best_scores:
            return torch.tensor(self._best_scores[0][0])
        return None


class LightningModelCheckpoint(_LightningModelCheckpoint):
    """
    Wrapper for Lightning's built-in ModelCheckpoint.

    This provides full compatibility with Lightning's checkpoint callback
    while maintaining the same interface as our custom ModelCheckpoint.

    Example:
        callback = LightningModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            dirpath="./checkpoints",
            filename="{epoch:02d}-{val_loss:.4f}",
        )
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        filename: str | None = None,
        save_last: bool = True,
        dirpath: str | Path | None = "./checkpoints",
        save_best: bool = True,
        auto_insert_metric_name: bool = True,
        every_n_epochs: int = 1,
        every_n_train_steps: int | None = None,
        train_time_interval: Any | None = None,
    ):
        if filename is None:
            filename = "{epoch:02d}-{" + monitor + ":.4f}"

        super().__init__(
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            filename=filename,
            save_last=save_last,
            dirpath=dirpath,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_epochs=every_n_epochs,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
        )


class EarlyStopping(Callback):
    """
    Stop training when monitored metric stops improving.

    This is a custom implementation that wraps Lightning's EarlyStopping
    for full compatibility with both Fabric and Trainer.

    Args:
        monitor: Metric name to monitor (default: "val_loss")
        mode: "min" or "max" (default: "min")
        patience: Number of epochs with no improvement (default: 10)
        min_delta: Minimum change to qualify as improvement (default: 0.0)
        check_finite: Whether to check for finite values (default: True)
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.0,
        check_finite: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.check_finite = check_finite

        self._early_stopping = _LightningEarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            mode=mode,
            check_finite=check_finite,
            verbose=False,
        )

        self.should_stop = False

    def on_fit_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        """Initialize state at the start of training."""
        super().on_fit_start(trainer, pl_module)
        torch_inf = torch.tensor(torch.inf)
        self._early_stopping.best_score = (
            torch_inf if self._early_stopping.monitor_op == torch.lt else -torch_inf
        )
        self._early_stopping.wait_count = 0
        self._early_stopping.stopped_epoch = 0
        self.should_stop = False

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: torch.nn.Module,
    ) -> None:
        """Check for early stopping condition after validation."""
        metrics = trainer.callback_metrics
        score = metrics.get(self.monitor)

        if score is None:
            return

        # Convert to tensor
        current = score if isinstance(score, torch.Tensor) else torch.tensor(score)

        if self.check_finite and not torch.isfinite(current):
            self.should_stop = True
            logger.warning(f"Early stopping triggered: {self.monitor} is not finite ({current})")
            return

        if self._early_stopping.best_score is None:
            self._early_stopping.best_score = current
            self._early_stopping.wait_count = 0
            return

        monitor_op = self._early_stopping.monitor_op

        if monitor_op(current - self.min_delta, self._early_stopping.best_score):
            self._early_stopping.best_score = current
            self._early_stopping.wait_count = 0
        else:
            self._early_stopping.wait_count += 1
            if self._early_stopping.wait_count >= self.patience:
                self.should_stop = True
                logger.warning(
                    f"Early stopping triggered after {self.patience} epochs without improvement. "
                    f"Best {self.monitor}: {self._early_stopping.best_score.item():.6f}"
                )

    @property
    def best_score(self) -> float | None:
        """Best score achieved."""
        if self._early_stopping.best_score is None:
            return None
        val = self._early_stopping.best_score.item()
        if val == float("inf") or val == float("-inf"):
            return None
        return val

    @property
    def counter(self) -> int:
        """Number of epochs without improvement."""
        return self._early_stopping.wait_count


class LightningEarlyStopping(_LightningEarlyStopping):
    """
    Wrapper for Lightning's built-in EarlyStopping.

    This provides full compatibility with Lightning's early stopping callback.

    Example:
        callback = LightningEarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            min_delta=0.0,
        )
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.0,
        check_finite: bool = True,
        divergence_threshold: float = 1e6,
        check_on_train_epoch_end: bool = False,
    ):
        super().__init__(
            monitor=monitor,
            mode=mode,
            patience=patience,
            min_delta=min_delta,
            check_finite=check_finite,
            divergence_threshold=divergence_threshold,
            check_on_train_epoch_end=check_on_train_epoch_end,
        )


class NaNMonitor(Callback):
    """
    Monitor for NaN values in losses and gradients.

    Detects NaN/Inf values in training loss and model gradients,
    and can either raise an error or skip the batch.

    Args:
        check_interval: Check every N batches (default: 10)
        raise_on_nan: Whether to raise error on NaN (default: True)
        on_nan_action: Action to take on NaN ("error", "skip", "log") (default: "error")
    """

    def __init__(
        self,
        check_interval: int = 10,
        raise_on_nan: bool = True,
        on_nan_action: str = "error",
    ):
        super().__init__()
        self.check_interval = check_interval
        self.raise_on_nan = raise_on_nan
        self.on_nan_action = on_nan_action
        self._batch_count = 0

    def on_fit_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        """Initialize state at the start of training."""
        super().on_fit_start(trainer, pl_module)
        self._batch_count = 0

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: torch.nn.Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Check for NaN values after each training batch."""
        self._batch_count += 1

        if self._batch_count % self.check_interval != 0:
            return

        # Get loss from outputs
        loss = None
        if outputs is not None:
            if isinstance(outputs, dict):
                loss = outputs.get("loss")
            elif isinstance(outputs, torch.Tensor):
                loss = outputs

        if loss is not None:
            if isinstance(loss, torch.Tensor):
                loss_value = loss.item() if loss.numel() == 1 else loss.mean().item()
            else:
                loss_value = float(loss)

            if math.isnan(loss_value) or math.isinf(loss_value):
                self._handle_nan("loss", loss_value)
                return

        # Check gradients
        has_nan_grad = False
        for name, param in pl_module.named_parameters():
            if param.grad is not None and (
                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
            ):
                has_nan_grad = True
                logger.warning(f"NaN/Inf detected in gradient of {name}")
                break

        if has_nan_grad:
            self._handle_nan("gradient", None)

    def _handle_nan(self, nan_type: str, value: float | None) -> None:
        msg = f"NaN/Inf detected in {nan_type}"
        if value is not None:
            msg += f": {value}"

        if self.on_nan_action == "error" or self.raise_on_nan:
            logger.error(msg)
            raise RuntimeError(f"Training stopped due to NaN/Inf in {nan_type}")
        elif self.on_nan_action == "log":
            logger.warning(msg)
        # "skip" action just logs and continues

    @property
    def batch_count(self) -> int:
        """Number of batches processed."""
        return self._batch_count


class MetricsLogger(Callback):
    """
    Log metrics to JSON and CSV files.

    Records training and validation metrics throughout training
    and exports them to files.

    Args:
        log_dir: Directory to save logs (default: "./logs")
        log_interval: Log every N epochs (default: 1)
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        log_interval: int = 1,
    ):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        self._history: dict[str, list[Any]] = {}
        self._epoch = 0

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: torch.nn.Module,
    ) -> None:
        """Record metrics after training epoch."""
        self._epoch += 1

        if self._epoch % self.log_interval != 0:
            return

        # Get metrics from trainer
        metrics = trainer.callback_metrics

        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, int | float):
                if key not in self._history:
                    self._history[key] = []
                self._history[key].append(value)

        self._save_metrics()

    def on_fit_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        """Save metrics at the end of training."""
        self._save_metrics()

    def _save_metrics(self) -> None:
        import csv
        import json

        if not self._history:
            return

        json_path = self.log_dir / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(self._history, f, indent=2)

        csv_path = self.log_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + list(self._history.keys()))

            max_len = max(len(v) for v in self._history.values())
            for i in range(max_len):
                row: list[Any] = [i]
                for key in self._history:
                    if i < len(self._history[key]):
                        row.append(self._history[key][i])
                    else:
                        row.append("")
                writer.writerow(row)

    @property
    def history(self) -> dict[str, list[Any]]:
        """Get metric history."""
        return self._history

    def get_history(self) -> dict[str, list[Any]]:
        """Get a copy of metric history."""
        return self._history.copy()


class AxialSliceVisualizationCallback(Callback):
    """
    Visualize random axial slices from 3D volumes during validation/testing.

    Extracts random axial slices from 3D medical volumes and saves them
    to disk and/or logs to TensorBoard for visualization.

    Args:
        num_samples: Number of samples to visualize per batch (default: 4)
        slice_range: Range around center for random slice selection (default: 3)
        output_dir: Directory to save visualizations (default: "./slices")
        enable_tensorboard: Whether to log to TensorBoard (default: True)
    """

    def __init__(
        self,
        num_samples: int = 4,
        slice_range: int = 3,
        output_dir: str = "./slices",
        enable_tensorboard: bool = True,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.slice_range = slice_range
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_tensorboard = enable_tensorboard
        self._tensorboard_writer: SummaryWriter | None = None

    def set_writer(self, writer: Any) -> None:
        """Set TensorBoard writer."""
        self._tensorboard_writer = writer

    def on_validation_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        """Initialize TensorBoard writer at start of validation."""
        super().on_validation_start(trainer, pl_module)
        if self.enable_tensorboard:
            self._tensorboard_writer = SummaryWriter(log_dir=str(self.output_dir))

    def on_test_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        """Initialize TensorBoard writer at start of testing."""
        super().on_test_start(trainer, pl_module)
        if self.enable_tensorboard:
            self._tensorboard_writer = SummaryWriter(log_dir=str(self.output_dir))

    def _extract_random_slice(self, volume: torch.Tensor) -> np.ndarray:
        """
        Extract a random axial slice from a 3D volume.

        Args:
            volume: 3D tensor of shape [B, C, D, H, W]

        Returns:
            2D numpy array of shape [H, W] with values normalized to [0, 1]
        """
        # volume shape: [B, C, D, H, W]
        depth = volume.shape[2]
        height = volume.shape[3]
        width = volume.shape[4]

        # Select random depth index near center
        center = depth // 2
        min_depth = max(0, center - self.slice_range)
        max_depth = min(depth - 1, center + self.slice_range)
        slice_idx = random.randint(min_depth, max_depth)

        # Extract slice at the selected depth
        slice_2d = volume[0, 0, slice_idx, :, :].cpu().numpy()

        # Normalize from [-1, 1] to [0, 1] range
        slice_2d = (slice_2d + 1.0) / 2.0
        slice_2d = np.clip(slice_2d, 0.0, 1.0)

        return slice_2d

    def _save_to_disk(self, image: torch.Tensor | np.ndarray, filepath: str) -> None:
        """
        Save an image to disk as PNG.

        Args:
            image: 2D array with values in [0, 1] range
            filepath: Path to save the image
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # Convert to uint8 for PIL
        image_uint8 = (image * 255).astype(np.uint8)

        # Save as PNG
        img_pil = Image.fromarray(image_uint8, mode="L")
        img_pil.save(filepath)

    def _log_to_tensorboard(self, image: torch.Tensor | np.ndarray, tag: str, step: int) -> None:
        """
        Log an image to TensorBoard.

        Args:
            image: 2D array with values in [0, 1] range
            tag: Tag for the image
            step: Global step for logging
        """
        if self._tensorboard_writer is None:
            return

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Add channel dimension for add_image (expects [C, H, W])
        image = image.unsqueeze(0)

        self._tensorboard_writer.add_image(tag, image, global_step=step)

    def _process_batch(self, batch: dict[str, Any], batch_idx: int, prefix: str) -> None:
        """
        Process a batch to visualize random axial slices.

        Args:
            batch: Dictionary containing 'volumes' key with 3D tensor
            batch_idx: Index of the batch
            prefix: Prefix for logging tags (e.g., 'val' or 'test')
        """
        volumes = batch.get("volumes")
        if volumes is None:
            return

        # Get global step from trainer if available
        global_step = 0
        if self._trainer is not None:
            global_step = self._trainer.global_step

        # Process up to num_samples
        num_to_process = min(self.num_samples, volumes.shape[0])

        for i in range(num_to_process):
            volume = volumes[i : i + 1]  # Keep batch dimension

            # Extract random slice
            slice_img = self._extract_random_slice(volume)

            # Save to disk
            filepath = self.output_dir / f"{prefix}_batch{batch_idx}_sample{i}.png"
            self._save_to_disk(slice_img, str(filepath))

            # Log to tensorboard
            if self.enable_tensorboard and self._tensorboard_writer is not None:
                tag = f"{prefix}/batch{batch_idx}/sample{i}"
                self._log_to_tensorboard(slice_img, tag, global_step)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: torch.nn.Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Visualize random axial slices after each validation batch."""
        self._process_batch(batch, batch_idx, "val")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: torch.nn.Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Visualize random axial slices after each test batch."""
        self._process_batch(batch, batch_idx, "test")

    def on_validation_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        """Close TensorBoard writer at end of validation."""
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.close()
            self._tensorboard_writer = None
        super().on_validation_end(trainer, pl_module)

    def on_test_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        """Close TensorBoard writer at end of testing."""
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.close()
            self._tensorboard_writer = None
        super().on_test_end(trainer, pl_module)


# Backward compatibility aliases
class FabricModelCheckpoint(ModelCheckpoint):
    """
    Backward compatible alias for ModelCheckpoint.

    This class maintains compatibility with the old Fabric-based API
    where callbacks receive fabric, model, and optimizer as separate arguments.
    """

    def on_validation_epoch_end(  # type: ignore[override]
        self,
        fabric: Fabric,
        model: Any,
        optimizer: Any,
    ) -> None:
        """Legacy method for FabricTrainingPipeline compatibility."""
        # For Fabric, we need to get metrics from fabric._current_metrics
        if hasattr(fabric, "_current_metrics"):
            score = fabric._current_metrics.get(self.monitor)  # type: ignore[union-attr]
        else:
            score = None

        if score is None:
            return

        if self.save_last:
            self._save_checkpoint_fabric(fabric, model, optimizer, "last")

        if self.save_top_k != 0:
            self._update_best_checkpoints_fabric(fabric, model, optimizer, score)

        self._current_epoch += 1

    def _update_best_checkpoints_fabric(
        self,
        fabric: Fabric,
        model: Any,
        optimizer: Any,
        score: float,
    ) -> None:
        is_better = False
        if not self._best_scores:
            is_better = True
        elif self.mode == "min":
            is_better = score < self._best_scores[-1][0]
        else:
            is_better = score > self._best_scores[-1][0]

        if is_better or len(self._best_scores) < self.save_top_k or self.save_top_k == -1:
            filename = self.filename.format(
                epoch=self._current_epoch,
                **{self.monitor: score},
            )

            self._save_checkpoint_fabric(fabric, model, optimizer, filename)
            filepath = self.dirpath / f"{filename}.ckpt"
            self._best_scores.append((score, filepath))
            self._best_scores.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

            if self.save_top_k > 0:
                while len(self._best_scores) > self.save_top_k:
                    _, old_path = self._best_scores.pop()
                    if old_path.exists():
                        old_path.unlink()

    def _save_checkpoint_fabric(
        self,
        fabric: Fabric,
        model: Any,
        optimizer: Any,
        filename: str,
    ) -> None:
        filepath = self.dirpath / f"{filename}.ckpt"

        checkpoint = {
            "epoch": self._current_epoch,
            "model": model,
            "optimizer": optimizer,
        }

        fabric.save(filepath, checkpoint)

    def on_fit_start(self, fabric: Fabric) -> None:  # type: ignore[override]
        """Legacy method for FabricTrainingPipeline compatibility."""
        self._current_epoch = 0
        self._best_scores = []


class FabricEarlyStopping(EarlyStopping):
    """
    Backward compatible alias for EarlyStopping.

    This class maintains compatibility with the old Fabric-based API.
    """

    def on_validation_epoch_end(  # type: ignore[override]
        self,
        fabric: Fabric,
        model: Any,
        optimizer: Any,
    ) -> None:
        """Legacy method for FabricTrainingPipeline compatibility."""
        # For Fabric, we need to get metrics from fabric._current_metrics
        if hasattr(fabric, "_current_metrics"):
            score = fabric._current_metrics.get(self.monitor)  # type: ignore[union-attr]
        else:
            score = None

        if score is None:
            return

        current = torch.tensor(score)

        if self.check_finite and not torch.isfinite(current):
            self.should_stop = True
            if fabric.is_global_zero:
                logger.warning(
                    f"Early stopping triggered: {self.monitor} is not finite ({current})"
                )
            return

        if self._early_stopping.best_score is None:
            self._early_stopping.best_score = current
            self._early_stopping.wait_count = 0
            return

        monitor_op = self._early_stopping.monitor_op

        if monitor_op(current - self.min_delta, self._early_stopping.best_score):
            self._early_stopping.best_score = current
            self._early_stopping.wait_count = 0
        else:
            self._early_stopping.wait_count += 1
            if self._early_stopping.wait_count >= self.patience:
                self.should_stop = True
                if fabric.is_global_zero:
                    logger.warning(
                        f"Early stopping triggered after {self.patience} epochs without improvement. "
                        f"Best {self.monitor}: {self._early_stopping.best_score.item():.6f}"
                    )

    def on_fit_start(self, fabric: Fabric) -> None:  # type: ignore[override]
        """Legacy method for FabricTrainingPipeline compatibility."""
        torch_inf = torch.tensor(torch.inf)
        self._early_stopping.best_score = (
            torch_inf if self._early_stopping.monitor_op == torch.lt else -torch_inf
        )
        self._early_stopping.wait_count = 0
        self._early_stopping.stopped_epoch = 0
        self.should_stop = False


class FabricNaNMonitor(NaNMonitor):
    """
    Backward compatible alias for NaNMonitor.

    This class maintains compatibility with the old Fabric-based API
    where the loss is passed explicitly.
    """

    def on_train_batch_end(  # type: ignore[override]
        self,
        fabric: Fabric,
        model: Any,
        optimizer: Any,
        batch: Any,
        batch_idx: int,
        loss: float,
    ) -> None:
        """Legacy method for FabricTrainingPipeline compatibility."""
        self._batch_count += 1

        if self._batch_count % self.check_interval != 0:
            return

        if math.isnan(loss) or math.isinf(loss):
            self._handle_nan("loss", loss)
            return

        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and (
                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
            ):
                has_nan_grad = True
                if fabric.is_global_zero:
                    logger.warning(f"NaN/Inf detected in gradient of {name}")
                break

        if has_nan_grad:
            self._handle_nan("gradient", None)

    def on_fit_start(self, fabric: Fabric) -> None:  # type: ignore[override]
        """Legacy method for FabricTrainingPipeline compatibility."""
        self._batch_count = 0


class FabricMetricsLogger(MetricsLogger):
    """
    Backward compatible alias for MetricsLogger.

    This class maintains compatibility with the old Fabric-based API.
    """

    def on_train_epoch_end(  # type: ignore[override]
        self,
        fabric: Fabric,
        model: Any,
        optimizer: Any,
    ) -> None:
        """Legacy method for FabricTrainingPipeline compatibility."""
        self._epoch += 1

        if self._epoch % self.log_interval != 0:
            return

        current_metrics = getattr(fabric, "_current_metrics", {})

        for key, value in current_metrics.items():
            if key not in self._history:
                self._history[key] = []
            self._history[key].append(value)

        self._save_metrics()

    def on_fit_end(self, fabric: Fabric) -> None:  # type: ignore[override]
        """Legacy method for FabricTrainingPipeline compatibility."""
        self._save_metrics()
