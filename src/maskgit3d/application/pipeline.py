"""
Application layer - Training and Test pipelines.

This module provides high-level pipelines that orchestrate
the training, validation, and testing workflows using Lightning Fabric.
"""

from pathlib import Path
from typing import Any, Literal

import torch
from tqdm import tqdm

from maskgit3d.infrastructure.checkpoints import load_checkpoint as load_ckpt

try:
    import lightning as L

    LIGHTNING_AVAILABLE = True
except ImportError:
    L = None
    LIGHTNING_AVAILABLE = False

from maskgit3d.domain.interfaces import (
    DataProvider,
    InferenceStrategy,
    Metrics,
    ModelInterface,
    OptimizerFactory,
    TrainingStrategy,
)


class TestPipeline:
    """
    Pipeline for inference and testing.

    Handles model loading, inference, and metrics computation
    for test data.
    """

    def __init__(
        self,
        model: ModelInterface,
        data_provider: DataProvider,
        inference_strategy: InferenceStrategy,
        metrics: Metrics | None = None,
        device: torch.device | None = None,
        output_dir: str = "./outputs",
    ):
        self.model = model
        self.data_provider = data_provider
        self.inference_strategy = inference_strategy
        self.metrics = metrics
        self.output_dir = Path(output_dir)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        checkpoint_path: str | None = None,
        save_predictions: bool = False,
    ) -> dict[str, float]:
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        if self.metrics:
            self.metrics.reset()

        all_predictions = []

        test_loader = self.data_provider.test_loader()

        pbar = tqdm(test_loader, desc="Testing")
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                if isinstance(batch, tuple | list):
                    images = batch[0].to(self.device)
                    targets = batch[1].to(self.device) if len(batch) > 1 else None
                else:
                    images = batch.to(self.device)
                    targets = None

                raw_predictions = self.inference_strategy.predict(self.model, images)
                processed = self.inference_strategy.post_process(raw_predictions)

                all_predictions.append(processed)

                if self.metrics and targets is not None:
                    self.metrics.update(processed, targets)

                if save_predictions:
                    self._save_predictions(processed, batch_idx)

                pbar.set_postfix({"batch": batch_idx + 1})

        if self.metrics:
            final_metrics = self.metrics.compute()
            print("Test Results:")
            for key, value in final_metrics.items():
                print(f"  {key}: {value:.4f}")
            return final_metrics

        return {"num_samples": len(all_predictions)}

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = load_ckpt(checkpoint_path, map_location=self.device)
        # Supports: {"model_state_dict": ...}, {"model": ...}, {"state_dict": ...}, or raw state_dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")

    def _save_predictions(
        self,
        predictions: dict[str, Any],
        batch_idx: int,
    ) -> None:
        import numpy as np

        masks_path = self.output_dir / f"predictions_batch_{batch_idx}.npy"
        np.save(masks_path, predictions["masks"])

        probs_path = self.output_dir / f"probabilities_batch_{batch_idx}.npy"
        np.save(probs_path, predictions["probs"])


class FabricTrainingPipeline:
    """
    Pipeline for training with Lightning Fabric.

    Provides lightweight distributed training support using Lightning Fabric API
    while maintaining the dependency injection architecture.

    Supports Lightning Callbacks for extensibility (checkpointing, logging, etc.)
    """

    def __init__(
        self,
        model: ModelInterface,
        data_provider: DataProvider,
        training_strategy: TrainingStrategy,
        optimizer_factory: OptimizerFactory,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        strategy: str = "auto",
        precision: Literal[
            64,
            32,
            16,
            "transformer-engine",
            "transformer-engine-float16",
            "16-true",
            "16-mixed",
            "bf16-true",
            "bf16-mixed",
            "32-true",
            "64-true",
            "64",
            "32",
            "16",
            "bf16",
        ]
        | None = "32-true",
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 10,
        callbacks: list[Any] | None = None,
    ):
        """
        Initialize Fabric training pipeline.

        Args:
            model: Model to train (implements ModelInterface)
            data_provider: Data provider for train/val loaders
            training_strategy: Training strategy with loss computation
            optimizer_factory: Factory for creating optimizer
            accelerator: Fabric accelerator ("cpu", "cuda", "auto")
            devices: Number of devices or device IDs
            strategy: Fabric strategy ("auto", "ddp", "fsdp")
            precision: Training precision ("32-true", "16-mixed", "bf16-mixed")
            checkpoint_dir: Directory to save checkpoints
            log_interval: Interval for logging training progress
            callbacks: List of Lightning Callback instances
        """
        if not LIGHTNING_AVAILABLE:
            raise ImportError("Lightning is not installed. Install with: pip install lightning")

        self.model = model
        self.data_provider = data_provider
        self.training_strategy = training_strategy
        self.optimizer_factory = optimizer_factory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.callbacks = callbacks or []

        self._accelerator = accelerator
        self._devices = devices
        self._strategy = strategy
        self._precision = precision

        self._fabric: Any = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._current_epoch = 0
        self._global_step = 0

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        num_epochs: int,
        val_frequency: int = 1,
        resume_from: str | None = None,
    ) -> dict[str, list[float]]:
        """
        Run the training loop with Fabric.

        Args:
            num_epochs: Number of epochs to train
            val_frequency: Run validation every N epochs
            resume_from: Path to checkpoint to resume from

        Returns:
            Dictionary of training history
        """
        self._fabric = L.Fabric(
            accelerator=self._accelerator,
            devices=self._devices,
            strategy=self._strategy,
            precision=self._precision,
            callbacks=self.callbacks,
        )
        self._fabric.launch()

        self._optimizer = self.optimizer_factory.create(self.model.parameters())

        self.model, self._optimizer = self._fabric.setup(self.model, self._optimizer)

        train_loader = self._fabric.setup_dataloaders(self.data_provider.train_loader())
        val_loader = self._fabric.setup_dataloaders(self.data_provider.val_loader())

        start_epoch = 0
        if resume_from:
            start_epoch = self._load_checkpoint(resume_from)

        self._call_callbacks("on_fit_start")

        history: dict[str, list[float]] = {}

        for epoch in range(start_epoch, num_epochs):
            self._current_epoch = epoch
            self._call_callbacks("on_train_epoch_start")

            train_metrics = self._train_epoch(epoch, train_loader)

            for key, values in train_metrics.items():
                history.setdefault(key, []).extend(values)

            self._call_callbacks("on_train_epoch_end")

            if (epoch + 1) % val_frequency == 0:
                self._call_callbacks("on_validation_epoch_start")
                val_metrics = self._validate_epoch(epoch, val_loader)

                for key, values in val_metrics.items():
                    history.setdefault(key, []).extend(values)

                self._call_callbacks("on_validation_epoch_end")

                self._print_epoch_summary(epoch, num_epochs, history)
                self._save_checkpoint(epoch, train_metrics, val_metrics)
            else:
                self._print_epoch_summary(epoch, num_epochs, history, has_validation=False)

        self._call_callbacks("on_fit_end")

        return history

    def _train_epoch(
        self,
        epoch: int,
        train_loader: Any,
    ) -> dict[str, list[float]]:
        """Run one training epoch with Fabric."""
        self.model.train()

        metrics_history: dict[str, list[float]] = {}

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            self._call_callbacks("on_train_batch_start", batch, batch_idx)

            x = batch[0] if isinstance(batch, tuple | list) else batch

            output = self.model(x)

            if hasattr(self.training_strategy, "compute_loss"):
                loss = self.training_strategy.compute_loss(self.model, batch, output)
            else:
                target = batch[1] if isinstance(batch, tuple | list) and len(batch) > 1 else x
                loss = torch.nn.functional.mse_loss(output, target)

            self._optimizer.zero_grad()
            self._fabric.backward(loss)
            self._optimizer.step()

            self._global_step += 1

            loss_value = loss.item()
            metrics_history.setdefault("train_loss", []).append(loss_value)

            self._call_callbacks("on_train_batch_end", batch, batch_idx, loss_value)

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss_value:.4f}"})

        return metrics_history

    def _validate_epoch(
        self,
        epoch: int,
        val_loader: Any,
    ) -> dict[str, list[float]]:
        """Run one validation epoch."""
        self.model.eval()

        metrics_history: dict[str, list[float]] = {}

        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                self._call_callbacks("on_validation_batch_start", batch, batch_idx)

                metrics = self.training_strategy.validate_step(self.model, batch)

                for key, value in metrics.items():
                    if not isinstance(value, int | float):
                        continue
                    prefixed_key = key if key.startswith("val_") else f"val_{key}"
                    metrics_history.setdefault(prefixed_key, []).append(float(value))

                self._call_callbacks("on_validation_batch_end", batch, batch_idx)

                if "val_loss" in metrics:
                    pbar.set_postfix({"val_loss": f"{metrics['val_loss']:.4f}"})

        return metrics_history

    def _call_callbacks(self, hook_name: str, *args, **kwargs) -> None:
        """Call callback hooks if they exist."""
        for callback in self.callbacks:
            if hasattr(callback, hook_name):
                method = getattr(callback, hook_name)
                if "fabric" in hook_name or "fit" in hook_name:
                    method(self._fabric, *args, **kwargs)
                else:
                    method(self._fabric, self.model, self._optimizer, *args, **kwargs)

    def _print_epoch_summary(
        self,
        epoch: int,
        num_epochs: int,
        history: dict[str, list[float]],
        has_validation: bool = True,
    ) -> None:
        """Print epoch summary."""
        train_loss_key = "train_loss" if "train_loss" in history else None
        val_loss_key = "val_loss" if "val_loss" in history else None

        if train_loss_key and history[train_loss_key]:
            avg_train_loss = sum(history[train_loss_key]) / len(history[train_loss_key])
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}", end="")
            if has_validation and val_loss_key and history[val_loss_key]:
                avg_val_loss = sum(history[val_loss_key]) / len(history[val_loss_key])
                print(f", Val Loss: {avg_val_loss:.4f}")
            else:
                print()
        else:
            print(f"Epoch {epoch + 1}/{num_epochs}")

    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: dict[str, list[float]],
        val_metrics: dict[str, list[float]],
    ) -> None:
        """Save model checkpoint with Fabric."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.ckpt"

        checkpoint_data: dict[str, Any] = {
            "epoch": epoch + 1,
            "model": self.model,
            "optimizer": self._optimizer,
            "global_step": self._global_step,
        }

        if "train_loss" in train_metrics and train_metrics["train_loss"]:
            checkpoint_data["train_loss"] = sum(train_metrics["train_loss"]) / len(
                train_metrics["train_loss"]
            )

        if "val_loss" in val_metrics and val_metrics["val_loss"]:
            checkpoint_data["val_loss"] = sum(val_metrics["val_loss"]) / len(
                val_metrics["val_loss"]
            )

        self._fabric.save(str(checkpoint_path), checkpoint_data)
        print(f"Checkpoint saved to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint with Fabric and return starting epoch."""
        checkpoint = self._fabric.load(checkpoint_path)

        start_epoch = checkpoint.get("epoch", 0)
        self._global_step = checkpoint.get("global_step", 0)
        print(f"Resumed from epoch {start_epoch}, checkpoint: {checkpoint_path}")

        return start_epoch

    @property
    def global_step(self) -> int:
        """Get current global step."""
        return self._global_step

    @property
    def current_epoch(self) -> int:
        """Get current epoch."""
        return self._current_epoch
