"""
Application layer - Training and Test pipelines.

This module provides high-level pipelines that orchestrate
the training, validation, and testing workflows.
"""

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from maskgit3d.infrastructure.checkpoints import load_checkpoint as load_ckpt

# Try importing lightning, handle gracefully if not available
try:
    import lightning as L

    LIGHTNING_AVAILABLE = True
except ImportError:
    L = None  # type: ignore
    LIGHTNING_AVAILABLE = False

from maskgit3d.domain.interfaces import (
    DataProvider,
    InferenceStrategy,
    Metrics,
    ModelInterface,
    OptimizerFactory,
    TrainingStrategy,
)


class TrainingPipeline:
    """
    Pipeline for training and validation.

    Orchestrates the complete training loop including:
    - Data loading
    - Forward/backward passes
    - Validation
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        model: ModelInterface,
        data_provider: DataProvider,
        training_strategy: TrainingStrategy,
        optimizer_factory: OptimizerFactory,
        device: torch.device | None = None,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 10,
    ):
        """
        Initialize training pipeline.

        Args:
            model: Model to train
            data_provider: Data provider for train/val data
            training_strategy: Training strategy with loss and metrics
            optimizer_factory: Factory for creating optimizer
            device: Device to use for training
            checkpoint_dir: Directory to save checkpoints
            log_interval: Interval for logging training progress
        """
        self.model = model
        self.data_provider = data_provider
        self.training_strategy = training_strategy
        self.optimizer_factory = optimizer_factory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Move model to device - model already implements ModelInterface
        self.model.to(self.device)

        # Create optimizer
        self.optimizer = self.optimizer_factory.create(self.model.parameters())

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        num_epochs: int,
        val_frequency: int = 1,
        resume_from: str | None = None,
    ) -> dict[str, list[float]]:
        """
        Run the training loop.

        Args:
            num_epochs: Number of epochs to train
            val_frequency: Run validation every N epochs
            resume_from: Path to checkpoint to resume from

        Returns:
            Dictionary of training history
        """
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            start_epoch = self._load_checkpoint(resume_from)

        # Initialize empty history - will be populated dynamically
        history: dict[str, list[float]] = {}

        for epoch in range(start_epoch, num_epochs):
            # Training phase
            train_metrics = self._train_epoch(epoch)

            # Dynamically extend history with train metrics
            for key, values in train_metrics.items():
                history.setdefault(key, []).extend(values)

            # Validation phase
            if (epoch + 1) % val_frequency == 0:
                val_metrics = self._validate_epoch(epoch)

                # Dynamically extend history with validation metrics
                for key, values in val_metrics.items():
                    history.setdefault(key, []).extend(values)

                # Print epoch summary - use first available loss key
                train_loss_key = "train_loss" if "train_loss" in history else None
                val_loss_key = "val_loss" if "val_loss" in history else None

                if train_loss_key and history[train_loss_key]:
                    avg_train_loss = sum(history[train_loss_key]) / len(history[train_loss_key])
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}", end=""
                    )
                    if val_loss_key and history[val_loss_key]:
                        avg_val_loss = sum(history[val_loss_key]) / len(history[val_loss_key])
                        print(f", Val Loss: {avg_val_loss:.4f}")
                    else:
                        print()
                else:
                    print(f"Epoch {epoch + 1}/{num_epochs}")

                # Save checkpoint
                self._save_checkpoint(epoch, train_metrics, val_metrics)
            else:
                # Print epoch summary (without validation) - use first available loss key
                train_loss_key = "train_loss" if "train_loss" in history else None
                if train_loss_key and history[train_loss_key]:
                    avg_train_loss = sum(history[train_loss_key]) / len(history[train_loss_key])
                    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
                else:
                    print(f"Epoch {epoch + 1}/{num_epochs}")

        return history

    def _train_epoch(self, epoch: int) -> dict[str, list[float]]:
        """Run one training epoch."""
        self.model.train()

        # Collect all metrics dynamically
        metrics_history: dict[str, list[float]] = {}

        train_loader = self.data_provider.train_loader()

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for _batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Training step
            metrics = self.training_strategy.train_step(self.model, batch, self.optimizer)

            # Dynamically collect all metrics from strategy
            for key, value in metrics.items():
                # Skip non-numeric values
                if not isinstance(value, int | float):
                    continue
                # Prefix with train_ if not already prefixed
                prefixed_key = key if key.startswith("train_") else f"train_{key}"
                metrics_history.setdefault(prefixed_key, []).append(float(value))

            # Logging - use loss if available
            log_dict = {}
            if "loss" in metrics:
                log_dict["loss"] = f"{metrics['loss']:.4f}"
            if "train_loss" in metrics:
                log_dict["loss"] = f"{metrics['train_loss']:.4f}"
            if log_dict:
                pbar.set_postfix(log_dict)

        return metrics_history

    def _validate_epoch(self, epoch: int) -> dict[str, list[float]]:
        """Run one validation epoch."""
        self.model.eval()

        # Collect all metrics dynamically
        metrics_history: dict[str, list[float]] = {}

        val_loader = self.data_provider.val_loader()

        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
        with torch.no_grad():
            for _batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Validation step
                metrics = self.training_strategy.validate_step(self.model, batch)

                # Dynamically collect all metrics from strategy
                for key, value in metrics.items():
                    # Skip non-numeric values
                    if not isinstance(value, int | float):
                        continue
                    # Use val_ prefix if not already prefixed
                    prefixed_key = key if key.startswith("val_") else f"val_{key}"
                    metrics_history.setdefault(prefixed_key, []).append(float(value))

                # Logging - use val_loss if available
                log_dict = {}
                if "val_loss" in metrics:
                    log_dict["loss"] = f"{metrics['val_loss']:.4f}"
                if log_dict:
                    pbar.set_postfix(log_dict)

        return metrics_history

    def _move_batch_to_device(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Move batch tensors to the training device."""
        return tuple(item.to(self.device) for item in batch)

    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: dict[str, list[float]],
        val_metrics: dict[str, list[float]],
    ) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"

        # Get model state dict - model implements ModelInterface
        model_state = self.model.state_dict()

        # Collect checkpoint data dynamically
        checkpoint_data: dict[str, Any] = {
            "epoch": epoch + 1,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        # Add train loss if available
        if "train_loss" in train_metrics and train_metrics["train_loss"]:
            checkpoint_data["train_loss"] = sum(train_metrics["train_loss"]) / len(
                train_metrics["train_loss"]
            )

        # Add val loss if available
        if "val_loss" in val_metrics and val_metrics["val_loss"]:
            checkpoint_data["val_loss"] = sum(val_metrics["val_loss"]) / len(
                val_metrics["val_loss"]
            )

        torch.save(checkpoint_data, checkpoint_path)

        print(f"Checkpoint saved to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return starting epoch."""
        # Use centralized checkpoint loader with security best practices
        checkpoint = load_ckpt(checkpoint_path, map_location=self.device)

        # Load model weights - model implements ModelInterface
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}, checkpoint: {checkpoint_path}")

        return start_epoch


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
        """
        Initialize test pipeline.

        Args:
            model: Model for inference
            data_provider: Data provider for test data
            inference_strategy: Inference strategy for prediction
            metrics: Metrics for evaluation (optional)
            device: Device to use for inference
            output_dir: Directory to save outputs
        """
        self.model = model
        self.data_provider = data_provider
        self.inference_strategy = inference_strategy
        self.metrics = metrics
        self.output_dir = Path(output_dir)

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Move model to device and set to eval mode - model implements ModelInterface
        self.model.to(self.device)
        self.model.eval()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        checkpoint_path: str | None = None,
        save_predictions: bool = False,
    ) -> dict[str, float]:
        """
        Run inference on test data.

        Args:
            checkpoint_path: Path to model checkpoint (if not already loaded)
            save_predictions: Whether to save predictions to disk

        Returns:
            Dictionary of test metrics
        """
        # Load checkpoint if specified
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        # Reset metrics
        if self.metrics:
            self.metrics.reset()

        all_predictions = []

        test_loader = self.data_provider.test_loader()

        pbar = tqdm(test_loader, desc="Testing")
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                if isinstance(batch, tuple | list):
                    images = batch[0].to(self.device)
                    targets = batch[1].to(self.device) if len(batch) > 1 else None
                else:
                    images = batch.to(self.device)
                    targets = None

                # Run inference
                raw_predictions = self.inference_strategy.predict(self.model, images)
                processed = self.inference_strategy.post_process(raw_predictions)

                all_predictions.append(processed)

                # Update metrics if we have targets
                if self.metrics and targets is not None:
                    self.metrics.update(processed, targets)

                # Save predictions if requested
                if save_predictions:
                    self._save_predictions(processed, batch_idx)

                pbar.set_postfix({"batch": batch_idx + 1})

        # Compute final metrics
        if self.metrics:
            final_metrics = self.metrics.compute()
            print("Test Results:")
            for key, value in final_metrics.items():
                print(f"  {key}: {value:.4f}")
            return final_metrics

        return {"num_samples": len(all_predictions)}

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        # Use centralized checkpoint loader with security best practices
        checkpoint = load_ckpt(checkpoint_path, map_location=self.device)

        # Load model weights - model implements ModelInterface
        self.model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Loaded checkpoint from {checkpoint_path}")

    def _save_predictions(
        self,
        predictions: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Save predictions to disk."""
        import numpy as np

        # Save masks
        masks_path = self.output_dir / f"predictions_batch_{batch_idx}.npy"
        np.save(masks_path, predictions["masks"])

        # Save probabilities
        probs_path = self.output_dir / f"probabilities_batch_{batch_idx}.npy"
        np.save(probs_path, predictions["probs"])




class FabricTrainingPipeline:
    """
    Pipeline for training with Lightning Fabric.

    Provides lightweight distributed training support using Lightning Fabric API
    while maintaining the dependency injection architecture.

    This pipeline gives full control over the training loop, unlike the heavier
    PyTorch Lightning Module approach.
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
        precision: str = "32-true",
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 10,
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
        """
        if not LIGHTNING_AVAILABLE:
            raise ImportError(
                "Lightning is not installed. Install with: pip install lightning"
            )

        self.model = model
        self.data_provider = data_provider
        self.training_strategy = training_strategy
        self.optimizer_factory = optimizer_factory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval

        # Fabric configuration
        self._accelerator = accelerator
        self._devices = devices
        self._strategy = strategy
        self._precision = precision

        # Fabric instance (initialized in run())
        self._fabric: Any = None
        self._optimizer: torch.optim.Optimizer | None = None

        # Ensure checkpoint directory exists
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
        # Initialize Fabric
        self._fabric = L.Fabric(  # type: ignore[misc]
            accelerator=self._accelerator,
            devices=self._devices,
            strategy=self._strategy,
            precision=self._precision,
        )
        self._fabric.launch()

        # Create optimizer
        self._optimizer = self.optimizer_factory.create(self.model.parameters())

        # Setup model and optimizer with Fabric
        self.model, self._optimizer = self._fabric.setup(  # type: ignore[misc]
            self.model, self._optimizer
        )

        # Setup dataloaders with Fabric
        train_loader = self._fabric.setup_dataloaders(  # type: ignore[misc]
            self.data_provider.train_loader()
        )
        val_loader = self._fabric.setup_dataloaders(  # type: ignore[misc]
            self.data_provider.val_loader()
        )

        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            start_epoch = self._load_checkpoint(resume_from)

        # Initialize empty history
        history: dict[str, list[float]] = {}

        for epoch in range(start_epoch, num_epochs):
            # Training phase
            train_metrics = self._train_epoch(epoch, train_loader)

            # Dynamically extend history with train metrics
            for key, values in train_metrics.items():
                history.setdefault(key, []).extend(values)

            # Validation phase
            if (epoch + 1) % val_frequency == 0:
                val_metrics = self._validate_epoch(epoch, val_loader)

                # Dynamically extend history with validation metrics
                for key, values in val_metrics.items():
                    history.setdefault(key, []).extend(values)

                # Print epoch summary
                self._print_epoch_summary(epoch, num_epochs, history)

                # Save checkpoint
                self._save_checkpoint(epoch, train_metrics, val_metrics)
            else:
                # Print epoch summary (without validation)
                self._print_epoch_summary(epoch, num_epochs, history, has_validation=False)

        return history

    def _train_epoch(
        self,
        epoch: int,
        train_loader: Any,
    ) -> dict[str, list[float]]:
        """Run one training epoch with Fabric."""
        self.model.train()

        # Collect all metrics dynamically
        metrics_history: dict[str, list[float]] = {}

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            x = batch[0] if isinstance(batch, tuple | list) else batch

            # Forward pass
            output = self.model(x)

            # Compute loss using training strategy's approach
            # Note: We don't use train_step because it calls backward internally
            # Instead, we compute loss and handle backward ourselves
            if hasattr(self.training_strategy, "compute_loss"):
                # Use strategy's loss computation if available
                loss = self.training_strategy.compute_loss(self.model, batch, output)  # type: ignore[attr-defined]
            else:
                # Fallback: use a simple loss computation
                # Strategies should implement compute_loss for Fabric compatibility
                target = batch[1] if isinstance(batch, tuple | list) and len(batch) > 1 else x
                loss = torch.nn.functional.mse_loss(output, target)

            # Fabric backward
            self._optimizer.zero_grad()  # type: ignore[union-attr]
            self._fabric.backward(loss)  # type: ignore[misc]
            self._optimizer.step()  # type: ignore[union-attr]

            # Collect metrics
            loss_value = loss.item()
            metrics_history.setdefault("train_loss", []).append(loss_value)

            # Logging
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

        # Collect all metrics dynamically
        metrics_history: dict[str, list[float]] = {}

        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
        with torch.no_grad():
            for _batch_idx, batch in enumerate(pbar):
                # Use strategy's validation step (doesn't need backward)
                metrics = self.training_strategy.validate_step(self.model, batch)

                # Dynamically collect all metrics from strategy
                for key, value in metrics.items():
                    if not isinstance(value, int | float):
                        continue
                    prefixed_key = key if key.startswith("val_") else f"val_{key}"
                    metrics_history.setdefault(prefixed_key, []).append(float(value))

                # Logging
                if "val_loss" in metrics:
                    pbar.set_postfix({"val_loss": f"{metrics['val_loss']:.4f}"})

        return metrics_history

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

        # Collect checkpoint data
        checkpoint_data: dict[str, Any] = {
            "epoch": epoch + 1,
            "model": self.model,
            "optimizer": self._optimizer,
        }

        # Add train loss if available
        if "train_loss" in train_metrics and train_metrics["train_loss"]:
            checkpoint_data["train_loss"] = sum(train_metrics["train_loss"]) / len(
                train_metrics["train_loss"]
            )

        # Add val loss if available
        if "val_loss" in val_metrics and val_metrics["val_loss"]:
            checkpoint_data["val_loss"] = sum(val_metrics["val_loss"]) / len(
                val_metrics["val_loss"]
            )

        # Save with Fabric
        self._fabric.save(str(checkpoint_path), checkpoint_data)  # type: ignore[misc]
        print(f"Checkpoint saved to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint with Fabric and return starting epoch."""
        # Load with Fabric
        checkpoint = self._fabric.load(checkpoint_path)  # type: ignore[misc]

        # Get starting epoch
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}, checkpoint: {checkpoint_path}")

        return start_epoch


