"""
Application layer - Training and Test pipelines.

This module provides high-level pipelines that orchestrate
the training, validation, and testing workflows.
"""
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm

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
        device: Optional[torch.device] = None,
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
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        # Move model to device
        self.model.model.to(self.device) if hasattr(self.model, "model") \
            else self.model.to(self.device)

        # Create optimizer
        self.optimizer = self.optimizer_factory.create(
            self.model.parameters()
        )

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        num_epochs: int,
        val_frequency: int = 1,
        resume_from: Optional[str] = None,
    ) -> Dict[str, List[float]]:
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

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_dice": [],
            "val_dice": [],
        }

        for epoch in range(start_epoch, num_epochs):
            # Training phase
            train_metrics = self._train_epoch(epoch)
            history["train_loss"].extend(train_metrics["losses"])
            history["train_dice"].extend(train_metrics["dice_scores"])

            # Validation phase
            if (epoch + 1) % val_frequency == 0:
                val_metrics = self._validate_epoch(epoch)
                history["val_loss"].extend(val_metrics["losses"])
                history["val_dice"].extend(val_metrics["dice_scores"])

                # Print epoch summary
                avg_train_loss = sum(train_metrics["losses"]) / len(
                    train_metrics["losses"]
                )
                avg_val_loss = sum(val_metrics["losses"]) / len(
                    val_metrics["losses"]
                )
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}"
                )

                # Save checkpoint
                self._save_checkpoint(epoch, train_metrics, val_metrics)
            else:
                # Print epoch summary (without validation)
                avg_train_loss = sum(train_metrics["losses"]) / len(
                    train_metrics["losses"]
                )
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}"
                )

        return history

    def _train_epoch(self, epoch: int) -> Dict[str, List[float]]:
        """Run one training epoch."""
        self.model.model.train() if hasattr(self.model, "model") \
            else self.model.train()

        losses = []
        dice_scores = []

        train_loader = self.data_provider.train_loader()

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Training step
            metrics = self.training_strategy.train_step(
                self.model, batch, self.optimizer
            )

            losses.append(metrics.get("loss", 0))
            dice_scores.append(metrics.get("dice_score", 0))

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "dice": f"{metrics.get('dice_score', 0):.4f}",
                })

        return {"losses": losses, "dice_scores": dice_scores}

    def _validate_epoch(self, epoch: int) -> Dict[str, List[float]]:
        """Run one validation epoch."""
        self.model.model.eval() if hasattr(self.model, "model") \
            else self.model.eval()

        losses = []
        dice_scores = []

        val_loader = self.data_provider.val_loader()

        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Validation step
                metrics = self.training_strategy.validate_step(
                    self.model, batch
                )

                losses.append(metrics.get("val_loss", 0))
                dice_scores.append(metrics.get("val_dice_score", 0))

                pbar.set_postfix({
                    "loss": f"{metrics['val_loss']:.4f}",
                    "dice": f"{metrics.get('val_dice_score', 0):.4f}",
                })

        return {"losses": losses, "dice_scores": dice_scores}

    def _move_batch_to_device(
        self,
        batch: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Move batch tensors to the training device."""
        return tuple(item.to(self.device) for item in batch)

    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict[str, List[float]],
        val_metrics: Dict[str, List[float]],
    ) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"

        # Get model state dict
        if hasattr(self.model, "model"):
            model_state = self.model.model.state_dict()
        else:
            model_state = self.model.state_dict()

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": sum(train_metrics["losses"]) / len(
                    train_metrics["losses"]
                ),
                "val_loss": sum(val_metrics["losses"]) / len(
                    val_metrics["losses"]
                ),
            },
            checkpoint_path,
        )

        print(f"Checkpoint saved to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return starting epoch."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        if hasattr(self.model, "model"):
            self.model.model.load_state_dict(checkpoint["model_state_dict"])
        else:
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
        metrics: Optional[Metrics] = None,
        device: Optional[torch.device] = None,
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
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        # Move model to device and set to eval mode
        self.model.model.to(self.device) if hasattr(self.model, "model") \
            else self.model.to(self.device)
        self.model.model.eval() if hasattr(self.model, "model") \
            else self.model.eval()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        checkpoint_path: Optional[str] = None,
        save_predictions: bool = False,
    ) -> Dict[str, float]:
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
                if isinstance(batch, (tuple, list)):
                    images = batch[0].to(self.device)
                    targets = batch[1].to(self.device) if len(batch) > 1 else None
                else:
                    images = batch.to(self.device)
                    targets = None

                # Run inference
                raw_predictions = self.inference_strategy.predict(
                    self.model, images
                )
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if hasattr(self.model, "model"):
            self.model.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Loaded checkpoint from {checkpoint_path}")

    def _save_predictions(
        self,
        predictions: Dict[str, Any],
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


class LightningTrainingPipeline:
    """
    Wrapper for PyTorch Lightning training.

    Provides compatibility with PyTorch Lightning's training loop
    while maintaining dependency injection architecture.
    """

    def __init__(
        self,
        model: ModelInterface,
        data_provider: DataProvider,
        training_strategy: TrainingStrategy,
        optimizer_factory: OptimizerFactory,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Lightning training pipeline.

        Args:
            model: Model to train
            data_provider: Data provider
            training_strategy: Training strategy
            optimizer_factory: Optimizer factory
            device: Device to use
        """
        self.model = model
        self.data_provider = data_provider
        self.training_strategy = training_strategy
        self.optimizer_factory = optimizer_factory

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        # Create optimizer
        self.optimizer = self.optimizer_factory.create(
            self.model.parameters()
        )

    def training_step(
        self,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Lightning-compatible training step."""
        batch = self._move_batch_to_device(batch)
        metrics = self.training_strategy.train_step(
            self.model, batch, self.optimizer
        )
        return {"loss": torch.tensor(metrics["loss"])}

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Lightning-compatible validation step."""
        batch = self._move_batch_to_device(batch)
        metrics = self.training_strategy.validate_step(self.model, batch)
        return {
            "val_loss": torch.tensor(metrics["val_loss"]),
            "val_dice": torch.tensor(metrics.get("val_dice_score", 0)),
        }

    def _move_batch_to_device(
        self,
        batch: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Move batch to device."""
        return tuple(item.to(self.device) for item in batch)

    def configure_optimizers(self):
        """Configure Lightning optimizers."""
        return self.optimizer
