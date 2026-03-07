"""
Application layer - Training and Test pipelines.

This module provides high-level pipelines that orchestrate
the training, validation, and testing workflows using Lightning Fabric.
"""

import inspect
import logging
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from tqdm import tqdm

from maskgit3d.domain.interfaces import (
    DataProvider,
    GANOptimizerFactory,
    InferenceStrategy,
    Metrics,
    ModelInterface,
    OptimizerFactory,
    TrainingStrategy,
)
from maskgit3d.infrastructure.checkpoints import load_checkpoint as load_ckpt

logger = logging.getLogger(__name__)

try:
    import lightning as L

    LIGHTNING_AVAILABLE = True
except ImportError:
    L = None  # type: ignore[misc,assignment]
    LIGHTNING_AVAILABLE = False


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

        num_samples = 0

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

                batch_size = processed["masks"].shape[0] if "masks" in processed else 1
                num_samples += batch_size

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

        return {"num_samples": num_samples}

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


class FabricTestPipeline:
    """
    Pipeline for inference and testing with Lightning Fabric.

    Provides lightweight distributed testing support using Lightning Fabric API
    while maintaining the dependency injection architecture.

    Supports Lightning Callbacks for extensibility (logging, visualization, etc.)
    """

    def __init__(
        self,
        model: ModelInterface,
        data_provider: DataProvider,
        inference_strategy: InferenceStrategy,
        metrics: Metrics | None = None,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        strategy: str = "auto",
        precision: (
            Literal[
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
            | None
        ) = "32-true",
        checkpoint_path: str | None = None,
        output_dir: str = "./outputs",
        callbacks: list[Any] | None = None,
    ):
        """
        Initialize Fabric testing pipeline.

        Args:
            model: Model to test (implements ModelInterface)
            data_provider: Data provider for test loader
            inference_strategy: Inference strategy with predict/post_process
            metrics: Metrics computation interface (optional)
            accelerator: Fabric accelerator ("cpu", "cuda", "auto")
            devices: Number of devices or device IDs
            strategy: Fabric strategy ("auto", "ddp", "fsdp")
            precision: Testing precision ("32-true", "16-mixed", "bf16-mixed")
            checkpoint_path: Path to checkpoint to load (optional)
            output_dir: Directory to save outputs
            callbacks: List of Lightning Callback instances
        """
        if not LIGHTNING_AVAILABLE:
            raise ImportError("Lightning is not installed. Install with: pip install lightning")

        self.model = model
        self.data_provider = data_provider
        self.inference_strategy = inference_strategy
        self.metrics = metrics
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.callbacks = callbacks or []

        self._accelerator = accelerator
        self._devices = devices
        self._strategy = strategy
        self._precision = precision

        self._fabric: Any = None
        self._global_step = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        save_predictions: bool = False,
        export_nifti: bool = False,
        enable_tensorboard: bool = False,
        tensorboard_dir: str | None = None,
    ) -> dict[str, float]:
        """
                Run the testing loop with Fabric.

                Args:
                    save_predictions: Whether to save prediction outputs
                    export_nifti: Whether to export predictions as NIfTI files
                    enable_tensorboard: Whether to enable TensorBoard logging
                    tensorboard_dir: Directory for TensorBoard logs (default: output_dir/tensorboard)

        Returns:
                    Dictionary of test metrics
        """
        assert L is not None, "Lightning is not installed"
        self._fabric = L.Fabric(
            accelerator=self._accelerator,
            devices=self._devices,
            strategy=self._strategy,
            precision=self._precision,  # type: ignore[arg-type]
            callbacks=self.callbacks,
        )
        self._fabric.launch()

        self.model = self._fabric.setup(self.model)

        if self.checkpoint_path:
            self._load_checkpoint(self.checkpoint_path)

        self.model.eval()

        if self.metrics:
            self.metrics.reset()

        test_loader = self._fabric.setup_dataloaders(self.data_provider.test_loader())

        writer = None
        if enable_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = Path(tensorboard_dir) if tensorboard_dir else self.output_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(tb_dir))

        slice_callbacks = [cb for cb in self.callbacks if hasattr(cb, "on_test_batch_end")]

        if writer is not None:
            for cb in slice_callbacks:
                if hasattr(cb, "set_writer"):
                    cb.set_writer(writer)

        num_samples = 0

        self._call_callbacks("on_test_epoch_start")

        pbar = tqdm(test_loader, desc="Testing")
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                self._call_callbacks("on_test_batch_start", batch, batch_idx)

                if isinstance(batch, tuple | list):
                    images = batch[0]
                    targets = batch[1] if len(batch) > 1 else None
                else:
                    images = batch
                    targets = None

                raw_predictions = self.inference_strategy.predict(self.model, images)
                processed = self.inference_strategy.post_process(raw_predictions)

                batch_size = processed["masks"].shape[0] if "masks" in processed else 1
                num_samples += batch_size

                if self.metrics and targets is not None:
                    self.metrics.update(processed, targets)

                if save_predictions:
                    self._save_predictions(processed, batch_idx)

                if export_nifti:
                    self._export_nifti(images, processed, targets, batch_idx)

                if writer is not None:
                    self._log_tensorboard(writer, images, processed, targets, batch_idx)

                self._global_step += 1
                pbar.set_postfix({"batch": batch_idx + 1})

                self._call_callbacks("on_test_batch_end", batch, batch_idx, processed)

        self._call_callbacks("on_test_epoch_end")

        if writer is not None:
            writer.close()

        if self.metrics:
            final_metrics = self.metrics.compute()
            if self._fabric.is_global_zero:
                print("Test Results:")
                for key, value in final_metrics.items():
                    print(f"  {key}: {value:.4f}")
            return final_metrics

        return {"num_samples": num_samples}

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint with Fabric.

        Supports both Fabric format (full model saved) and legacy state_dict format.
        """
        try:
            checkpoint = self._fabric.load(checkpoint_path)

            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    self.model.load_state_dict(checkpoint["model"])
                    print(f"Loaded checkpoint (Fabric format) from {checkpoint_path}")
                    return
                elif "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    print(f"Loaded checkpoint (legacy format) from {checkpoint_path}")
                    return
                elif "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                    print(f"Loaded checkpoint (legacy format) from {checkpoint_path}")
                    return
                else:
                    self.model.load_state_dict(checkpoint)
                    print(f"Loaded checkpoint (raw state_dict) from {checkpoint_path}")
                    return

            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Fabric checkpoint loading failed, trying legacy format: {e}")
            checkpoint = load_ckpt(checkpoint_path, map_location="cpu")

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)
            print(f"Loaded checkpoint (legacy format) from {checkpoint_path}")

    def _save_predictions(
        self,
        predictions: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Save predictions to output directory."""
        import numpy as np

        masks_path = self.output_dir / f"predictions_batch_{batch_idx}.npy"
        np.save(masks_path, predictions["masks"])

        probs_path = self.output_dir / f"probabilities_batch_{batch_idx}.npy"
        np.save(probs_path, predictions["probs"])

    def _export_nifti(
        self,
        images: "torch.Tensor",
        predictions: dict[str, Any],
        targets: "torch.Tensor | None",
        batch_idx: int,
    ) -> None:
        """Export inputs, predictions, and targets as NIfTI files.

        Files are saved to ``output_dir`` with names:
        - ``input_batch_{batch_idx}.nii.gz``        — raw input volume
        - ``predictions_batch_{batch_idx}.nii.gz``  — predicted mask / reconstruction
        - ``probabilities_batch_{batch_idx}.nii.gz``— predicted probabilities
        - ``target_batch_{batch_idx}.nii.gz``       — ground-truth label (if available)
        """
        try:
            import nibabel as nib
            import numpy as np

            def _to_numpy(t: "torch.Tensor") -> "np.ndarray":
                """Detach → CPU → float32 → numpy, drop batch dim."""
                arr = t.detach().cpu().float().numpy()
                # shape: (B, C, D, H, W) → take first sample → (C, D, H, W)
                if arr.ndim == 5:
                    arr = arr[0]
                # (C, D, H, W) → (D, H, W) for single-channel or keep all channels
                if arr.shape[0] == 1:
                    arr = arr[0]
                return arr

            nib.save(
                nib.Nifti1Image(_to_numpy(images), affine=np.eye(4)),
                str(self.output_dir / f"input_batch_{batch_idx}.nii.gz"),
            )

            masks = predictions.get("masks")
            probs = predictions.get("probs")

            if masks is not None:
                masks_arr = _to_numpy(masks) if hasattr(masks, "detach") else masks
                if isinstance(masks_arr, np.ndarray):
                    nib.save(
                        nib.Nifti1Image(masks_arr, affine=np.eye(4)),
                        str(self.output_dir / f"predictions_batch_{batch_idx}.nii.gz"),
                    )

            if probs is not None:
                probs_arr = _to_numpy(probs) if hasattr(probs, "detach") else probs
                if isinstance(probs_arr, np.ndarray):
                    nib.save(
                        nib.Nifti1Image(probs_arr, affine=np.eye(4)),
                        str(self.output_dir / f"probabilities_batch_{batch_idx}.nii.gz"),
                    )

            if targets is not None:
                nib.save(
                    nib.Nifti1Image(_to_numpy(targets), affine=np.eye(4)),
                    str(self.output_dir / f"target_batch_{batch_idx}.nii.gz"),
                )

        except ImportError:
            print("Warning: nibabel not installed. Skipping NIfTI export.")
        except Exception as e:
            print(f"Warning: Failed to export NIfTI for batch {batch_idx}: {e}")

    def _log_tensorboard(
        self,
        writer: Any,
        images: "torch.Tensor",
        predictions: dict[str, Any],
        targets: "torch.Tensor | None",
        batch_idx: int,
    ) -> None:
        """Log input, prediction, target centre slices and metrics to TensorBoard.

        For 3D volumes (B, C, D, H, W) the centre axial slice is extracted.
        Images are normalised to [0, 1] before logging.

        Tags written:
        - ``test/input``      — centre slice of the input volume
        - ``test/prediction`` — centre slice of the predicted mask / reconstruction
        - ``test/target``     — centre slice of the ground-truth (if available)
        - ``test/metrics/*``  — PSNR, SSIM, LPIPS metrics (if available)
        """
        import numpy as np

        step = self._global_step

        def _centre_slice(t: "torch.Tensor") -> "np.ndarray":
            """Return a normalised (H, W) centre-slice numpy array from a (B,C,D,H,W) tensor."""
            arr = t.detach().cpu().float()
            while arr.ndim > 3:
                arr = arr[0]
            if arr.ndim == 3:
                arr = arr[arr.shape[0] // 2]
            arr_np = arr.numpy()
            lo, hi = arr_np.min(), arr_np.max()
            if hi > lo:
                arr_np = (arr_np - lo) / (hi - lo)
            return arr_np

        writer.add_image("test/input", _centre_slice(images), step, dataformats="HW")

        masks = predictions.get("masks")
        if masks is not None and hasattr(masks, "detach"):
            writer.add_image("test/prediction", _centre_slice(masks), step, dataformats="HW")

        if targets is not None:
            writer.add_image("test/target", _centre_slice(targets), step, dataformats="HW")

        # Log PSNR/SSIM/LPIPS metrics to TensorBoard
        if self.metrics is not None:
            try:
                metrics = self.metrics.compute()
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"test/metrics/{key}", value, step)
            except Exception:
                # Silently skip if metrics computation fails
                pass

    def _call_callbacks(self, hook_name: str, *args, **kwargs) -> None:
        """Call callback hooks if they exist."""
        for callback in self.callbacks:
            if hasattr(callback, hook_name):
                method = getattr(callback, hook_name)
                if "fabric" in hook_name or "test" in hook_name:
                    method(self._fabric, *args, **kwargs)
                else:
                    method(self._fabric, self.model, *args, **kwargs)


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
        precision: (
            Literal[
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
            | None
        ) = "32-true",
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
        self._discriminator_optimizer: torch.optim.Optimizer | None = None
        self._discriminator: nn.Module | None = None
        self._current_epoch = 0
        self._global_step = 0
        self._is_gan_training = False

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
        logger.info("Creating Fabric instance...")
        assert L is not None, "Lightning is not installed"
        self._fabric = L.Fabric(
            accelerator=self._accelerator,
            devices=self._devices,
            strategy=self._strategy,
            precision=self._precision,  # type: ignore[arg-type]
            callbacks=self.callbacks,
        )
        logger.info("Launching Fabric...")
        self._fabric.launch()
        logger.info("Fabric launched successfully")

        logger.info("Creating optimizer...")

        # Check if using GAN-style training (VQGAN)
        if isinstance(self.optimizer_factory, GANOptimizerFactory):
            self._is_gan_training = True
            logger.info("Detected GAN training mode (VQGAN)")

            # Get discriminator from training strategy if available
            discriminator = getattr(self.training_strategy, "discriminator", None)

            if discriminator is not None:
                gen_params = self.model.parameters()
                disc_params = discriminator.parameters()
                opt_g, opt_d = cast(GANOptimizerFactory, self.optimizer_factory).create(
                    gen_params, disc_params
                )
                self._optimizer = opt_g
                self._discriminator_optimizer = opt_d
                self._discriminator = discriminator
                logger.info("Created separate optimizers for Generator and Discriminator")
            else:
                opt_g, _ = cast(GANOptimizerFactory, self.optimizer_factory).create(
                    self.model.parameters(), None
                )
                self._optimizer = opt_g
                logger.warning("GAN optimizer factory detected but no discriminator found")
        else:
            self._optimizer = self.optimizer_factory.create(self.model.parameters())

        logger.info("Setting up model and optimizer with Fabric...")

        if self._discriminator_optimizer is not None and self._discriminator is not None:
            self.model, self._discriminator, self._optimizer, self._discriminator_optimizer = (
                self._fabric.setup(
                    self.model, self._discriminator, self._optimizer, self._discriminator_optimizer
                )
            )
            disc_attr = getattr(self.training_strategy, "discriminator", None)
            if disc_attr is not None:
                object.__setattr__(self.training_strategy, "discriminator", self._discriminator)
        else:
            self.model, self._optimizer = self._fabric.setup(self.model, self._optimizer)

        logger.info("Model and optimizer setup complete")

        logger.info("Setting up dataloaders...")
        logger.info(f"Getting train loader from data_provider: {self.data_provider}")
        train_loader_orig = self.data_provider.train_loader()
        logger.info(f"Train loader created: {len(train_loader_orig)} batches")
        train_loader = self._fabric.setup_dataloaders(train_loader_orig)
        logger.info("Train loader setup complete")
        logger.info(f"Getting val loader from data_provider: {self.data_provider}")
        val_loader_orig = self.data_provider.val_loader()
        logger.info(f"Val loader created: {len(val_loader_orig)} batches")
        val_loader = self._fabric.setup_dataloaders(val_loader_orig)
        logger.info("Validation loader setup complete")

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

            optimizer = self._optimizer
            assert optimizer is not None

            disc_metrics: dict[str, float] | None = None
            if self._is_gan_training and self._discriminator_optimizer is not None:
                if hasattr(self.training_strategy, "train_discriminator_step"):
                    disc_step_method = getattr(self.training_strategy, "train_discriminator_step")
                    disc_metrics = disc_step_method(
                        self.model, batch, self._discriminator_optimizer
                    )

            train_step_params = inspect.signature(self.training_strategy.train_step).parameters
            if len(train_step_params) >= 3:
                step_output = self.training_strategy.train_step(self.model, batch, optimizer)
            else:
                step_output = self.training_strategy.train_step(self.model, batch)  # type: ignore[call-arg]

            if disc_metrics is not None and isinstance(step_output, dict):
                step_output.update(disc_metrics)

            if isinstance(step_output, dict):
                step_metrics = step_output
            elif isinstance(step_output, torch.Tensor):
                step_metrics = {"loss": float(step_output.detach().item())}
            elif isinstance(step_output, int | float):
                step_metrics = {"loss": float(step_output)}
            else:
                raise TypeError(
                    "Training strategy train_step must return a metrics dict or scalar loss"
                )

            if "loss" not in step_metrics:
                raise KeyError("Training strategy train_step must return metrics including 'loss'")

            self._global_step += 1

            loss_value = float(step_metrics["loss"])

            for key, value in step_metrics.items():
                if not isinstance(value, int | float):
                    continue

                metric_name = "train_loss" if key == "loss" else key
                if not metric_name.startswith("train_"):
                    metric_name = f"train_{metric_name}"

                metrics_history.setdefault(metric_name, []).append(float(value))

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

                outputs = {k: v for k, v in metrics.items() if not isinstance(v, int | float)}
                self._call_callbacks("on_validation_batch_end", batch, batch_idx, outputs)

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

        if self._discriminator_optimizer is not None:
            checkpoint_data["discriminator_optimizer"] = self._discriminator_optimizer

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

        if "optimizer" in checkpoint and self._optimizer is not None:
            self._optimizer.load_state_dict(checkpoint["optimizer"].state_dict())

        if "discriminator_optimizer" in checkpoint and self._discriminator_optimizer is not None:
            self._discriminator_optimizer.load_state_dict(
                checkpoint["discriminator_optimizer"].state_dict()
            )

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
