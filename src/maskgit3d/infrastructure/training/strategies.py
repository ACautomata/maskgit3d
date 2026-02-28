"""
Training strategy implementations for VQGAN and MaskGIT models.

This module provides concrete implementations of TrainingStrategy
including loss functions, optimization, and metrics computation.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast
import torch
import torch.nn as nn
import torch.nn.functional as F

from maskgit3d.domain.interfaces import (
    GANOptimizerFactory,
    InferenceStrategy,
    MaskGITModelInterface,
    Metrics,
    ModelInterface,
    OptimizerFactory,
    TrainingStrategy,
    VQModelInterface,
)


# =============================================================================
# Mixed Precision Training
# =============================================================================


class MixedPrecisionTrainer:
    """
    Mixed precision training helper for FP16/BF16.

    Provides automatic mixed precision (AMP) support for training
    to reduce memory usage and improve throughput.
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: str = "float16",
        grad_clip: Optional[float] = None,
    ):
        """
        Initialize mixed precision trainer.

        Args:
            enabled: Whether to enable mixed precision
            dtype: Precision type ("float16" or "bfloat16")
            grad_clip: Maximum gradient norm for clipping
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype
        self.grad_clip = grad_clip

        if self.enabled:
            if dtype == "float16":
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.scaler = None  # BF16 doesn't need scaler

    def autocast_context(self):
        """Get autocast context for forward pass."""
        if not self.enabled:
            return torch.amp.autocast(device_type="cuda", dtype=torch.float32, enabled=False)

        dtype = torch.float16 if self.dtype == "float16" else torch.bfloat16
        return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for backward pass.

        Args:
            loss: Computed loss tensor

        Returns:
            Scaled loss (if FP16), original loss otherwise
        """
        if not self.enabled or self.dtype == "bfloat16":
            return loss
        return self.scaler.scale(loss)

    def step_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
    ) -> None:
        """
        Step optimizer with gradient scaling.

        Args:
            optimizer: Optimizer to step
            loss: Loss tensor for backward
        """
        if not self.enabled:
            optimizer.step()
            optimizer.zero_grad()
            return

        if self.dtype == "float16":
            # Unscale gradients
            self.scaler.unscale_(optimizer)

            # Clip gradients
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], self.grad_clip)

            # Step with scaler
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # BF16 - just clip and step
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], self.grad_clip)
            optimizer.step()

        optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        state = {"enabled": self.enabled, "dtype": self.dtype}
        if self.enabled and self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.enabled = state.get("enabled", self.enabled)
        self.dtype = state.get("dtype", self.dtype)
        if self.enabled and self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])


# =============================================================================
# Optimizer Factories
# =============================================================================


class AdamOptimizerFactory(OptimizerFactory):
    """Factory for Adam optimizer."""

    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        betas: Tuple[float, float] = (0.9, 0.999),
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas

    def create(
        self,
        model_params: Iterator[torch.Tensor],
    ) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            model_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )


class SGDOptimizerFactory(OptimizerFactory):
    """Factory for SGD optimizer with momentum."""

    def __init__(
        self,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        nesterov: bool = True,
    ):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def create(
        self,
        model_params: Iterator[torch.Tensor],
    ) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            model_params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )


class AdamWOptimizerFactory(OptimizerFactory):
    """Factory for AdamW optimizer."""

    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas

    def create(
        self,
        model_params: Iterator[torch.Tensor],
    ) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )


class VQGANOptimizerFactory(GANOptimizerFactory):
    """Factory for VQGAN optimizer with separate G/D optimizers."""

    def __init__(
        self,
        lr_g: float = 1e-4,
        lr_d: float = 2e-4,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
    ):
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.weight_decay = weight_decay
        self.betas = betas

    def create(
        self,
        gen_params: Iterator[torch.Tensor],
        disc_params: Optional[Iterator[torch.Tensor]] = None,
    ) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.Optimizer]]:
        opt_g = torch.optim.AdamW(
            gen_params,
            lr=self.lr_g,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        if disc_params is not None:
            opt_d = torch.optim.AdamW(
                disc_params,
                lr=self.lr_d,
                weight_decay=self.weight_decay,
                betas=self.betas,
            )
        else:
            opt_d = None
        return opt_g, opt_d


# =============================================================================
# VQGAN Training Strategies
# =============================================================================


class VQGANTrainingStrategy(TrainingStrategy):
    """
    Training strategy for VQGAN models.

    Supports VQ-VAE training with codebook commitment loss
    and mixed precision training.
    """

    def __init__(
        self,
        codebook_weight: float = 1.0,
        pixel_loss_weight: float = 1.0,
        mixed_precision: bool = False,
        amp_dtype: str = "float16",
        grad_clip: Optional[float] = None,
    ):
        """
        Initialize VQGAN training strategy.

        Args:
            codebook_weight: Weight for codebook commitment loss
            pixel_loss_weight: Weight for pixel reconstruction loss
            mixed_precision: Enable automatic mixed precision training
            amp_dtype: Mixed precision dtype ("float16" or "bfloat16")
            grad_clip: Maximum gradient norm for clipping
        """
        self.codebook_weight = codebook_weight
        self.pixel_loss_weight = pixel_loss_weight

        # Mixed precision
        self.mixed_precision = MixedPrecisionTrainer(
            enabled=mixed_precision,
            dtype=amp_dtype,
            grad_clip=grad_clip,
        )

    def train_step(
        self,
        model: VQModelInterface,
        batch: Tuple[torch.Tensor, ...],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Execute one training step.

        Args:
            model: VQModel instance
            batch: Tuple of (input_images,)
            optimizer: Optimizer for parameter updates

        Returns:
            Dictionary of training metrics
        """
        # Unpack batch
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        model.train()
        vq_model = cast(VQModelInterface, model)

        # Forward pass with mixed precision
        with self.mixed_precision.autocast_context():
            xrec, qloss = vq_model.forward_with_loss(x)
            rec_loss = torch.abs(x - xrec)
            loss = self.pixel_loss_weight * rec_loss.mean() + self.codebook_weight * qloss.mean()

        # Backward
        scaled_loss = self.mixed_precision.scale_loss(loss)
        scaled_loss.backward()
        self.mixed_precision.step_optimizer(optimizer, loss)

        return {
            "loss": loss.item(),
            "rec_loss": rec_loss.mean().item(),
            "codebook_loss": qloss.mean().item(),
        }

    def validate_step(
        self,
        model: ModelInterface,
        batch: Tuple[torch.Tensor, ...],
    ) -> Dict[str, float]:
        """
        Execute validation step.

        Args:
            model: VQModel instance
            batch: Tuple of (input_images,)

        Returns:
            Dictionary of validation metrics
        """
        model.eval()
        vq_model = cast(VQModelInterface, model)

        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        with torch.no_grad():
            xrec, qloss = vq_model.forward_with_loss(x)
            rec_loss = torch.abs(x - xrec)

        return {
            "val_loss": rec_loss.mean().item(),
            "val_rec_loss": rec_loss.mean().item(),
            "val_codebook_loss": qloss.mean().item(),
        }


# =============================================================================
# VQGAN Inference Strategies
# =============================================================================


class VQGANInference(InferenceStrategy):
    """
    Inference strategy for VQGAN models.

    Supports:
    - Reconstruction: Encode-decode through the model
    - Generation: Decode from random latent codes
    - Codebook lookup: Decode from specific codebook indices
    """

    def __init__(
        self,
        mode: str = "reconstruct",
        num_samples: int = 1,
        temperature: float = 1.0,
    ):
        """
        Initialize VQGAN inference strategy.

        Args:
            mode: Inference mode ("reconstruct", "generate", or "decode_code")
            num_samples: Number of samples to generate (for "generate" mode)
            temperature: Sampling temperature for codebook selection
        """
        self.mode = mode
        self.num_samples = num_samples
        self.temperature = temperature

    def predict(
        self,
        model: ModelInterface,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run inference.

        Args:
            model: VQModel instance
            batch: Input batch

        Returns:
            Model outputs (reconstructed or generated images)
        """
        model.eval()

        if self.mode == "reconstruct":
            with torch.no_grad():
                return model(batch)[0]

        elif self.mode == "generate":
            with torch.no_grad():
                # Generate random codes
                # latent_shape is (C, D, H, W) for 3D
                latent_shape = model.latent_shape
                dd, hh, ww = latent_shape[1], latent_shape[2], latent_shape[3]
                codes = torch.randint(
                    0,
                    model.codebook_size,
                    (batch.shape[0], dd, hh, ww),
                    device=batch.device,
                )
                return model.decode_code(codes)

        elif self.mode == "decode_code":
            with torch.no_grad():
                # batch contains codebook indices
                return model.decode_code(batch)

        raise ValueError(f"Unknown mode: {self.mode}")

    def post_process(
        self,
        predictions: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Post-process predictions.

        Args:
            predictions: Raw model outputs

        Returns:
            Processed predictions dictionary
        """
        # Convert to numpy and normalize to [0, 1]
        images = predictions.cpu().float()
        images = (images + 1) / 2  # From [-1, 1] to [0, 1]
        images = images.clamp(0, 1)

        return {
            "images": images.numpy(),
        }


class VQGANMetrics(Metrics):
    """
    Metrics for VQGAN evaluation.

    Computes reconstruction metrics including PSNR, SSIM using MONAI/torchmetrics.
    """

    def __init__(
        self,
        data_range: float = 1.0,
        spatial_dims: int = 2,
    ):
        """
        Initialize VQGAN metrics.

        Args:
            data_range: Data range of input images (e.g., 1.0 for [-1,1], 255 for [0,255])
            spatial_dims: Number of spatial dimensions (2 for images, 3 for volumetric)
        """
        self.data_range = data_range
        self.spatial_dims = spatial_dims

        # Use MONAI for PSNR and SSIM
        try:
            from monai.metrics import PSNRMetric, SSIMMetric

            self.psnr_metric = PSNRMetric(
                max_val=data_range,
                reduction="mean",
            )
            self.ssim_metric = SSIMMetric(
                max_val=data_range,
                spatial_dims=spatial_dims,
                reduction="mean",
            )
        except ImportError:
            # Fallback to manual implementation
            self.psnr_metric = None
            self.ssim_metric = None

        self.reset()

    def reset(self) -> None:
        """Reset metrics."""
        if self.psnr_metric:
            self.psnr_metric.reset()
        if self.ssim_metric:
            self.ssim_metric.reset()
        self.psnr_values = []
        self.ssim_values = []

    def update(
        self,
        predictions: Any,
        targets: Any,
    ) -> None:
        """
        Update metrics with new predictions.

        Args:
            predictions: Reconstructed images [B, C, H, W] or [B, C, D, H, W]
            targets: Original images
        """
        # Convert to tensor if needed
        if not isinstance(predictions, torch.Tensor):
            pred = torch.from_numpy(predictions["images"])
        else:
            pred = predictions

        if not isinstance(targets, torch.Tensor):
            target = torch.from_numpy(targets)
        else:
            target = targets

        # Ensure predictions are in [0, data_range] range for MONAI
        # VQGAN outputs are typically in [-1, 1], so rescale to [0, 1]
        pred = (pred + 1) / 2 * self.data_range
        target = (target + 1) / 2 * self.data_range

        if self.psnr_metric and self.ssim_metric:
            # Use MONAI metrics
            self.psnr_metric(pred, target)
            self.ssim_metric(pred, target)
        else:
            # Fallback to manual computation
            mse = F.mse_loss(pred, target)
            psnr = 20 * torch.log10(self.data_range / torch.sqrt(mse))
            self.psnr_values.append(psnr.item())

            # Simplified SSIM
            ssim_val = self._compute_ssim_fallback(pred, target)
            self.ssim_values.append(ssim_val.item())

    def _compute_ssim_fallback(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Fallback SSIM computation using torchmetrics if available."""
        try:
            from torchmetrics.functional import structural_similarity_index_measure

            return structural_similarity_index_measure(
                img1.unsqueeze(0) if img1.dim() == 3 else img1,
                img2.unsqueeze(0) if img2.dim() == 3 else img2,
                data_range=self.data_range,
            )
        except ImportError:
            # Ultimate fallback: simple correlation-based metric
            return torch.tensor(1.0 - F.mse_loss(img1, img2) / (self.data_range**2))

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        if self.psnr_metric and self.ssim_metric:
            psnr = self.psnr_metric.aggregate().item()
            ssim = self.ssim_metric.aggregate().item()
        else:
            psnr = sum(self.psnr_values) / len(self.psnr_values) if self.psnr_values else 0.0
            ssim = sum(self.ssim_values) / len(self.ssim_values) if self.ssim_values else 0.0

        return {
            "psnr": psnr,
            "ssim": ssim,
        }


# =============================================================================
# MaskGIT Training Strategies
# =============================================================================


class MaskGITTrainingStrategy(TrainingStrategy):
    """
    Training strategy for MaskGIT models.

    Uses BERT-style masked token prediction where a portion of
    tokens are randomly masked and the Transformer learns to predict them.
    Supports mixed precision training.
    """

    def __init__(
        self,
        mask_ratio: float = 0.5,
        reconstruction_weight: float = 1.0,
        mixed_precision: bool = False,
        amp_dtype: str = "float16",
        grad_clip: Optional[float] = None,
    ):
        """
        Initialize MaskGIT training strategy.

        Args:
            mask_ratio: Ratio of tokens to mask during training
            reconstruction_weight: Weight for reconstruction loss
            mixed_precision: Enable automatic mixed precision training
            amp_dtype: Mixed precision dtype ("float16" or "bfloat16")
            grad_clip: Maximum gradient norm for clipping
        """
        self.mask_ratio = mask_ratio
        self.reconstruction_weight = reconstruction_weight

        # Mixed precision
        self.mixed_precision = MixedPrecisionTrainer(
            enabled=mixed_precision,
            dtype=amp_dtype,
            grad_clip=grad_clip,
        )

    def train_step(
        self,
        model: MaskGITModelInterface,
        batch: Tuple[torch.Tensor, ...],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Execute one training step with masked token prediction.

        Args:
            model: MaskGITModel instance
            batch: Tuple of (input_images,)
            optimizer: Optimizer for parameter updates

        Returns:
            Dictionary of training metrics
        """
        # Unpack batch
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        model.train()

        # Forward pass with masking and mixed precision
        with self.mixed_precision.autocast_context():
            if hasattr(model, "compute_maskgit_loss"):
                loss, metrics = model.compute_maskgit_loss(x, mask_ratio=self.mask_ratio)
            else:
                # Manual implementation for compatibility
                tokens = model.encode_tokens(x)
                B, D, H, W = tokens.shape
                tokens_flat = tokens.view(B, -1)

                # Random masking
                mask = torch.rand(B, D * H * W, device=tokens.device) < self.mask_ratio

                # Ensure at least one token masked per sample
                for i in range(B):
                    if not mask[i].any():
                        mask[i, torch.randint(0, D * H * W, (1,), device=tokens.device)] = True

                # Get predictions
                logits = model.transformer.forward(tokens_flat, mask_indices=mask)

                # Compute loss only on masked positions
                masked_logits = logits[mask]
                masked_targets = tokens_flat[mask]

                loss = F.cross_entropy(masked_logits, masked_targets)

                metrics = {
                    "loss": loss.item(),
                    "mask_acc": (masked_logits.argmax(dim=-1) == masked_targets)
                    .float()
                    .mean()
                    .item(),
                    "mask_ratio": mask.float().mean().item(),
                }

        # Backward with mixed precision on live loss tensor
        scaled_loss = self.mixed_precision.scale_loss(loss)
        scaled_loss.backward()
        self.mixed_precision.step_optimizer(optimizer, loss)

        return metrics

    def validate_step(
        self,
        model: ModelInterface,
        batch: Tuple[torch.Tensor, ...],
    ) -> Dict[str, float]:
        """
        Execute validation step.

        Args:
            model: MaskGITModel instance
            batch: Tuple of (input_images,)

        Returns:
            Dictionary of validation metrics
        """
        model.eval()

        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        with torch.no_grad():
            # Reconstruction
            x_rec = model(x)

            # Compute reconstruction loss
            rec_loss = torch.abs(x - x_rec).mean()

            # Encode and compute token accuracy
            tokens = model.encode_tokens(x)
            tokens_pred = model.transformer.encode(tokens, return_logits=True).argmax(dim=-1)
            token_acc = (tokens_pred == tokens.view(tokens.shape[0], -1)).float().mean()

        return {
            "val_loss": rec_loss.item(),
            "val_token_acc": token_acc.item(),
        }


class MaskGITInference(InferenceStrategy):
    """
    Inference strategy for MaskGIT models.

    Supports:
    - Reconstruction: Encode-decode through the model
    - Generation: Generate from random tokens using iterative decoding
    """

    def __init__(
        self,
        mode: str = "generate",
        num_iterations: int = 12,
        temperature: float = 1.0,
    ):
        """
        Initialize MaskGIT inference strategy.

        Args:
            mode: Inference mode ("reconstruct" or "generate")
            num_iterations: Number of iterations for generation
            temperature: Sampling temperature
        """
        self.mode = mode
        self.num_iterations = num_iterations
        self.temperature = temperature

    def predict(
        self,
        model: ModelInterface,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run inference.

        Args:
            model: MaskGITModel instance
            batch: Input batch

        Returns:
            Model outputs (reconstructed or generated volumes)
        """
        model.eval()

        if self.mode == "reconstruct":
            with torch.no_grad():
                return model(batch)

        elif self.mode == "generate":
            with torch.no_grad():
                # Get latent shape
                latent_shape = model.latent_shape
                B = batch.shape[0]

                # Generate
                return model.generate(
                    shape=(B,) + latent_shape[1:],
                    temperature=self.temperature,
                    num_iterations=self.num_iterations,
                )

        raise ValueError(f"Unknown mode: {self.mode}")

    def post_process(
        self,
        predictions: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Post-process predictions.

        Args:
            predictions: Raw model outputs

        Returns:
            Processed predictions dictionary
        """
        # Convert to numpy and normalize to [0, 1]
        volumes = predictions.cpu().float()
        volumes = (volumes + 1) / 2  # From [-1, 1] to [0, 1]
        volumes = volumes.clamp(0, 1)

        return {
            "volumes": volumes.numpy(),
        }
