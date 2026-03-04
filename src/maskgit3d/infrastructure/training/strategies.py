"""
Training strategy implementations for VQGAN and MaskGIT models.

This module provides concrete implementations of TrainingStrategy
including loss functions, optimization, and metrics computation.
"""

from collections.abc import Iterator
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from maskgit3d.domain.interfaces import (
    DiscriminatorInterface,
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
        grad_clip: float | None = None,
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
            return torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False)

        dtype = torch.float16 if self.dtype == "float16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for backward pass.

        Args:
            loss: Computed loss tensor

        Returns:
            Scaled loss (if FP16), original loss otherwise
        """
        if not self.enabled or self.dtype == "bfloat16" or self.scaler is None:
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

        if self.dtype == "float16" and self.scaler is not None:
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

    def state_dict(self) -> dict[str, Any]:
        """Get state dict for checkpointing."""
        state = {"enabled": self.enabled, "dtype": self.dtype}
        if self.enabled and self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
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
        betas: tuple[float, float] = (0.9, 0.999),
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
        betas: tuple[float, float] = (0.9, 0.95),
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
    """Factory for VQGAN optimizer with separate G/D optimizers using same learning rate."""

    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.5, 0.9),
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas

    def create(
        self,
        gen_params: Iterator[torch.Tensor],
        disc_params: Iterator[torch.Tensor] | None = None,
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer | None]:
        opt_g = torch.optim.AdamW(
            gen_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        if disc_params is not None:
            opt_d = torch.optim.AdamW(
                disc_params,
                lr=self.lr,  # Same learning rate for discriminator
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

    Supports VQ-VAE training with:
    - L1 reconstruction loss
    - LPIPS perceptual loss
    - Adversarial loss with discriminator
    - Codebook commitment loss

    All loss weights are configurable via config.
    """

    def __init__(
        self,
        codebook_weight: float = 1.0,
        pixel_loss_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_weight: float = 0.1,
        disc_start: int = 10000,
        discriminator: "DiscriminatorInterface | None" = None,
        mixed_precision: bool = False,
        amp_dtype: str = "float16",
        grad_clip: float | None = None,
    ):
        """
        Initialize VQGAN training strategy.

        Args:
            codebook_weight: Weight for codebook commitment loss
            pixel_loss_weight: Weight for L1 pixel reconstruction loss
            perceptual_weight: Weight for LPIPS perceptual loss
            disc_weight: Weight for adversarial discriminator loss
            disc_start: Number of steps before starting discriminator training
            discriminator: Discriminator model for adversarial training (optional)
            mixed_precision: Enable automatic mixed precision training
            amp_dtype: Mixed precision dtype ("float16" or "bfloat16")
            grad_clip: Maximum gradient norm for clipping
        """
        self.codebook_weight = codebook_weight
        self.pixel_loss_weight = pixel_loss_weight
        self.perceptual_weight = perceptual_weight
        self.disc_weight = disc_weight
        self.disc_start = disc_start
        self.discriminator = discriminator
        self.global_step = 0

        # Initialize LPIPS loss (only when perceptual_weight > 0 to avoid
        # downloading VGG weights unnecessarily)
        self.lpips_fn = None
        if perceptual_weight > 0:
            try:
                import lpips

                self.lpips_fn = lpips.LPIPS(net="vgg", verbose=False)
                # Freeze LPIPS parameters
                for param in self.lpips_fn.parameters():
                    param.requires_grad = False
            except ImportError:
                import warnings

                warnings.warn(
                    "lpips not installed. Perceptual loss will be disabled. "
                    "Install with: pip install lpips>=0.1.4"
                )

        # Mixed precision
        self.mixed_precision = MixedPrecisionTrainer(
            enabled=mixed_precision,
            dtype=amp_dtype,
            grad_clip=grad_clip,
        )

    def _compute_perceptual_loss(self, x: torch.Tensor, xrec: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS perceptual loss between real and reconstructed images.

        LPIPS expects RGB images (3 channels) with minimum spatial dimensions.
        For medical images with single channel, we repeat to 3 channels.
        For small spatial dimensions, we skip LPIPS (returns 0).
        """
        if self.lpips_fn is None or self.perceptual_weight == 0:
            return torch.tensor(0.0, device=x.device)

        # LPIPS expects 2D images [B, C, H, W], need to handle 3D volumes
        # For 3D volumes, we compute perceptual loss slice-wise
        if x.dim() == 5:  # [B, C, D, H, W]
            # Take middle slices along depth dimension
            mid_slice = x.shape[2] // 2
            x_2d = x[:, :, mid_slice, :, :]  # [B, C, H, W]
            xrec_2d = xrec[:, :, mid_slice, :, :]

            # Check minimum spatial dimensions (LPIPS VGG needs at least 32x32)
            if x_2d.shape[2] < 32 or x_2d.shape[3] < 32:
                return torch.tensor(0.0, device=x.device)

            # LPIPS expects RGB images (3 channels)
            # For single-channel medical images, repeat to 3 channels
            if x_2d.shape[1] == 1:
                x_2d = x_2d.repeat(1, 3, 1, 1)  # [B, 3, H, W]
                xrec_2d = xrec_2d.repeat(1, 3, 1, 1)

            # LPIPS expects input in range [-1, 1], normalize from [0, 1]
            x_norm = 2.0 * x_2d - 1.0
            xrec_norm = 2.0 * xrec_2d - 1.0

            return self.lpips_fn(x_norm, xrec_norm).mean()
        else:
            # 2D case
            # Check minimum spatial dimensions
            if x.shape[2] < 32 or x.shape[3] < 32:
                return torch.tensor(0.0, device=x.device)

            # Handle single channel
            x_2d = x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)
            xrec_2d = xrec if xrec.shape[1] == 3 else xrec.repeat(1, 3, 1, 1)

            x_norm = 2.0 * x_2d - 1.0
            xrec_norm = 2.0 * xrec_2d - 1.0
            return self.lpips_fn(x_norm, xrec_norm).mean()

    def _compute_adversarial_loss(
        self,
        x: torch.Tensor,
        xrec: torch.Tensor,
        optimizer_idx: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute adversarial loss for generator or discriminator.

        Args:
            x: Real images
            xrec: Reconstructed images
            optimizer_idx: 0 for generator, 1 for discriminator

        Returns:
            Tuple of (loss, metrics_dict)
        """
        if self.discriminator is None or self.disc_weight == 0:
            return torch.tensor(0.0, device=x.device), {}

        metrics = {}

        if optimizer_idx == 0:  # Generator update
            # Try to fool discriminator with reconstructed images
            fake_logits = self.discriminator.forward(xrec)
            g_loss = -torch.mean(fake_logits)
            metrics["g_loss"] = g_loss.item()
            return g_loss, metrics

        else:  # Discriminator update
            # Distinguish real from fake
            real_logits = self.discriminator.forward(x.detach())
            fake_logits = self.discriminator.forward(xrec.detach())

            # Hinge loss
            d_loss_real = torch.mean(F.relu(1.0 - real_logits))
            d_loss_fake = torch.mean(F.relu(1.0 + fake_logits))
            d_loss = d_loss_real + d_loss_fake

            metrics["d_loss"] = d_loss.item()
            metrics["d_loss_real"] = d_loss_real.item()
            metrics["d_loss_fake"] = d_loss_fake.item()

            return d_loss, metrics

    def train_step(  # type: ignore[override]
        self,
        model: VQModelInterface,
        batch: tuple[torch.Tensor, ...],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
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
        x = batch[0] if isinstance(batch, tuple | list) else batch

        model.train()
        vq_model = cast(VQModelInterface, model)

        # Forward pass with mixed precision
        with self.mixed_precision.autocast_context():
            xrec, qloss = vq_model.forward_with_loss(x)

            # 1. L1 reconstruction loss
            rec_loss = torch.abs(x - xrec).mean()

            # 2. LPIPS perceptual loss
            perceptual_loss = self._compute_perceptual_loss(x, xrec)

            # 3. Adversarial loss (only after disc_start)
            if (
                self.discriminator is not None
                and self.disc_weight > 0
                and self.global_step >= self.disc_start
            ):
                g_loss, g_metrics = self._compute_adversarial_loss(x, xrec, optimizer_idx=0)
            else:
                g_loss = torch.tensor(0.0, device=x.device)
                g_metrics = {}

            # Total generator loss
            loss = (
                self.pixel_loss_weight * rec_loss
                + self.perceptual_weight * perceptual_loss
                + self.disc_weight * g_loss
                + self.codebook_weight * qloss.mean()
            )

        # Backward
        scaled_loss = self.mixed_precision.scale_loss(loss)
        scaled_loss.backward()
        self.mixed_precision.step_optimizer(optimizer, loss)

        self.global_step += 1

        metrics = {
            "loss": loss.item(),
            "rec_loss": rec_loss.item(),
            "perceptual_loss": perceptual_loss.item(),
            "codebook_loss": qloss.mean().item(),
            "g_loss": g_loss.item(),
            **g_metrics,
        }

        return metrics

    def train_discriminator_step(
        self,
        model: VQModelInterface,
        batch: tuple[torch.Tensor, ...],
        discriminator_optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """
        Execute discriminator training step.

        This should be called alternately with train_step for GAN training.

        Args:
            model: VQModel instance
            batch: Tuple of (input_images,)
            discriminator_optimizer: Optimizer for discriminator

        Returns:
            Dictionary of discriminator metrics
        """
        if self.discriminator is None or self.global_step < self.disc_start:
            return {}

        # Unpack batch
        x = batch[0] if isinstance(batch, tuple | list) else batch

        model.eval()
        vq_model = cast(VQModelInterface, model)
        if self.discriminator is not None:
            cast(nn.Module, self.discriminator).train()

        # Forward pass with mixed precision
        with self.mixed_precision.autocast_context():
            with torch.no_grad():
                xrec, _ = vq_model.forward_with_loss(x)

            # Discriminator loss
            d_loss, d_metrics = self._compute_adversarial_loss(x, xrec, optimizer_idx=1)

        # Backward
        scaled_loss = self.mixed_precision.scale_loss(d_loss)
        scaled_loss.backward()
        self.mixed_precision.step_optimizer(discriminator_optimizer, d_loss)

        return d_metrics

    def validate_step(
        self,
        model: ModelInterface,
        batch: tuple[torch.Tensor, ...],
    ) -> dict[str, float]:
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

        x = batch[0] if isinstance(batch, tuple | list) else batch

        with torch.no_grad():
            xrec, qloss = vq_model.forward_with_loss(x)
            rec_loss = torch.abs(x - xrec).mean()
            perceptual_loss = self._compute_perceptual_loss(x, xrec)

        return {
            "val_loss": rec_loss.item(),
            "val_rec_loss": rec_loss.item(),
            "val_perceptual_loss": perceptual_loss.item(),
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
        vq_model = cast(VQModelInterface, model)

        if self.mode == "reconstruct":
            with torch.no_grad():
                quantized, _, _ = vq_model.encode(x=batch)
                return vq_model.decode(quantized)

        elif self.mode == "generate":
            with torch.no_grad():
                # Generate random codes
                # latent_shape is (C, D, H, W) for 3D
                latent_shape = vq_model.latent_shape
                dd, hh, ww = latent_shape[1], latent_shape[2], latent_shape[3]
                codes = torch.randint(
                    0,
                    vq_model.codebook_size,
                    (batch.shape[0], dd, hh, ww),
                    device=batch.device,
                )
                return vq_model.decode_code(codes)

        elif self.mode == "decode_code":
            with torch.no_grad():
                # batch contains codebook indices
                return vq_model.decode_code(batch)

        raise ValueError(f"Unknown mode: {self.mode}")

    def post_process(
        self,
        predictions: torch.Tensor,
    ) -> dict[str, Any]:
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

    Computes reconstruction metrics including PSNR, SSIM, LPIPS using MONAI.
    Tracks per-batch values and computes mean±std statistics.
    Supports TensorBoard logging and JSON/CSV export.
    """

    def __init__(
        self,
        data_range: float = 1.0,
        spatial_dims: int = 2,
        enable_lpips: bool = True,
    ):
        """
        Initialize VQGAN metrics.

        Args:
            data_range: Data range of input images (e.g., 1.0 for [-1,1], 255 for [0,255])
            spatial_dims: Number of spatial dimensions (2 for images, 3 for volumetric)
            enable_lpips: Whether to enable LPIPS perceptual metric
        """
        self.data_range = data_range
        self.spatial_dims = spatial_dims
        self.enable_lpips = enable_lpips

        # Use MONAI for PSNR and SSIM
        try:
            from monai.metrics.regression import PSNRMetric, SSIMMetric

            self.psnr_metric: PSNRMetric | None = PSNRMetric(
                max_val=data_range,
                reduction="mean",
            )
            self.ssim_metric: SSIMMetric | None = SSIMMetric(
                spatial_dims=spatial_dims,
                data_range=data_range,
                reduction="mean",
            )
        except ImportError:
            # Fallback to manual implementation
            self.psnr_metric = None
            self.ssim_metric = None

        self.lpips_loss: Any = None
        if enable_lpips:
            try:
                from monai.losses.perceptual import PerceptualLoss

                self.lpips_loss = PerceptualLoss(
                    spatial_dims=spatial_dims,
                    network_type="alex",
                    is_fake_3d=True,
                    fake_3d_ratio=0.5,
                )
                for param in self.lpips_loss.parameters():
                    param.requires_grad = False
            except ImportError:
                import warnings

                warnings.warn("MONAI PerceptualLoss not available. Upgrade MONAI to >=1.3")
                self.enable_lpips = False

        self.reset()

    def reset(self) -> None:
        """Reset metrics."""
        if self.psnr_metric is not None:
            self.psnr_metric.reset()
        if self.ssim_metric is not None:
            self.ssim_metric.reset()

        # Per-batch values for mean±std computation
        self.psnr_values: list[Any] = []
        self.ssim_values: list[Any] = []
        self.lpips_values: list[Any] = []

        # Accumulated values for MONAI metrics
        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._lpips_sum = 0.0
        self._count = 0

    def update(
        self,
        predictions: Any,
        targets: Any,
    ) -> None:
        if not isinstance(predictions, torch.Tensor):
            pred = torch.from_numpy(predictions["images"])
        else:
            pred = predictions

        target = torch.from_numpy(targets) if not isinstance(targets, torch.Tensor) else targets

        if target.is_cuda and not pred.is_cuda:
            pred = pred.to(target.device)

        pred_normalized = (pred + 1) / 2 * self.data_range
        target_normalized = (target + 1) / 2 * self.data_range

        if self.psnr_metric is not None:
            self.psnr_metric(pred_normalized, target_normalized)
            psnr_val = self._safe_item(self.psnr_metric.aggregate())
            self.psnr_values.append(psnr_val)
            self._psnr_sum += psnr_val

        if self.ssim_metric is not None:
            try:
                self.ssim_metric(pred_normalized, target_normalized)
                ssim_val = self._safe_item(self.ssim_metric.aggregate())
                self.ssim_values.append(ssim_val)
                self._ssim_sum += ssim_val
            except RuntimeError:
                # SSIM kernel size (default 11) may exceed small input dimensions
                pass

        if self.enable_lpips and self.lpips_loss is not None:
            try:
                lpips_val = self._compute_lpips(pred_normalized, target_normalized)
                self.lpips_values.append(lpips_val)
                self._lpips_sum += lpips_val
            except Exception:
                pass

        self._count += 1

    def _safe_item(self, tensor_or_tuple: torch.Tensor | tuple) -> Any:
        if isinstance(tensor_or_tuple, tuple):
            return tensor_or_tuple[0].mean().item()
        if tensor_or_tuple.numel() == 1:
            return tensor_or_tuple.item()
        return tensor_or_tuple.mean().item()

    def _compute_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return self.lpips_loss(pred, target).mean().item()

    def _compute_ssim_fallback(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Fallback SSIM computation using torchmetrics if available."""
        try:
            from torchmetrics.functional import structural_similarity_index_measure

            result = structural_similarity_index_measure(
                img1.unsqueeze(0) if img1.dim() == 3 else img1,
                img2.unsqueeze(0) if img2.dim() == 3 else img2,
                data_range=self.data_range,
            )
            # Handle both single tensor and tuple returns
            if isinstance(result, tuple):
                return result[0]
            return result
        except ImportError:
            # Ultimate fallback: simple correlation-based metric
            return torch.tensor(1.0 - F.mse_loss(img1, img2) / (self.data_range**2))

    def compute(self) -> dict[str, float]:
        """Compute final metrics with mean±std statistics."""
        metrics: dict[str, float] = {
            "psnr": 0.0,
            "ssim": 0.0,
            "psnr_mean": 0.0,
            "psnr_std": 0.0,
            "ssim_mean": 0.0,
            "ssim_std": 0.0,
        }

        # PSNR
        if self.psnr_values:
            psnr_mean = sum(self.psnr_values) / len(self.psnr_values)
            psnr_std = (
                (sum((x - psnr_mean) ** 2 for x in self.psnr_values) / len(self.psnr_values)) ** 0.5
                if len(self.psnr_values) > 1
                else 0.0
            )
            metrics["psnr"] = psnr_mean
            metrics["psnr_mean"] = psnr_mean
            metrics["psnr_std"] = psnr_std

        # SSIM
        if self.ssim_values:
            ssim_mean = sum(self.ssim_values) / len(self.ssim_values)
            ssim_std = (
                (sum((x - ssim_mean) ** 2 for x in self.ssim_values) / len(self.ssim_values)) ** 0.5
                if len(self.ssim_values) > 1
                else 0.0
            )
            metrics["ssim"] = ssim_mean
            metrics["ssim_mean"] = ssim_mean
            metrics["ssim_std"] = ssim_std

        # LPIPS
        if self.lpips_values:
            lpips_mean = sum(self.lpips_values) / len(self.lpips_values)
            lpips_std = (
                (sum((x - lpips_mean) ** 2 for x in self.lpips_values) / len(self.lpips_values))
                ** 0.5
                if len(self.lpips_values) > 1
                else 0.0
            )
            metrics["lpips"] = lpips_mean
            metrics["lpips_mean"] = lpips_mean
            metrics["lpips_std"] = lpips_std

        return metrics

    def compute_with_stats(self) -> dict[str, Any]:
        """
        Compute metrics with full statistics (mean, std, min, max, count).

        Returns:
            Dictionary with full statistics for each metric
        """
        stats = {}

        for metric_name, values in [
            ("psnr", self.psnr_values),
            ("ssim", self.ssim_values),
            ("lpips", self.lpips_values),
        ]:
            if values:
                stats[metric_name] = {
                    "mean": sum(values) / len(values),
                    "std": (
                        (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values))
                        ** 0.5
                        if len(values) > 1
                        else 0.0
                    ),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        return stats

    def export_json(self, path: str) -> None:
        """
        Export metrics to JSON file.

        Args:
            path: Path to save JSON file
        """
        import json

        metrics = self.compute()
        stats = self.compute_with_stats()

        data = {
            "summary": metrics,
            "detailed_stats": stats,
            "per_batch_values": {
                "psnr": self.psnr_values,
                "ssim": self.ssim_values,
                "lpips": self.lpips_values,
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, path: str) -> None:
        """
        Export per-batch metrics to CSV file.

        Args:
            path: Path to save CSV file
        """
        import csv

        # Determine max length
        max_len = max(len(self.psnr_values), len(self.ssim_values), len(self.lpips_values))

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header
            header = ["batch"]
            if self.psnr_values:
                header.append("psnr")
            if self.ssim_values:
                header.append("ssim")
            if self.lpips_values:
                header.append("lpips")
            writer.writerow(header)

            # Data rows
            for i in range(max_len):
                row = [i]
                if i < len(self.psnr_values):
                    row.append(self.psnr_values[i])
                if i < len(self.ssim_values):
                    row.append(self.ssim_values[i])
                if i < len(self.lpips_values):
                    row.append(self.lpips_values[i])
                writer.writerow(row)


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
        grad_clip: float | None = None,
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

    def train_step(  # type: ignore[override]
        self,
        model: MaskGITModelInterface,
        batch: tuple[torch.Tensor, ...],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
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
        x = batch[0] if isinstance(batch, tuple | list) else batch

        model.train()
        maskgit_model = cast(MaskGITModelInterface, model)

        # Forward pass with masking and mixed precision
        with self.mixed_precision.autocast_context():
            # Use getattr to safely access compute_maskgit_loss if it exists
            compute_loss_fn = getattr(maskgit_model, "compute_maskgit_loss", None)
            if compute_loss_fn is not None:
                loss, metrics = compute_loss_fn(x, mask_ratio=self.mask_ratio)
            else:
                # Manual implementation for compatibility
                tokens = maskgit_model.encode_tokens(x)
                B, D, H, W = tokens.shape
                tokens_flat = tokens.view(B, -1)

                # Random masking
                mask = torch.rand(B, D * H * W, device=tokens.device) < self.mask_ratio

                # Ensure at least one token masked per sample
                for i in range(B):
                    if not mask[i].any():
                        mask[i, torch.randint(0, D * H * W, (1,), device=tokens.device)] = True

                # Get predictions - access transformer via the model
                transformer = getattr(maskgit_model, "transformer", None)
                if transformer is not None:
                    logits = transformer.forward(tokens_flat, mask_indices=mask)
                else:
                    # Fallback: use model directly if it has forward method
                    logits = maskgit_model.forward(tokens_flat)

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
        batch: tuple[torch.Tensor, ...],
    ) -> dict[str, float]:
        """
        Execute validation step.

        Args:
            model: MaskGITModel instance
            batch: Tuple of (input_images,)

        Returns:
            Dictionary of validation metrics
        """
        model.eval()
        maskgit_model = cast(MaskGITModelInterface, model)

        x = batch[0] if isinstance(batch, tuple | list) else batch

        with torch.no_grad():
            # Reconstruction
            x_rec = maskgit_model.decode_tokens(maskgit_model.encode_tokens(x))

            # Compute reconstruction loss
            rec_loss = torch.abs(x - x_rec).mean()

            # Encode and compute token accuracy
            tokens = maskgit_model.encode_tokens(x)
            # Flatten tokens to [B, N] if needed
            if tokens.dim() > 2:
                tokens = tokens.view(tokens.shape[0], -1)
            elif tokens.dim() == 1:
                # Single batch or flattened - infer batch size
                B = x.shape[0]
                tokens = tokens.view(B, -1)
            transformer = getattr(maskgit_model, "transformer", None)
            if transformer is not None:
                tokens_pred = transformer.encode(tokens, return_logits=True).argmax(dim=-1)
            else:
                tokens_pred = tokens.argmax(dim=1)
            token_acc = (tokens_pred == tokens).float().mean()

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
        maskgit_model = cast(MaskGITModelInterface, model)

        if self.mode == "reconstruct":
            with torch.no_grad():
                tokens = maskgit_model.encode_tokens(batch)
                return maskgit_model.decode_tokens(tokens)

        elif self.mode == "generate":
            with torch.no_grad():
                # Get latent shape
                latent_shape = maskgit_model.latent_shape
                B = batch.shape[0]

                # Generate
                return maskgit_model.generate(
                    shape=(B,) + latent_shape[1:],
                    temperature=self.temperature,
                    num_iterations=self.num_iterations,
                )

        raise ValueError(f"Unknown mode: {self.mode}")

    def post_process(
        self,
        predictions: torch.Tensor,
    ) -> dict[str, Any]:
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


# =============================================================================
# Sliding Window Inference Strategies
# =============================================================================


class SlidingWindowVQGANInference(InferenceStrategy):
    """
    VQGAN inference strategy with sliding window support for large volumes.

    Uses MONAI's sliding_window_inference to process volumes that are larger
    than the training crop size. This is essential for:
    - BraTS: Full-resolution MRI volumes (~240x240x155)
    - Any volumes larger than training size

    The sliding window approach:
    1. Divides the input volume into overlapping patches
    2. Processes each patch through the VQGAN model
    3. Blends the results using Gaussian weighting

    Note: For VQVAE models, roi_size must be divisible by the downsampling
    factor (default 16) and overlap * roi_size must be an integer.
    """

    def __init__(
        self,
        roi_size: tuple[int, int, int] = (128, 128, 128),
        sw_batch_size: int = 4,
        overlap: float = 0.25,
        mode: str = "gaussian",
        sigma_scale: float = 0.125,
        progress: bool = False,
        downsampling_factor: int = 16,
        original_size: tuple[int, int, int] | None = None,
    ):
        """
        Initialize sliding window VQGAN inference.

        Args:
            roi_size: Region of interest size for each window (D, H, W).
                Should match the training crop size. Must be divisible by
                downsampling_factor for VQVAE compatibility.
            sw_batch_size: Number of windows to process in parallel.
                Higher values use more GPU memory but are faster.
            overlap: Overlap ratio between windows (0-1).
                0.25 is typical. Must satisfy: overlap * roi_size is integer.
            mode: Blending mode - "gaussian" or "constant".
                Gaussian provides smoother transitions between patches.
            sigma_scale: Sigma scale for Gaussian blending.
                Controls how quickly the weight decays near patch edges.
            progress: Show progress bar during inference.
            downsampling_factor: VQVAE downsampling factor for validation.
                Default is 16 (4 downsampling layers with stride 2).
            original_size: Original spatial size (D, H, W) before padding.
                If provided, output will be cropped back to this size.
        """
        from maskgit3d.infrastructure.data.padding import validate_roi_size

        self.roi_size = validate_roi_size(roi_size, overlap, downsampling_factor)
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode
        self.sigma_scale = sigma_scale
        self.progress = progress
        self.original_size = original_size

    def predict(
        self,
        model: ModelInterface,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run sliding window inference.

        Args:
            model: VQModel instance
            batch: Input volumes [B, C, D, H, W]

        Returns:
            Reconstructed volumes [B, C, D, H, W]
        """
        from monai.inferers.utils import sliding_window_inference

        model.eval()
        device = model.device
        vq_model = cast(VQModelInterface, model)

        with torch.no_grad():
            output = sliding_window_inference(
                inputs=batch,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=vq_model.forward,
                overlap=self.overlap,
                mode=self.mode,
                sigma_scale=self.sigma_scale,
                sw_device=device,
                device=device,
                progress=self.progress,
            )

            if self.original_size is not None:
                from maskgit3d.infrastructure.data.padding import compute_output_crop

                padded_size = (batch.shape[2], batch.shape[3], batch.shape[4])
                crop_slices = compute_output_crop(self.original_size, padded_size)
                # crop_slices is a tuple of 3 slices for (D, H, W)
                d_slice, h_slice, w_slice = crop_slices
                output = output[:, :, d_slice, h_slice, w_slice]  # type: ignore[index]

        return output  # type: ignore[return-value]

    def post_process(
        self,
        predictions: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Post-process predictions.

        Args:
            predictions: Raw model outputs

        Returns:
            Processed predictions dictionary
        """
        images = predictions.cpu().float()
        images = (images + 1) / 2
        images = images.clamp(0, 1)

        return {
            "images": images.numpy(),
        }


class SlidingWindowVQGANLatentExtractor(InferenceStrategy):
    """
    VQGAN inference strategy for extracting latent codes with sliding window.

    This is designed for the second stage (MaskGIT training):
    1. Extract latent representations using sliding window
    2. Convert latents to codebook indices
    3. Return both latents and indices for downstream use

    The sliding window approach ensures that:
    - Large volumes can be processed without OOM
    - Latent codes maintain spatial correspondence with input
    """

    def __init__(
        self,
        roi_size: tuple[int, int, int] = (128, 128, 128),
        sw_batch_size: int = 4,
        overlap: float = 0.25,
        mode: str = "gaussian",
        sigma_scale: float = 0.125,
        progress: bool = False,
    ):
        """
        Initialize sliding window latent extractor.

        Args:
            roi_size: Region of interest size for each window (D, H, W).
            sw_batch_size: Number of windows to process in parallel.
            overlap: Overlap ratio between windows (0-1).
            mode: Blending mode - "gaussian" or "constant".
            sigma_scale: Sigma scale for Gaussian blending.
            progress: Show progress bar during inference.
        """
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode
        self.sigma_scale = sigma_scale
        self.progress = progress

    def _encode_patch(
        self,
        model: VQModelInterface,
        patch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single patch to latent and indices.

        Args:
            model: VQModel instance
            patch: Input patch [B, C, D, H, W]

        Returns:
            Tuple of (quantized_latent, indices)
        """
        # Access internal components using getattr with type assertion
        encoder = cast(nn.Module, getattr(model, "encoder", None))
        quant_conv = cast(nn.Module, getattr(model, "quant_conv", None))
        quantize = cast(nn.Module, getattr(model, "quantize", None))

        assert encoder is not None, "Model must have encoder attribute"
        assert quant_conv is not None, "Model must have quant_conv attribute"
        assert quantize is not None, "Model must have quantize attribute"

        h = encoder(patch)
        h = quant_conv(h)
        quant, _, info = quantize(h)
        indices = info[2]  # min_encoding_indices
        return quant, indices

    def predict(
        self,
        model: ModelInterface,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run sliding window latent extraction.

        Args:
            model: VQModel instance
            batch: Input volumes [B, C, D, H, W]

        Returns:
            Codebook indices [B, D', H', W'] where D', H', W' are latent dimensions
        """
        from monai.inferers.utils import sliding_window_inference

        model.eval()
        device = model.device
        vq_model = cast(VQModelInterface, model)

        # Get encoder and quant_conv for shape inspection
        encoder = cast(nn.Module, getattr(vq_model, "encoder", None))
        quant_conv = cast(nn.Module, getattr(vq_model, "quant_conv", None))
        quantize = cast(nn.Module, getattr(vq_model, "quantize", None))

        assert encoder is not None and quant_conv is not None and quantize is not None

        with torch.no_grad():
            # Test with a dummy to get actual latent size
            dummy_out = quant_conv(encoder(batch[:1]))
            _, d_latent, h_latent, w_latent = dummy_out.shape

        B = batch.shape[0]
        c_latent = vq_model.latent_shape[0]

        # Create predictor that returns quantized latent
        def encode_predictor(x: torch.Tensor) -> torch.Tensor:
            quant, _ = self._encode_patch(vq_model, x)
            return quant

        with torch.no_grad():
            # Get quantized latents via sliding window
            quantized_latent = sliding_window_inference(
                # type: ignore[assignment]
                inputs=batch,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=encode_predictor,
                overlap=self.overlap,
                mode=self.mode,
                sigma_scale=self.sigma_scale,
                sw_device=device,
                device=device,
                progress=self.progress,
            )

            # Now get indices from the quantized latent
            # Reshape for quantize: (B, C, D, H, W) -> (B, D, H, W, C)
            quant_flat = (
                cast(torch.Tensor, quantized_latent).permute(0, 2, 3, 4, 1).reshape(-1, c_latent)
            )

            # Get indices by finding nearest codebook entry
            embedding = cast(nn.Embedding, getattr(quantize, "embedding", None))
            assert embedding is not None, "quantize must have embedding attribute"

            dist = (
                torch.sum(quant_flat**2, dim=1, keepdim=True)
                + torch.sum(embedding.weight**2, dim=1)
                - 2 * torch.matmul(quant_flat, embedding.weight.t())
            )
            indices = torch.argmin(dist, dim=1)

            # Reshape indices to spatial
            indices = indices.reshape(B, d_latent, h_latent, w_latent)

        return indices

    def extract_latent_and_indices(
        self,
        model: ModelInterface,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract both latent representations and codebook indices.

        Args:
            model: VQModel instance
            batch: Input volumes [B, C, D, H, W]

        Returns:
            Tuple of (quantized_latent, indices)
            - quantized_latent: [B, C', D', H', W']
            - indices: [B, D', H', W']
        """
        from monai.inferers.utils import sliding_window_inference

        model.eval()
        device = model.device
        vq_model = cast(VQModelInterface, model)

        # Get internal components
        encoder = cast(nn.Module, getattr(vq_model, "encoder", None))
        quant_conv = cast(nn.Module, getattr(vq_model, "quant_conv", None))
        quantize = cast(nn.Module, getattr(vq_model, "quantize", None))

        assert encoder is not None and quant_conv is not None and quantize is not None

        with torch.no_grad():
            # Get latent shape
            dummy_out = quant_conv(encoder(batch[:1]))
            c_latent, d_latent, h_latent, w_latent = dummy_out.shape[1:]

        B = batch.shape[0]

        def encode_predictor(x: torch.Tensor) -> torch.Tensor:
            """Returns encoder output (before quantization) for blending."""
            h = encoder(x)
            h = quant_conv(h)
            return h

        with torch.no_grad():
            # Get encoder output via sliding window (blended)
            encoder_output = sliding_window_inference(
                inputs=batch,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=encode_predictor,
                overlap=self.overlap,
                mode=self.mode,
                sigma_scale=self.sigma_scale,
                sw_device=device,
                device=device,
                progress=self.progress,
            )

            # Quantize the blended encoder output
            quantized_latent, _, info = quantize(encoder_output)

            # Get indices from the quantized result
            # Note: This gives indices for the blended latent, which is a reasonable approximation
            indices = info[2]  # min_encoding_indices
            indices = indices.reshape(B, d_latent, h_latent, w_latent)

        return quantized_latent, indices

    def post_process(
        self,
        predictions: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Post-process predictions.

        Args:
            predictions: Codebook indices

        Returns:
            Dictionary with indices
        """
        return {
            "indices": predictions.cpu().numpy(),
        }
