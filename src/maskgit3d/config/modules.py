"""
Configuration layer - Dependency injection modules.

This module provides Injector modules that bind interfaces
to concrete implementations for the application.
"""

from typing import Any, Literal

import torch
from injector import Module, provider, singleton

from maskgit3d.application.pipeline import FabricTrainingPipeline
from maskgit3d.domain.interfaces import (
    DataProvider,
    InferenceStrategy,
    MaskGITModelInterface,
    Metrics,
    ModelInterface,
    OptimizerFactory,
    TrainingStrategy,
    VQModelInterface,
)
from maskgit3d.infrastructure.metrics.fid_2p5d import FID2p5DMetric
from maskgit3d.infrastructure.training.strategies import (
    AdamOptimizerFactory,
    AdamWOptimizerFactory,
    MaskGITInference,
    MaskGITTrainingStrategy,
    SGDOptimizerFactory,
    VQGANInference,
    VQGANMetrics,
    VQGANOptimizerFactory,
    VQGANTrainingStrategy,
)
from maskgit3d.infrastructure.vqgan import (
    VQVAE,
    get_vqvae_config,
)

# =============================================================================
# Parameter Validation
# =============================================================================


def _validate_param(
    name: str, value: int, min_val: int | None = None, max_val: int | None = None
) -> None:
    """Validate integer parameter with optional min/max bounds."""
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")


def _validate_float_param(
    name: str, value: float, min_val: float | None = None, max_val: float | None = None
) -> None:
    """Validate float parameter with optional min/max bounds."""
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")


# =============================================================================
# Core Modules
# =============================================================================


class ModelModule(Module):
    """
    Module for model configuration.

    Provides model implementations for different tasks.
    """

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
    ):
        """
        Initialize model module.

        Args:
            model_config: Configuration dictionary for model
        """
        self.model_config = model_config or {}

    @singleton
    @provider
    def provide_model(self) -> ModelInterface:
        """
        Provide model instance.

        Returns:
            Configured model
        """
        model_type = self.model_config.get("type", "maskgit")
        model_params = self.model_config.get("params", {})

        if model_type == "maskgit":
            from maskgit3d.infrastructure.maskgit import MaskGITModel

            return MaskGITModel(**model_params)
        elif model_type in ("vqgan", "vqgan3d", "vqvae"):
            return VQVAE(**model_params)
        raise ValueError(f"Unknown model type: {model_type}")


class DataModule(Module):
    """
    Module for data configuration.

    Provides data providers for training, validation, and testing.
    """

    def __init__(
        self,
        data_config: dict[str, Any] | None = None,
    ):
        """
        Initialize data module.

        Args:
            data_config: Configuration dictionary for data
        """
        self.data_config = data_config or {}

    @singleton
    @provider
    def provide_data_provider(self) -> DataProvider:
        """
        Provide data provider instance.

        Returns:
            Configured data provider
        """
        from maskgit3d.infrastructure.data.brats_provider import BraTSDataProvider
        from maskgit3d.infrastructure.data.dataset import SimpleDataProvider
        from maskgit3d.infrastructure.data.medmnist_provider import MedMnist3DDataProvider

        data_type = self.data_config.get("type", "simple")
        data_params = self.data_config.get("params", {})

        providers = {
            "simple": SimpleDataProvider,
            "medmnist3d": MedMnist3DDataProvider,
            "brats": BraTSDataProvider,
        }

        if data_type not in providers:
            raise ValueError(f"Unknown data type: {data_type}. Available: {list(providers.keys())}")

        return providers[data_type](**data_params)


class TrainingModule(Module):
    """
    Module for training configuration.

    Provides training strategies, optimizers, and schedulers.
    """

    def __init__(
        self,
        training_config: dict[str, Any] | None = None,
        optimizer_config: dict[str, Any] | None = None,
    ):
        """
        Initialize training module.

        Args:
            training_config: Configuration for training strategy
            optimizer_config: Configuration for optimizer
        """
        self.training_config = training_config or {}
        self.optimizer_config = optimizer_config or {}

    @singleton
    @provider
    def provide_training_strategy(self) -> TrainingStrategy:
        """
        Provide training strategy instance.

        Returns:
            Configured training strategy
        """
        strategy_type = self.training_config.get("type", "maskgit")
        strategy_params = self.training_config.get("params", {})

        strategies = {
            "maskgit": MaskGITTrainingStrategy,
            "vqgan": VQGANTrainingStrategy,
        }

        if strategy_type not in strategies:
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. Available: {list(strategies.keys())}"
            )

        return strategies[strategy_type](**strategy_params)

    @singleton
    @provider
    def provide_optimizer_factory(self) -> OptimizerFactory:
        """
        Provide optimizer factory instance.

        Returns:
            Configured optimizer factory
        """
        optimizer_type = self.optimizer_config.get("type", "adam")
        optimizer_params = self.optimizer_config.get("params", {})

        optimizers = {
            "adam": AdamOptimizerFactory,
            "sgd": SGDOptimizerFactory,
            "adamw": AdamWOptimizerFactory,
            "vqgan": VQGANOptimizerFactory,
        }

        if optimizer_type not in optimizers:
            raise ValueError(
                f"Unknown optimizer type: {optimizer_type}. Available: {list(optimizers.keys())}"
            )

        return optimizers[optimizer_type](**optimizer_params)


class InferenceModule(Module):
    """
    Module for inference configuration.

    Provides inference strategies and metrics.
    """

    def __init__(
        self,
        inference_config: dict[str, Any] | None = None,
        metrics_config: dict[str, Any] | None = None,
    ):
        """
        Initialize inference module.

        Args:
            inference_config: Configuration for inference strategy
            metrics_config: Configuration for metrics
        """
        self.inference_config = inference_config or {}
        self.metrics_config = metrics_config or {}

    @singleton
    @provider
    def provide_inference_strategy(self) -> InferenceStrategy:
        """
        Provide inference strategy instance.

        Returns:
            Configured inference strategy
        """
        inference_type = self.inference_config.get("type", "maskgit")
        inference_params = self.inference_config.get("params", {})

        strategies = {
            "maskgit": MaskGITInference,
            "vqgan": VQGANInference,
        }

        if inference_type not in strategies:
            raise ValueError(
                f"Unknown inference type: {inference_type}. Available: {list(strategies.keys())}"
            )

        return strategies[inference_type](**inference_params)

    @singleton
    @provider
    def provide_metrics(self) -> Metrics | None:
        """
        Provide metrics instance.

        Returns:
            Configured metrics or None if not configured
        """
        metrics_type = self.metrics_config.get("type")
        if not metrics_type:
            return None

        metrics_params = self.metrics_config.get("params", {})

        metrics = {
            "vqgan": VQGANMetrics,
            "fid_2p5d": FID2p5DMetric,
        }

        if metrics_type not in metrics:
            raise ValueError(
                f"Unknown metrics type: {metrics_type}. Available: {list(metrics.keys())}"
            )

        return metrics[metrics_type](**metrics_params)


class SystemModule(Module):
    """
    Module for system configuration.

    Provides device and other system-level configurations.
    """

    @singleton
    @provider
    def provide_device(self) -> torch.device:
        """
        Provide device (CPU/CUDA).

        Returns:
            torch.device
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# VQVAE Module
# =============================================================================


class VQVAEModule(Module):
    """
    Composite module for VQVAE tasks.

    Provides VQVAE-specific bindings for model, training, and inference.
    """

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
        data_config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        optimizer_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
        metrics_config: dict[str, Any] | None = None,
    ):
        self.model_config = model_config or {}
        self.training_config = training_config or {}
        self.optimizer_config = optimizer_config or {}
        self.inference_config = inference_config or {}
        self.metrics_config = metrics_config or {}

    def configure(self, binder):
        """Configure VQVAE bindings."""
        # Bind VQVAE Model
        model_params = self.model_config.get("params", {})
        vqvae_model = VQVAE(**model_params)
        binder.bind(VQModelInterface, to=lambda: vqvae_model)
        binder.bind(ModelInterface, to=lambda: vqvae_model)

        # Training Strategy
        strategy_params = self.training_config.get("params", {})
        binder.bind(TrainingStrategy, to=lambda: VQGANTrainingStrategy(**strategy_params))

        # Optimizer Factory
        binder.bind(
            VQGANOptimizerFactory,
            to=lambda: VQGANOptimizerFactory(**self.optimizer_config.get("params", {})),
        )

        # Inference Strategy
        inf_params = self.inference_config.get("params", {})
        binder.bind(InferenceStrategy, to=lambda: VQGANInference(**inf_params))

        # Metrics
        metrics_type = self.metrics_config.get("type")
        if metrics_type == "vqgan":
            binder.bind(Metrics, to=lambda: VQGANMetrics(**self.metrics_config.get("params", {})))


def create_vqvae_module(
    image_size: int = 64,
    in_channels: int = 1,
    codebook_size: int = 1024,
    embed_dim: int = 256,
    latent_channels: int = 4,
    num_channels: tuple[int, ...] = (64, 128, 256),
    num_res_blocks: tuple[int, ...] = (2, 2, 2),
    attention_levels: tuple[bool, ...] = (False, False, False),
    lr: float = 1e-4,
    batch_size: int = 1,
) -> VQVAEModule:
    """
    Factory function to create a VQVAE module with sensible defaults.

    Args:
        image_size: Input volume size (for reference)
        in_channels: Number of input channels
        codebook_size: Number of codebook entries
        embed_dim: Codebook embedding dimension
        latent_channels: Number of latent channels (encoder output)
        num_channels: Channel numbers for each level
        num_res_blocks: Number of residual blocks per level
        attention_levels: Whether to use attention at each level
        lr: Learning rate (same for both Generator and Discriminator)
        batch_size: Batch size

    Returns:
        Configured VQVAEModule
    """
    # Validate parameters
    _validate_param("image_size", image_size, min_val=8)
    _validate_param("in_channels", in_channels, min_val=1)
    _validate_param("codebook_size", codebook_size, min_val=1)
    _validate_param("embed_dim", embed_dim, min_val=1)
    _validate_param("latent_channels", latent_channels, min_val=1)
    _validate_float_param("lr", lr, min_val=1e-10)
    _validate_param("batch_size", batch_size, min_val=1)

    # Model config
    model_config = {
        "type": "vqvae",
        "params": get_vqvae_config(
            image_size=image_size,
            in_channels=in_channels,
            codebook_size=codebook_size,
            embed_dim=embed_dim,
            latent_channels=latent_channels,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
        ),
    }

    # Training config
    training_config = {
        "type": "vqgan",
        "params": {
            "codebook_weight": 1.0,
            "pixel_loss_weight": 1.0,
            "perceptual_weight": 0.3,
            "disc_weight": 0.1,
            "disc_start": 10000,
        },
    }

    # Use same learning rate for both Generator and Discriminator
    optimizer_config = {
        "type": "vqgan",
        "params": {
            "lr": lr,  # Single learning rate for both G and D
            "weight_decay": 0.01,
            "betas": (0.5, 0.9),
        },
    }

    # Inference config
    inference_config = {
        "type": "vqgan",
        "params": {
            "mode": "reconstruct",
        },
    }

    # Metrics config
    metrics_config = {
        "type": "vqgan",
        "params": {
            "data_range": 1.0,
            "spatial_dims": 3,
            "enable_lpips": True,
            "lpips_backbone": "alex",
        },
    }

    return VQVAEModule(
        model_config=model_config,
        training_config=training_config,
        optimizer_config=optimizer_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
    )


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Keep old names for backward compatibility
VQGANModelModule = VQVAEModule
VQGANModule = VQVAEModule
MaisiVQModelModule = VQVAEModule
MaisiVQModule = VQVAEModule

# Aliases for factory functions
create_vqgan_module = create_vqvae_module
create_maisi_vq_module = create_vqvae_module


# =============================================================================
# MaskGIT Modules
# =============================================================================


class MaskGITModelModule(Module):
    """MaskGIT model configuration module."""

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
        pretrained_vqvae_path: str | None = None,
        freeze_vqvae: bool = True,
    ):
        self.model_config = model_config or {}
        self.pretrained_vqvae_path = pretrained_vqvae_path
        self.freeze_vqvae = freeze_vqvae

    @singleton
    @provider
    def provide_maskgit_model(self) -> MaskGITModelInterface:
        """Provide MaskGIT model instance."""
        model_type = self.model_config.get("type", "maskgit")
        model_params = self.model_config.get("params", {})

        if model_type == "maskgit":
            from maskgit3d.infrastructure.maskgit import MaskGITModel
            from maskgit3d.infrastructure.maskgit.transformer import MaskGITTransformer

            vqvae = VQVAE(
                in_channels=model_params["in_channels"],
                codebook_size=model_params["codebook_size"],
                embed_dim=model_params["embed_dim"],
                latent_channels=model_params["latent_channels"],
            )

            if self.pretrained_vqvae_path is not None:
                self._load_pretrained_vqvae(vqvae, self.pretrained_vqvae_path)

            if self.freeze_vqvae:
                for param in vqvae.parameters():
                    param.requires_grad = False

            transformer = MaskGITTransformer(
                vocab_size=model_params["codebook_size"] + 1,
                mask_token_id=model_params.get("mask_token_id", model_params["codebook_size"]),
                hidden_size=model_params["transformer_hidden"],
                num_layers=model_params["transformer_layers"],
                num_heads=model_params["transformer_heads"],
            )

            model = MaskGITModel(
                vqgan=vqvae,
                transformer=transformer,
                mask_ratio=model_params.get("mask_ratio", 0.5),
            )
            return model
        raise ValueError(f"Unknown MaskGIT model type: {model_type}")

    def _load_pretrained_vqvae(self, vqvae: "VQVAE", checkpoint_path: str) -> None:
        """Load pretrained VQVAE weights from checkpoint.

        Handles both stage 1 format (model_state_dict) and stage 2 format (vqvae).
        """

        from maskgit3d.infrastructure.checkpoints import load_checkpoint

        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

        if "model_state_dict" in checkpoint:
            vqvae.load_state_dict(checkpoint["model_state_dict"])
        elif "vqvae" in checkpoint:
            vqvae.load_state_dict(checkpoint["vqvae"])
        elif "vqgan" in checkpoint:
            vqvae.load_state_dict(checkpoint["vqgan"])
        elif "state_dict" in checkpoint:
            vqvae.load_state_dict(checkpoint["state_dict"])
        else:
            vqvae.load_state_dict(checkpoint)


class MaskGITModule(Module):
    """
    Composite module for MaskGIT tasks.

    Provides MaskGIT-specific bindings for model, training, and inference.
    """

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
        data_config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        optimizer_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
        metrics_config: dict[str, Any] | None = None,
        pretrained_vqvae_path: str | None = None,
        freeze_vqvae: bool = True,
    ):
        """
        Initialize MaskGIT module.

        Args:
            model_config: MaskGIT model configuration
            data_config: Data configuration
            training_config: Training strategy configuration
            optimizer_config: Optimizer configuration
            inference_config: Inference configuration
            metrics_config: Metrics configuration
            pretrained_vqvae_path: Path to pretrained VQVAE checkpoint
            freeze_vqvae: Whether to freeze VQVAE weights
        """
        self.model_module = MaskGITModelModule(
            model_config,
            pretrained_vqvae_path=pretrained_vqvae_path,
            freeze_vqvae=freeze_vqvae,
        )
        self.model_config = model_config or {}
        self.training_config = training_config or {}
        self.optimizer_config = optimizer_config or {}
        self.inference_config = inference_config or {}
        self.metrics_config = metrics_config or {}

    def configure(self, binder):
        """configure MaskGIT bindings."""
        # Bind MaskGIT Model
        maskgit_model = self.model_module.provide_maskgit_model()
        binder.bind(MaskGITModelInterface, to=lambda: maskgit_model)
        binder.bind(ModelInterface, to=lambda: maskgit_model)

        # Training Strategy
        strategy_params = self.training_config.get("params", {})
        binder.bind(TrainingStrategy, to=lambda: MaskGITTrainingStrategy(**strategy_params))

        # Optimizer Factory
        optimizer_params = self.optimizer_config.get("params", {})
        binder.bind(OptimizerFactory, to=lambda: AdamOptimizerFactory(**optimizer_params))

        # Inference Strategy
        inf_params = self.inference_config.get("params", {})
        binder.bind(InferenceStrategy, to=lambda: MaskGITInference(**inf_params))

        # Metrics
        metrics_type = self.metrics_config.get("type")
        if metrics_type == "maskgit":
            binder.bind(Metrics, to=lambda: VQGANMetrics(**self.metrics_config.get("params", {})))


def create_maskgit_module(
    image_size: int = 64,
    in_channels: int = 1,
    codebook_size: int = 1024,
    embed_dim: int = 256,
    latent_channels: int = 256,
    transformer_hidden: int = 768,
    transformer_layers: int = 12,
    transformer_heads: int = 12,
    mask_ratio: float = 0.5,
    lr: float = 1e-4,
    batch_size: int = 1,
    num_train: int = 1000,
    num_val: int = 100,
    pretrained_vqvae_path: str | None = None,
    freeze_vqvae: bool = True,
) -> MaskGITModule:
    """
    Factory function to create a MaskGIT module with sensible defaults.

    Args:
        image_size: Input volume size
        in_channels: Number of input channels (1 for MRI/CT)
        codebook_size: Size of VQ codebook
        embed_dim: Embedding dimension for codebook
        latent_channels: Latent space channels
        transformer_hidden: Hidden dimension for transformer
        transformer_layers: Number of transformer layers
        transformer_heads: Number of attention heads
        mask_ratio: Ratio of tokens to mask during training
        lr: Learning rate
        batch_size: Batch size
        num_train: Number of training samples
        num_val: Number of validation samples
        pretrained_vqvae_path: Path to pretrained VQVAE checkpoint from stage 1
        freeze_vqvae: Whether to freeze VQVAE weights during training

    Returns:
        Configured MaskGITModule
    """
    # Validate parameters
    _validate_param("image_size", image_size, min_val=8)
    _validate_param("in_channels", in_channels, min_val=1)
    _validate_param("codebook_size", codebook_size, min_val=1)
    _validate_param("embed_dim", embed_dim, min_val=1)
    _validate_param("latent_channels", latent_channels, min_val=1)
    _validate_param("transformer_hidden", transformer_hidden, min_val=1)
    _validate_param("transformer_layers", transformer_layers, min_val=1)
    _validate_param("transformer_heads", transformer_heads, min_val=1)
    _validate_float_param("mask_ratio", mask_ratio, min_val=0.0, max_val=1.0)
    _validate_float_param("lr", lr, min_val=1e-10)
    _validate_param("batch_size", batch_size, min_val=1)
    _validate_param("num_train", num_train, min_val=1)
    _validate_param("num_val", num_val, min_val=1)

    # Model config
    model_config = {
        "type": "maskgit",
        "params": {
            "in_channels": in_channels,
            "codebook_size": codebook_size,
            "embed_dim": embed_dim,
            "latent_channels": latent_channels,
            "transformer_hidden": transformer_hidden,
            "transformer_layers": transformer_layers,
            "transformer_heads": transformer_heads,
            "mask_ratio": mask_ratio,
        },
    }

    # Training config
    training_config = {
        "type": "maskgit",
        "params": {
            "mask_ratio": mask_ratio,
            "reconstruction_weight": 1.0,
        },
    }

    optimizer_config = {
        "type": "adam",
        "params": {
            "lr": lr,
        },
    }

    # Inference config
    inference_config = {
        "type": "maskgit",
        "params": {
            "mode": "generate",
            "num_iterations": 12,
            "temperature": 1.0,
        },
    }

    # Metrics config
    metrics_config = {
        "type": "maskgit",
        "params": {
            "data_range": 1.0,
            "spatial_dims": 3,
        },
    }

    return MaskGITModule(
        model_config=model_config,
        training_config=training_config,
        optimizer_config=optimizer_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        pretrained_vqvae_path=pretrained_vqvae_path,
        freeze_vqvae=freeze_vqvae,
    )


# =============================================================================
# Fabric Pipeline Factories
# =============================================================================


def create_fabric_pipeline(
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
) -> FabricTrainingPipeline:
    """
    Factory function to create a FabricTrainingPipeline with specified configuration.

    Args:
        model: Model to train
        data_provider: Data provider for train/val loaders
        training_strategy: Training strategy with loss computation
        optimizer_factory: Factory for creating optimizer
        accelerator: Fabric accelerator ("cpu", "cuda", "auto")
        devices: Number of devices or device IDs
        strategy: Fabric strategy ("auto", "ddp", "fsdp")
        precision: Training precision ("32-true", "16-mixed", "bf16-mixed")
        checkpoint_dir: Directory to save checkpoints
        log_interval: Interval for logging training progress

    Returns:
        Configured FabricTrainingPipeline instance
    """
    return FabricTrainingPipeline(
        model=model,
        data_provider=data_provider,
        training_strategy=training_strategy,
        optimizer_factory=optimizer_factory,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        checkpoint_dir=checkpoint_dir,
    )
