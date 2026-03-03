"""
Configuration layer - Dependency injection modules.

This module provides Injector modules that bind interfaces
to concrete implementations for the application.
"""

from typing import Any

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
)
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
    MaisiVQModel3D,
    VQModel3D,
    get_encoder_decoder_config_3d,
    get_maisi_vq_config,
)

# =============================================================================
# Parameter Validation
# =============================================================================


def _validate_param(name: str, value: int, min_val: int | None = None, max_val: int | None = None) -> None:
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
        elif model_type in ("vqgan3d", "vqgan"):
            from maskgit3d.infrastructure.vqgan import VQModel3D

            return VQModel3D(**model_params)
        elif model_type == "maisi_vq":
            from maskgit3d.infrastructure.vqgan import MaisiVQModel3D

            return MaisiVQModel3D(**model_params)
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
# VQGAN Modules (3D)
# =============================================================================


class VQGANModelModule(Module):
    """VQGAN 3D model configuration module."""

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
    ):
        self.model_config = model_config or {}

    @singleton
    @provider
    def provide_vqgan_model(self) -> VQModel3D:
        """Provide VQGAN 3D model instance."""
        model_type = self.model_config.get("type", "vqgan3d")
        model_params = self.model_config.get("params", {})

        if model_type == "vqgan3d":
            return VQModel3D(**model_params)
        raise ValueError(f"Unknown VQ model type: {model_type}")


class VQGANModule(Module):
    """
    Composite module for 3D VQGAN tasks.

    Provides VQGAN-specific bindings for model, training, and inference.
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
        """
        Initialize VQGAN module.

        Args:
            model_config: VQGAN model configuration
            data_config: Data configuration
            training_config: Training strategy configuration
            optimizer_config: Optimizer configuration
            inference_config: Inference configuration
            metrics_config: Metrics configuration
        """
        self.model_module = VQGANModelModule(model_config)
        self.model_config = model_config or {}
        self.training_config = training_config or {}
        self.optimizer_config = optimizer_config or {}
        self.inference_config = inference_config or {}
        self.metrics_config = metrics_config or {}

    def configure(self, binder):
        """configure VQGAN bindings."""
        # Bind VQ Model
        vq_model = self.model_module.provide_vqgan_model()
        binder.bind(VQModel3D, to=lambda: vq_model)
        binder.bind(ModelInterface, to=lambda: vq_model)

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


def create_vqgan_module(
    image_size: int = 64,
    in_channels: int = 1,
    n_embed: int = 1024,
    embed_dim: int = 256,
    latent_channels: int = 256,
    lr: float = 4.5e-6,
    batch_size: int = 1,
    num_train: int = 1000,
    num_val: int = 100,
) -> VQGANModule:
    """
    Factory function to create a 3D VQGAN module with sensible defaults.

    Args:
        image_size: Input volume size
        in_channels: Number of input channels
        n_embed: Number of codebook entries
        embed_dim: Codebook embedding dimension
        latent_channels: Latent space channels
        lr: Learning rate
        batch_size: Batch size
        num_train: Number of training samples
        num_val: Number of validation samples

    Returns:
        Configured VQGANModule
    """
    # Validate parameters
    _validate_param("image_size", image_size, min_val=8)
    _validate_param("in_channels", in_channels, min_val=1)
    _validate_param("n_embed", n_embed, min_val=1)
    _validate_param("embed_dim", embed_dim, min_val=1)
    _validate_param("latent_channels", latent_channels, min_val=1)
    _validate_float_param("lr", lr, min_val=1e-10)
    _validate_param("batch_size", batch_size, min_val=1)
    _validate_param("num_train", num_train, min_val=1)
    _validate_param("num_val", num_val, min_val=1)

    # Derive channel and attention settings compatible with VQModel3D
    ddconfig = get_encoder_decoder_config_3d(
        volume_size=image_size,
        in_channels=in_channels,
        out_channels=in_channels,
        latent_channels=latent_channels,
    )["ddconfig"]

    channel_multipliers = tuple(ddconfig["channel_multipliers"])
    num_res_blocks = ddconfig["num_res_blocks"]
    attn_resolutions = tuple(ddconfig["attn_resolutions"])
    dropout = ddconfig["dropout"]

    # Model config
    model_config = {
        "type": "vqgan3d",
        "params": {
            "in_channels": in_channels,
            "codebook_size": n_embed,
            "embed_dim": embed_dim,
            "latent_channels": latent_channels,
            "resolution": image_size,
            "channel_multipliers": channel_multipliers,
            "num_res_blocks": num_res_blocks,
            "attn_resolutions": attn_resolutions,
            "dropout": dropout,
        },
    }

    # Training config
    training_config = {
        "type": "vqgan",
        "params": {
            "codebook_weight": 1.0,
            "pixel_loss_weight": 1.0,
        },
    }

    optimizer_config = {
        "type": "vqgan",
        "params": {
            "lr_g": lr,
            "lr_d": lr,
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
        },
    }

    return VQGANModule(
        model_config=model_config,
        training_config=training_config,
        optimizer_config=optimizer_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
    )


# =============================================================================
# MAISI VQGAN Modules
# =============================================================================


class MaisiVQModelModule(Module):
    """MAISI VQGAN model configuration module."""

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
    ):
        self.model_config = model_config or {}

    @singleton
    @provider
    def provide_maisi_vq_model(self) -> MaisiVQModel3D:
        """Provide MAISI VQGAN model instance."""
        model_type = self.model_config.get("type", "maisi_vq")
        model_params = self.model_config.get("params", {})

        if model_type == "maisi_vq":
            return MaisiVQModel3D(**model_params)
        raise ValueError(f"Unknown MAISI VQ model type: {model_type}")


class MaisiVQModule(Module):
    """
    Composite module for MAISI VQGAN tasks.

    Provides MAISI VQGAN-specific bindings for model, training, and inference.
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
        self.model_module = MaisiVQModelModule(model_config)
        self.model_config = model_config or {}
        self.training_config = training_config or {}
        self.optimizer_config = optimizer_config or {}
        self.inference_config = inference_config or {}
        self.metrics_config = metrics_config or {}

    def configure(self, binder):
        """Configure MAISI VQGAN bindings."""
        # Bind MAISI VQ Model
        maisi_model = self.model_module.provide_maisi_vq_model()
        binder.bind(MaisiVQModel3D, to=lambda: maisi_model)
        binder.bind(ModelInterface, to=lambda: maisi_model)

        # Training Strategy (reuse VQGAN strategy)
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


def create_maisi_vq_module(
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
) -> MaisiVQModule:
    """
    Factory function to create a MAISI VQGAN module with sensible defaults.

    Args:
        image_size: Input volume size (for reference)
        in_channels: Number of input channels
        codebook_size: Number of codebook entries
        embed_dim: Codebook embedding dimension
        latent_channels: Number of latent channels (encoder output)
        num_channels: Channel numbers for each level
        num_res_blocks: Number of residual blocks per level
        attention_levels: Whether to use attention at each level
        lr: Learning rate
        batch_size: Batch size

    Returns:
        Configured MaisiVQModule
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
        "type": "maisi_vq",
        "params": get_maisi_vq_config(
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

    optimizer_config = {
        "type": "vqgan",
        "params": {
            "lr_g": lr,
            "lr_d": lr,
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
        },
    }

    return MaisiVQModule(
        model_config=model_config,
        training_config=training_config,
        optimizer_config=optimizer_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
    )


# =============================================================================
# MaskGIT Modules
# =============================================================================


class MaskGITModelModule(Module):
    """MaskGIT model configuration module."""

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
    ):
        self.model_config = model_config or {}

    @singleton
    @provider
    def provide_maskgit_model(self) -> MaskGITModelInterface:
        """Provide MaskGIT model instance."""
        model_type = self.model_config.get("type", "maskgit")
        model_params = self.model_config.get("params", {})

        if model_type == "maskgit":
            from maskgit3d.infrastructure.maskgit import MaskGITModel
            from maskgit3d.infrastructure.maskgit.transformer import MaskGITTransformer
            from maskgit3d.infrastructure.vqgan.vqgan_model_3d import VQModel3D

            # Create VQGAN
            vqgan = VQModel3D(
                in_channels=model_params["in_channels"],
                codebook_size=model_params["codebook_size"],
                embed_dim=model_params["embed_dim"],
                latent_channels=model_params["latent_channels"],
                resolution=model_params["resolution"],
                channel_multipliers=model_params["channel_multipliers"],
            )

            # Create Transformer
            transformer = MaskGITTransformer(
                vocab_size=model_params["codebook_size"] + 1,
                hidden_size=model_params["transformer_hidden"],
                num_layers=model_params["transformer_layers"],
                num_heads=model_params["transformer_heads"],
            )

            return MaskGITModel(
                vqgan=vqgan,
                transformer=transformer,
                mask_ratio=model_params.get("mask_ratio", 0.5),
            )
        raise ValueError(f"Unknown MaskGIT model type: {model_type}")


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
        """
        self.model_module = MaskGITModelModule(model_config)
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
            "resolution": image_size,
            "channel_multipliers": (1, 1, 2, 2, 4),
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
    precision: str = "32-true",
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
