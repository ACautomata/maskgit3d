"""
Configuration layer - Dependency injection modules.

This module provides Injector modules that bind interfaces
to concrete implementations for the application.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from injector import Module, provider, singleton

from maskgit3d.domain.interfaces import (
    DataProvider,
    InferenceStrategy,
    LRSchedulerFactory,
    MaskGITModelInterface,
    Metrics,
    ModelInterface,
    OptimizerFactory,
    TrainingStrategy,
)
from maskgit3d.infrastructure.training.strategies import (
    AdamOptimizerFactory,
    AdamWOptimizerFactory,
    SGDOptimizerFactory,
    VQGANTrainingStrategy,
    VQGANInference,
    VQGANOptimizerFactory,
    VQGANMetrics,
    MaskGITTrainingStrategy,
    MaskGITInference,
)
from maskgit3d.infrastructure.vqgan import (
    VQModel3D,
    NLayerDiscriminator,
    VectorQuantizer2,
    get_encoder_decoder_config_3d,
)


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
        model_config: Optional[Dict[str, Any]] = None,
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
        # Default to MaskGIT model
        model_type = self.model_config.get("type", "maskgit")
        model_params = self.model_config.get("params", {})

        if model_type == "maskgit":
            from maskgit3d.infrastructure.maskgit import MaskGITModel
            return MaskGITModel(**model_params)
        raise ValueError(f"Unknown model type: {model_type}")


class DataModule(Module):
    """
    Module for data configuration.

    Provides data providers for training, validation, and testing.
    """

    def __init__(
        self,
        data_config: Optional[Dict[str, Any]] = None,
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
        # Default to simple data provider
        from maskgit3d.infrastructure.data.dataset import SimpleDataProvider
        data_type = self.data_config.get("type", "simple")
        data_params = self.data_config.get("params", {})

        providers = {
            "simple": SimpleDataProvider,
        }

        if data_type not in providers:
            raise ValueError(
                f"Unknown data type: {data_type}. "
                f"Available: {list(providers.keys())}"
            )

        return providers[data_type](**data_params)


class TrainingModule(Module):
    """
    Module for training configuration.

    Provides training strategies, optimizers, and schedulers.
    """

    def __init__(
        self,
        training_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
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
                f"Unknown strategy type: {strategy_type}. "
                f"Available: {list(strings.keys())}"
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
                f"Unknown optimizer type: {optimizer_type}. "
                f"Available: {list(optimizers.keys())}"
            )

        return optimizers[optimizer_type](**optimizer_params)


class InferenceModule(Module):
    """
    Module for inference configuration.

    Provides inference strategies and metrics.
    """

    def __init__(
        self,
        inference_config: Optional[Dict[str, Any]] = None,
        metrics_config: Optional[Dict[str, Any]] = None,
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
                f"Unknown inference type: {inference_type}. "
                f"Available: {list(strategies.keys())}"
            )

        return strategies[inference_type](**inference_params)

    @singleton
    @provider
    def provide_metrics(self) -> Optional[Metrics]:
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
                f"Unknown metrics type: {metrics_type}. "
                f"Available: {list(metrics.keys())}"
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
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )


# =============================================================================
# VQGAN Modules (3D)
# =============================================================================


class VQGANModelModule(Module):
    """VQGAN 3D model configuration module."""

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
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
        model_config: Optional[Dict[str, Any]] = None,
        data_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        inference_config: Optional[Dict[str, Any]] = None,
        metrics_config: Optional[Dict[str, Any]] = None,
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
        strategy_type = self.training_config.get("type", "vqgan")
        strategy_params = self.training_config.get("params", {})
        binder.bind(TrainingStrategy, to=lambda: VQGANTrainingStrategy(**strategy_params))

        # Optimizer Factory
        binder.bind(VQGANOptimizerFactory, to=lambda: VQGANOptimizerFactory(**self.optimizer_config.get("params", {})))

        # Inference Strategy
        inf_type = self.inference_config.get("type", "vqgan")
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
    # Get 3D encoder/decoder config
    ddconfig = get_encoder_decoder_config_3d(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=in_channels,
        latent_channels=latent_channels,
    )["ddconfig"]

    # Model config
    model_config = {
        "type": "vqgan3d",
        "params": {
            "ddconfig": ddconfig,
            "n_embed": n_embed,
            "embed_dim": embed_dim,
        },
    }

    # Training config
    training_config = {
        "type": "vqgan",
        "params": {
            "codebook_weight": 1.0,
            "pixel_loss_weight": 1.0,
            "perceptual_weight": 1.0,
            "disc_weight": 1.0,
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

    return VQGANModule(
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
        model_config: Optional[Dict[str, Any]] = None,
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
            return MaskGITModel(**model_params)
        raise ValueError(f"Unknown MaskGIT model type: {model_type}")


class MaskGITModule(Module):
    """
    Composite module for MaskGIT tasks.

    Provides MaskGIT-specific bindings for model, training, and inference.
    """

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        data_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        inference_config: Optional[Dict[str, Any]] = None,
        metrics_config: Optional[Dict[str, Any]] = None,
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
        strategy_type = self.training_config.get("type", "maskgit")
        strategy_params = self.training_config.get("params", {})
        binder.bind(TrainingStrategy, to=lambda: MaskGITTrainingStrategy(**strategy_params))

        # Optimizer Factory
        optimizer_type = self.optimizer_config.get("type", "adam")
        optimizer_params = self.optimizer_config.get("params", {})
        binder.bind(OptimizerFactory, to=lambda: AdamOptimizerFactory(**optimizer_params))

        # Inference Strategy
        inf_type = self.inference_config.get("type", "maskgit")
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
