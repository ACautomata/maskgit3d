"""
Domain Layer - Core interfaces for the deep learning framework.

This module defines the abstraction layer that decouples business logic
from concrete implementations (PyTorch, MONAI, etc.).
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import DataLoader

# =============================================================================
# Data Interfaces
# =============================================================================


class DataProvider(ABC):
    """
    Data provider interface for training, validation, and testing.

    Implementations should handle data loading, preprocessing, and batching
    for the specific task (segmentation, classification, etc.).
    """

    @abstractmethod
    def train_loader(self) -> DataLoader:
        """
        Returns a DataLoader for training batches.

        Returns:
            DataLoader yielding tuples (input, target, ...). The first element
            should be the input tensor, second should be the target/label.
        """
        pass

    @abstractmethod
    def val_loader(self) -> DataLoader:
        """
        Returns a DataLoader for validation batches.

        Returns:
            DataLoader yielding tuples (input, target, ...).
        """
        pass

    @abstractmethod
    def test_loader(self) -> DataLoader:
        """
        Returns a DataLoader for test batches.

        Returns:
            DataLoader yielding tuples (input, target, ...). Target may be
            None for inference-only scenarios.
        """
        pass


# =============================================================================
# Model Interfaces
# =============================================================================


class ModelInterface(torch.nn.Module, ABC):
    """
    Base interface for all models.

    This interface wraps PyTorch nn.Module and provides additional methods
    for checkpoint management and configuration.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor (logits, segmentation map, etc.)
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint to disk.

        Args:
            path: Path to save the checkpoint
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint from disk.

        Args:
            path: Path to the checkpoint file
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Get the device of the model."""
        pass


# =============================================================================
# Training Interfaces
# =============================================================================


class TrainingStrategy(ABC):
    """
    Interface for training strategies.

    Encapsulates the logic for a single training step, including
    loss computation, backward pass, and optimizer step.
    """

    @abstractmethod
    def train_step(
        self,
        model: ModelInterface,
        batch: tuple[torch.Tensor, ...],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """
        Execute one training step.

        Args:
            model: The model being trained
            batch: Input batch (input, target, ...)
            optimizer: Optimizer for parameter updates

        Returns:
            Dictionary of metrics (must include 'loss' key)
        """
        pass

    @abstractmethod
    def validate_step(
        self,
        model: ModelInterface,
        batch: tuple[torch.Tensor, ...],
    ) -> dict[str, float]:
        """
        Execute one validation step.

        Args:
            model: The model being validated
            batch: Input batch (input, target, ...)

        Returns:
            Dictionary of validation metrics
        """
        pass


class OptimizerFactory(ABC):
    """
    Interface for optimizer creation.

    Allows flexible configuration of optimizer parameters without
    coupling to specific model implementations.
    """

    @abstractmethod
    def create(
        self,
        model_params: Iterator[torch.Tensor],
    ) -> torch.optim.Optimizer:
        """
        Create an optimizer instance.

        Args:
            model_params: Iterator of model parameters

        Returns:
            Configured optimizer
        """
        pass


class GANOptimizerFactory(ABC):
    """
    Interface for GAN-style optimizer creation with separate G/D optimizers.

    Used for VQGAN which requires separate optimizers for generator and discriminator.
    """

    @abstractmethod
    def create(
        self,
        gen_params: Iterator[torch.Tensor],
        disc_params: Iterator[torch.Tensor] | None = None,
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer | None]:
        """
        Create optimizer instances for generator and discriminator.

        Args:
            gen_params: Iterator of generator parameters
            disc_params: Iterator of discriminator parameters (optional)

        Returns:
            Tuple of (generator_optimizer, discriminator_optimizer or None)
        """
        pass


# =============================================================================
# Inference Interfaces
# =============================================================================


class InferenceStrategy(ABC):
    """
    Interface for inference strategies.

    Handles model prediction and post-processing for deployment.
    """

    @abstractmethod
    def predict(
        self,
        model: ModelInterface,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run inference on a batch.

        Args:
            model: The model for inference
            batch: Input batch

        Returns:
            Raw predictions (logits, probabilities, etc.)
        """
        pass

    @abstractmethod
    def post_process(
        self,
        predictions: torch.Tensor,
    ) -> Any:
        """
        Post-process raw predictions.

        Args:
            predictions: Raw model outputs

        Returns:
            Processed predictions (class indices, masks, bboxes, etc.)
        """
        pass


# =============================================================================
# Metrics Interfaces
# =============================================================================


class Metrics(ABC):
    """
    Interface for metrics computation.

    Supports both training metrics (loss) and evaluation metrics
    (accuracy, dice score, FID, etc.).
    """

    @abstractmethod
    def update(
        self,
        predictions: Any,
        targets: Any,
    ) -> None:
        """
        Update metrics with new batch predictions.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
        """
        pass

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """
        Compute final metrics.

        Returns:
            Dictionary of metric names and values
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset accumulated metrics for next epoch/test."""
        pass


# =============================================================================
# VQGAN Interfaces
# =============================================================================


class QuantizerInterface(ABC):
    """
    Interface for vector quantization components.

    VQGAN uses discrete latent codes from a learned codebook.
    """

    @abstractmethod
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Quantize continuous latents to discrete codes.

        Args:
            z: Continuous latent from encoder [B, C, H, W]

        Returns:
            Tuple of (quantized_z, commitment_loss, info)
            - quantized_z: Quantized latent [B, C, H, W]
            - commitment_loss: Codebook commitment loss
            - info: Tuple of (perplexity, encodings, indices)
        """
        pass

    @abstractmethod
    def get_codebook_entry(
        self,
        indices: torch.Tensor,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """
        Retrieve quantized latents from codebook indices.

        Args:
            indices: Codebook indices [B, H, W] or [B*H*W]
            shape: Optional shape hint

        Returns:
            Quantized latents
        """
        pass


class DiscriminatorInterface(ABC):
    """
    Interface for GAN discriminator.

    VQGAN uses a PatchGAN-style discriminator for adversarial training.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Discrimination logits [B, 1, H', W']
        """
        pass


class VQModelInterface(ModelInterface):
    """
    Interface for VQGAN/VQVAE models.

    Extends ModelInterface with VQ-specific methods for encoding,
    decoding, and codebook manipulation.
    """

    @abstractmethod
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Encode images to discrete latent codes.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Tuple of (quantized, commitment_loss, info)
        """
        pass

    @abstractmethod
    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latents to images.

        Args:
            quant: Quantized latents [B, C, H, W]

        Returns:
            Reconstructed images [B, C, H, W]
        """
        pass

    @abstractmethod
    def decode_code(self, code: torch.Tensor) -> torch.Tensor:
        """
        Decode from codebook indices directly.

        Args:
            code: Codebook indices [B, H, W]

        Returns:
            Reconstructed images [B, C, H', W']
        """
        pass

    @property
    @abstractmethod
    def codebook_size(self) -> int:
        """Get the number of codes in the codebook."""
        pass

    @property
    @abstractmethod
    def latent_shape(self) -> tuple[int, int, int, int]:
        """Get the shape of latent representations (C, D, H, W)."""
        pass

    @abstractmethod
    def forward_with_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning reconstruction and quantization loss.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstructed, quantization_loss)
        """
        pass


# =============================================================================
# MaskGIT Interfaces
# =============================================================================


class MaskGITModelInterface(ModelInterface):
    """
    Interface for MaskGIT models.

    MaskGIT combines a VQGAN tokenizer with a bidirectional Transformer
    for masked token prediction during training and iterative decoding
    during inference.
    """

    @abstractmethod
    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to discrete latent tokens.

        Args:
            x: Input images [B, C, D, H, W] or [B, C, H, W]

        Returns:
            Token indices [B, D, H, W] or [B, H, W]
        """
        pass

    @abstractmethod
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens to images.

        Args:
            tokens: Token indices [B, D, H, W] or [B, H, W]

        Returns:
            Reconstructed images [B, C, D, H, W] or [B, C, H, W]
        """
        pass

    @abstractmethod
    def generate(
        self,
        shape: tuple[int, ...] | None = None,
        temperature: float = 1.0,
        num_iterations: int = 12,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate images from random tokens using iterative decoding.

        Args:
            shape: Shape of token grid (B, D, H, W)
            temperature: Sampling temperature
            num_iterations: Number of decoding iterations

        Returns:
            Generated images
        """
        pass

    @property
    @abstractmethod
    def num_tokens(self) -> int:
        """Get the total number of tokens (codebook size + 1 for mask)."""
        pass

    @property
    @abstractmethod
    def latent_shape(self) -> tuple[int, int, int, int]:
        """Get the shape of latent representations (B, D, H, W)."""
        pass

    @property
    @abstractmethod
    def codebook_size(self) -> int:
        """Get the number of codes in the codebook."""
        pass
