"""
Training strategy interfaces for MaskGIT models.

This module contains framework-specific (architecture-specific) training
strategy interfaces that belong in the infrastructure layer.
"""

from abc import abstractmethod

import torch

from maskgit3d.domain.interfaces import MaskGITModelInterface, TrainingStrategy


class MaskGITTrainingStrategyInterface(TrainingStrategy):
    """
    Interface for MaskGIT training strategies.

    Uses BERT-style masked prediction training where a portion of
    tokens are masked and the Transformer learns to predict them.
    """

    @abstractmethod
    def train_step(
        self,
        model: MaskGITModelInterface,
        batch: tuple[torch.Tensor, ...],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """
        Execute one training step with masked token prediction.

        Args:
            model: The MaskGIT model being trained
            batch: Tuple of (input_images,)
            optimizer: Optimizer for parameter updates

        Returns:
            Dictionary of training metrics including loss
        """
        pass
