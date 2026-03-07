"""Mask scheduling utilities for MaskGIT training and inference.

This module provides mask ratio scheduling following the MaskGIT paper:
- Training: Random mask ratio sampled from gamma distribution (cosine, linear, square, cubic)
- Inference: Progressive token revealing with cosine/linear/sqrt schedules
"""

import math
from collections.abc import Callable

import numpy as np
import torch


class TrainingMaskScheduler:
    """Scheduler for computing mask ratios during training.

    Following the MaskGIT paper, during training the mask ratio is NOT fixed.
    Instead, it is randomly sampled from a gamma distribution for each batch.

    The gamma functions available are:
    - cosine: γ(r) = cos(r * π/2) - smooth decay
    - linear: γ(r) = 1 - r - linear decay
    - square: γ(r) = 1 - r² - quadratic decay
    - cubic: γ(r) = 1 - r³ - cubic decay

    For each batch:
    1. Sample u ~ Uniform(0, 1)
    2. Compute mask_ratio = gamma(u)
    3. Number of masked tokens = floor(mask_ratio * total_tokens)

    This provides curriculum learning where the model sees varying difficulty
    levels during training, improving generalization.
    """

    def __init__(
        self,
        gamma_type: str = "cosine",
        choice_temperature: float = 4.5,
    ):
        """Initialize the training mask scheduler.

        Args:
            gamma_type: Type of gamma function ("cosine", "linear", "square", "cubic")
            choice_temperature: Temperature for Gumbel noise during inference (not used in training)
        """
        self.gamma_type = gamma_type
        self.choice_temperature = choice_temperature

        # Select gamma function
        self.gamma = self._get_gamma_function(gamma_type)

    def _get_gamma_function(self, gamma_type: str) -> Callable[[float], float]:
        """Get the gamma function for mask ratio scheduling.

        Args:
            gamma_type: Type of gamma function

        Returns:
            Gamma function that maps [0, 1] -> [1, 0]
        """
        if gamma_type == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif gamma_type == "linear":
            return lambda r: 1 - r
        elif gamma_type == "square":
            return lambda r: 1 - r**2
        elif gamma_type == "cubic":
            return lambda r: 1 - r**3
        else:
            raise ValueError(
                f"Unknown gamma type: {gamma_type}. Supported: cosine, linear, square, cubic"
            )

    def sample_mask_ratio(self) -> float:
        """Sample a random mask ratio for training.

        Returns:
            Mask ratio in [0, 1] sampled from gamma distribution
        """
        # Sample uniform random value
        u = np.random.uniform(0, 1)

        # Apply gamma function
        mask_ratio = self.gamma(u)

        return float(mask_ratio)

    def compute_num_masked(
        self,
        total_tokens: int,
        mask_ratio: float | None = None,
    ) -> int:
        """Compute the number of tokens to mask.

        Args:
            total_tokens: Total number of tokens
            mask_ratio: Optional mask ratio (if None, samples a new one)

        Returns:
            Number of tokens to mask
        """
        if mask_ratio is None:
            mask_ratio = self.sample_mask_ratio()

        num_masked = int(math.floor(mask_ratio * total_tokens))

        # Ensure at least 1 token is masked
        num_masked = max(1, num_masked)

        # Ensure not all tokens are masked
        num_masked = min(num_masked, total_tokens - 1)

        return num_masked

    def get_inference_schedule(
        self,
        num_iterations: int,
        mode: str = "cosine",
    ) -> torch.Tensor:
        """Generate mask schedule for iterative inference decoding.

        This is used during inference (not training) to determine
        how many tokens to reveal at each iteration.

        Args:
            num_iterations: Number of decoding iterations
            mode: Schedule type ("cosine", "linear", "sqrt")

        Returns:
            Tensor of reveal ratios for each iteration
        """
        if mode == "cosine":
            steps = torch.arange(num_iterations + 1, dtype=torch.float32)
            # Cumulative percentage of tokens to reveal
            schedule = (1 - torch.cos(steps * np.pi / num_iterations)) / 2
            # Convert to per-iteration reveal ratios
            reveal_ratios = schedule[1:] - schedule[:-1]
        elif mode == "linear":
            reveal_ratios = torch.ones(num_iterations, dtype=torch.float32) / num_iterations
        elif mode == "sqrt":
            steps = torch.arange(1, num_iterations + 1, dtype=torch.float32)
            reveal_ratios = 1 / torch.sqrt(steps)
            reveal_ratios = reveal_ratios / reveal_ratios.sum()
        else:
            raise ValueError(
                f"Unknown inference schedule mode: {mode}. Supported: cosine, linear, sqrt"
            )

        return reveal_ratios

    def __repr__(self) -> str:
        return f"TrainingMaskScheduler(gamma_type={self.gamma_type!r})"


def mask_by_random_topk(
    topk: int,
    confidence: torch.Tensor,
    temperature: float = 4.5,
) -> torch.Tensor:
    """Select tokens to mask using Gumbel-max sampling.

    Used during inference to determine which tokens to keep masked
    based on model confidence.

    Args:
        topk: Number of tokens to keep masked (lower confidence = more likely to stay masked)
        confidence: Confidence scores for each token [B, N]
        temperature: Temperature for Gumbel noise

    Returns:
        Boolean mask where True indicates tokens to keep masked
    """
    B, N = confidence.shape

    # Add Gumbel noise for randomness
    uniform_noise = torch.rand_like(confidence)
    uniform_noise = uniform_noise.clamp(1e-6, 1 - 1e-6)
    gumbel_noise = -torch.log(-torch.log(uniform_noise))

    # Adjust confidence with temperature-scaled Gumbel noise
    # Higher temperature = more random selection
    scores = confidence + temperature * gumbel_noise

    # Select bottom-k tokens (lowest scores) to keep masked
    _, indices = torch.topk(scores, k=topk, largest=False, dim=-1)

    # Create mask
    mask = torch.zeros(B, N, dtype=torch.bool, device=confidence.device)
    mask.scatter_(1, indices, True)

    return mask
