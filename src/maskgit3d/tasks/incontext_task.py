"""InContextMaskGIT training task with automatic optimization."""

import warnings
from typing import TYPE_CHECKING, Any, cast

import torch
from omegaconf import DictConfig

from ..models.incontext import InContextMaskGIT
from ..models.vqvae import VQVAE
from ..training.incontext_steps import (
    InContextEvalStepOutput,
    InContextPredictStepOutput,
    InContextTrainingStepOutput,
    InContextTrainingSteps,
    InContextValidationStepOutput,
)
from .base_task import BaseTask

if TYPE_CHECKING:
    from ..runtime.optimizer_factory import TransformerOptimizerFactory


class InContextMaskGITTask(BaseTask):
    """InContextMaskGIT training task with automatic optimization.

    Uses standard automatic optimization for single-stage training.
    Frozen VQVAE as tokenizer + Trainable InContextMaskGIT Transformer.
    Supports multi-modal in-context learning for 3D medical image generation.

    Args:
        model: Injected InContextMaskGIT model (if provided, uses injected path)
        vqvae: Injected VQVAE model (if provided, uses injected path)
        training_steps: Injected InContextTrainingSteps (if provided, uses injected path)
        optimizer_factory: Injected TransformerOptimizerFactory (if provided, uses injected path)
        model_config: DictConfig for model construction (legacy path)
        optimizer_config: DictConfig for optimizer construction (legacy path)
        vqvae_ckpt_path: Path to pretrained VQVAE checkpoint
        num_modalities: Number of distinct modalities to support (default: 4)
        hidden_size: Transformer hidden dimension (default: 768)
        num_layers: Number of transformer layers (default: 12)
        num_heads: Number of attention heads (default: 12)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        dropout: Dropout probability (default: 0.1)
        gamma_type: Mask scheduling gamma type (default: "cosine")
        lr: Learning rate (default: 2e-4)
        weight_decay: Weight decay (default: 0.05)
        warmup_steps: Learning rate warmup steps (default: 1000)
        sliding_window: Sliding window configuration for encoding/decoding

    Attributes:
        vqvae: Frozen VQVAE model for encoding images to tokens and decoding.
        model: InContextMaskGIT model containing transformer and generation logic.
        lr: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
        warmup_steps: Number of warmup steps for learning rate scheduler.
    """

    optimizer_factory: "TransformerOptimizerFactory | None"

    def __init__(
        self,
        model: InContextMaskGIT | None = None,
        vqvae: VQVAE | None = None,
        training_steps: InContextTrainingSteps | None = None,
        optimizer_factory: "TransformerOptimizerFactory | None" = None,
        model_config: DictConfig | None = None,
        optimizer_config: DictConfig | None = None,
        vqvae_ckpt_path: str | None = None,
        num_modalities: int = 4,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        gamma_type: str = "cosine",
        lr: float = 2e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
        sliding_window: dict[str, Any] | None = None,
    ):
        super().__init__()

        uses_injected_components = all(
            component is not None for component in (model, vqvae, training_steps, optimizer_factory)
        )

        if uses_injected_components:
            injected_model = cast(InContextMaskGIT, model)
            injected_vqvae = cast(VQVAE, vqvae)
            injected_training_steps = cast(InContextTrainingSteps, training_steps)
            injected_optimizer = cast("TransformerOptimizerFactory", optimizer_factory)

            self.vqvae = injected_vqvae
            self.incontext_model = injected_model
            self.training_steps = injected_training_steps
            self.lr = injected_optimizer.lr
            self.weight_decay = injected_optimizer.weight_decay
            self.warmup_steps = injected_optimizer.warmup_steps
            self.optimizer_factory = injected_optimizer
            self.save_hyperparameters(
                ignore=["model", "vqvae", "training_steps", "optimizer_factory"]
            )
            return

        warnings.warn(
            "Passing legacy scalar/config constructor arguments to InContextMaskGITTask is deprecated; "
            "inject model, vqvae, training_steps, and optimizer_factory instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.optimizer_factory = None

        self.vqvae = self._build_vqvae(vqvae_ckpt_path)
        self.incontext_model = self._build_incontext_model(
            model_config=model_config,
            num_modalities=num_modalities,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            gamma_type=gamma_type,
            sliding_window=sliding_window,
        )
        self.training_steps = InContextTrainingSteps(model=self.incontext_model)
        self.save_hyperparameters(ignore=["vqvae_ckpt_path"])

    def _build_vqvae(self, vqvae_ckpt_path: str | None) -> VQVAE:
        """Build and freeze VQVAE from checkpoint or create default.

        Args:
            vqvae_ckpt_path: Path to pretrained VQVAE checkpoint.

        Returns:
            Frozen VQVAE model in eval mode.
        """
        if vqvae_ckpt_path is None:
            vqvae = VQVAE()
        else:
            from ..runtime.checkpoints import VQVAECheckpointLoader

            vqvae = VQVAECheckpointLoader().load(vqvae_ckpt_path)
        vqvae.eval()
        vqvae.requires_grad_(False)
        return vqvae

    def _build_incontext_model(
        self,
        model_config: DictConfig | None,
        num_modalities: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        gamma_type: str,
        sliding_window: dict[str, Any] | None,
    ) -> InContextMaskGIT:
        """Build InContextMaskGIT model from config or scalar params.

        Args:
            model_config: Optional DictConfig for model instantiation.
            num_modalities: Number of modalities.
            hidden_size: Transformer hidden dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            mlp_ratio: MLP expansion ratio.
            dropout: Dropout probability.
            gamma_type: Mask scheduling type.
            sliding_window: Sliding window configuration.

        Returns:
            InContextMaskGIT model instance.
        """
        if model_config is not None:
            from hydra.utils import instantiate

            return instantiate(
                model_config,
                vqvae=self.vqvae,
                num_modalities=num_modalities,
            )
        return InContextMaskGIT(
            vqvae=self.vqvae,
            num_modalities=num_modalities,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            gamma_type=gamma_type,
            sliding_window_cfg=sliding_window,
        )

    def forward(
        self,
        context_images: list[torch.Tensor],
        context_modality_ids: list[int],
        target_image: torch.Tensor,
        target_modality_id: int,
    ) -> torch.Tensor:
        """Forward pass for inference (reconstruction).

        Args:
            context_images: List of context image tensors.
            context_modality_ids: Modality IDs for each context image.
            target_image: Target image tensor.
            target_modality_id: Modality ID for target image.

        Returns:
            Reconstructed target image.
        """
        return self.incontext_model(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
        )

    def _encode_context(
        self, context_images: list[torch.Tensor], context_modality_ids: list[int]
    ) -> list[torch.Tensor]:
        """Pass through context images (encoding done in compute_loss)."""
        return context_images

    def _encode_target(self, target_image: torch.Tensor, target_modality_id: int) -> torch.Tensor:
        """Pass through target image (encoding done in compute_loss)."""
        return target_image

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> InContextTrainingStepOutput:
        """Training step delegates to training_steps.

        Args:
            batch: Dict containing context_images, context_modality_ids,
                   target_image, target_modality_id.
            batch_idx: Batch index (unused).

        Returns:
            Dict with 'loss' tensor.
        """
        del batch_idx
        resolved_batch = cast(dict[str, Any], batch)
        return self.training_steps.training_step(
            batch=resolved_batch,
            encode_context_fn=self._encode_context,
            encode_target_fn=self._encode_target,
        )

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> InContextValidationStepOutput:
        """Validation step delegates to training_steps.

        Args:
            batch: Dict containing context_images, context_modality_ids,
                   target_image, target_modality_id.
            batch_idx: Batch index (unused).

        Returns:
            Dict with context_images, target_image, generated_image, loss.
        """
        del batch_idx
        resolved_batch = cast(dict[str, Any], batch)
        return self.training_steps.validation_step(
            batch=resolved_batch,
            encode_context_fn=self._encode_context,
            encode_target_fn=self._encode_target,
        )

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> InContextEvalStepOutput:
        """Test step delegates to training_steps (same as validation).

        Args:
            batch: Dict containing context_images, context_modality_ids,
                   target_image, target_modality_id.
            batch_idx: Batch index (unused).

        Returns:
            Dict with context_images, target_image, generated_image, loss.
        """
        del batch_idx
        resolved_batch = cast(dict[str, Any], batch)
        return self.training_steps.validation_step(
            batch=resolved_batch,
            encode_context_fn=self._encode_context,
            encode_target_fn=self._encode_target,
        )

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> InContextPredictStepOutput:
        """Predict step delegates to training_steps.

        Args:
            batch: Dict containing context_images, context_modality_ids,
                   target_image, target_modality_id.
            batch_idx: Batch index (unused).

        Returns:
            Dict with context_images, target_image, generated_image.
        """
        del batch_idx
        resolved_batch = cast(dict[str, Any], batch)
        return self.training_steps.predict_step(
            batch=resolved_batch,
            encode_context_fn=self._encode_context,
            encode_target_fn=self._encode_target,
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW optimizer with warmup scheduler.

        Returns:
            Dict with 'optimizer' and 'lr_scheduler'.
        """
        total_steps = self._get_total_steps()
        if self.optimizer_factory is not None:
            return self.optimizer_factory.create_optimizer_and_scheduler(
                model=self.incontext_model,
                total_steps=total_steps,
            )
        # Legacy fallback (deprecated path)
        from omegaconf import OmegaConf

        from ..runtime.optimizer_factory import create_optimizer
        from ..runtime.scheduler_factory import create_scheduler

        optimizer_config = self.hparams.get("optimizer_config")
        if optimizer_config is not None:
            cfg = OmegaConf.create(OmegaConf.to_container(optimizer_config, resolve=True))
            cfg.lr = self.lr
            cfg.weight_decay = self.weight_decay
            optimizer = create_optimizer(self.incontext_model.parameters(), cast(DictConfig, cfg))
        else:
            optimizer = torch.optim.AdamW(
                self.incontext_model.transformer.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        scheduler_config = OmegaConf.create(
            {"warmup_steps": self.warmup_steps, "total_steps": total_steps}
        )
        scheduler = create_scheduler(optimizer, scheduler_config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _get_total_steps(self) -> int | None:
        """Get total training steps from trainer.

        Returns:
            Total steps or None if trainer not available.
        """
        try:
            if self.trainer is not None:
                return int(self.trainer.estimated_stepping_batches)
        except RuntimeError:
            pass
        return None
