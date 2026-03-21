"""MaskGIT training task with automatic optimization."""

from typing import TYPE_CHECKING, Any, cast
import warnings

import torch
from omegaconf import DictConfig

from ..models.maskgit import MaskGIT
from ..models.vqvae import VQVAE
from ..training import MaskGITTrainingSteps
from .base_task import BaseTask

if TYPE_CHECKING:
    from ..runtime.optimizer_factory import TransformerOptimizerFactory


class MaskGITTask(BaseTask):
    """MaskGIT training task with automatic optimization.

    Uses standard automatic optimisation for single-stage training.
    Frozen VQVAE as tokenizer + Trainable Transformer.
    Supports sliding window inference for large images (handled by MaskGIT model).

    Args:
        model: Injected MaskGIT model (if provided, uses injected path)
        vqvae: Injected VQVAE model (if provided, uses injected path)
        training_steps: Injected MaskGITTrainingSteps (if provided, uses injected path)
        optimizer_factory: Injected TransformerOptimizerFactory (if provided, uses injected path)
        vqvae_ckpt_path: Path to pretrained VQVAE checkpoint
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
        maskgit: MaskGIT model containing transformer and generation logic.
        lr: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
        warmup_steps: Number of warmup steps for learning rate scheduler.
    """

    def __init__(
        self,
        model: MaskGIT | None = None,
        vqvae: VQVAE | None = None,
        training_steps: MaskGITTrainingSteps | None = None,
        optimizer_factory: "TransformerOptimizerFactory | None" = None,
        model_config: DictConfig | None = None,
        optimizer_config: DictConfig | None = None,
        vqvae_ckpt_path: str | None = None,
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
            injected_model = cast(MaskGIT, model)
            injected_vqvae = cast(VQVAE, vqvae)
            injected_training_steps = cast(MaskGITTrainingSteps, training_steps)
            injected_optimizer = cast("TransformerOptimizerFactory", optimizer_factory)

            self.vqvae = injected_vqvae
            self.maskgit = injected_model
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
            "Passing legacy scalar/config constructor arguments to MaskGITTask is deprecated; "
            "inject model, vqvae, training_steps, and optimizer_factory instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.optimizer_factory = None

        self.vqvae = self._build_vqvae(vqvae_ckpt_path)
        self.maskgit = self._build_maskgit(
            model_config=model_config,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            gamma_type=gamma_type,
            sliding_window=sliding_window,
        )
        self.training_steps = MaskGITTrainingSteps(maskgit=self.maskgit)
        self.save_hyperparameters(ignore=["vqvae_ckpt_path"])

    def _build_vqvae(self, vqvae_ckpt_path: str | None) -> VQVAE:
        if vqvae_ckpt_path is not None:
            from ..runtime.checkpoints import VQVAECheckpointLoader

            vqvae = VQVAECheckpointLoader().load(vqvae_ckpt_path)
        else:
            vqvae = VQVAE()
        vqvae.eval()
        vqvae.requires_grad_(False)
        return vqvae

    def _build_maskgit(
        self,
        model_config: DictConfig | None,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        gamma_type: str,
        sliding_window: dict[str, Any] | None,
    ) -> MaskGIT:
        if model_config is not None:
            from ..runtime.model_factory import create_maskgit_model

            return create_maskgit_model(model_config, self.vqvae)
        return MaskGIT(
            vqvae=self.vqvae,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            gamma_type=gamma_type,
            sliding_window=sliding_window,
        )

    def encode_images_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to discrete tokens using VQVAE.

        Uses sliding window inference if enabled in MaskGIT model.

        Args:
            x: Input images [B, C, D, H, W]

        Returns:
            Token indices [B, D', H', W']
        """
        return self.maskgit.encode_images_to_tokens(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.maskgit(x)
        return out

    def _extract_input_tensor(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    ) -> torch.Tensor:
        resolved_batch = cast(torch.Tensor | tuple[Any, ...] | list[Any], batch)
        return self.training_steps.extract_input_tensor(resolved_batch)

    def _compute_masked_loss(
        self, tokens: torch.Tensor, mask_ratio: float | None = None
    ) -> tuple[torch.Tensor, dict[str, int | float]]:
        return self.training_steps.compute_masked_loss(tokens, mask_ratio=mask_ratio)

    def training_step(
        self, batch: torch.Tensor | tuple[Any, ...] | list[Any], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        resolved_batch = cast(torch.Tensor | tuple[Any, ...] | list[Any], batch)
        loss, callback_payload = self.training_steps.training_step(
            batch=resolved_batch,
            encode_images_to_tokens_fn=self.encode_images_to_tokens,
        )
        self.save_callback_payload("train", callback_payload)
        return loss

    def validation_step(
        self, batch: torch.Tensor | tuple[Any, ...] | list[Any], batch_idx: int
    ) -> dict[str, Any]:
        del batch_idx
        resolved_batch = cast(torch.Tensor | tuple[Any, ...] | list[Any], batch)
        return self.training_steps.validation_step(
            batch=resolved_batch,
            encode_images_to_tokens_fn=self.encode_images_to_tokens,
        )

    def _decode_tokens_to_latent(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens to latent using VQVAE decoder with sliding window support.

        Args:
            tokens: Token indices [B, D, H, W]

        Returns:
            Decoded latent/images [B, C, D*16, H*16, W*16]
        """
        return self.maskgit.decode_tokens_to_latent(tokens)

    def test_step(
        self, batch: torch.Tensor | tuple[Any, ...] | list[Any], batch_idx: int
    ) -> dict[str, Any]:
        del batch_idx
        resolved_batch = cast(torch.Tensor | tuple[Any, ...] | list[Any], batch)
        return self.training_steps.test_step(
            batch=resolved_batch,
            encode_images_to_tokens_fn=self.encode_images_to_tokens,
        )

    def predict_step(
        self, batch: torch.Tensor | tuple[Any, ...] | list[Any], batch_idx: int
    ) -> dict[str, Any]:
        del batch_idx
        resolved_batch = cast(torch.Tensor | tuple[Any, ...] | list[Any], batch)
        return self.training_steps.predict_step(
            batch=resolved_batch,
            encode_images_to_tokens_fn=self.encode_images_to_tokens,
        )

    def configure_optimizers(self) -> Any:
        total_steps = self._get_total_steps()
        if self.optimizer_factory is not None:
            return self.optimizer_factory.create_optimizer_and_scheduler(
                model=self.maskgit,
                total_steps=total_steps,
            )
        # Legacy fallback (deprecated path)
        from ..runtime.optimizer_factory import create_optimizer
        from ..runtime.scheduler_factory import create_scheduler
        from omegaconf import OmegaConf

        optimizer_config = self.hparams.get("optimizer_config")
        if optimizer_config is not None:
            optimizer = create_optimizer(self.maskgit.parameters(), optimizer_config)
        else:
            optimizer = torch.optim.AdamW(
                self.maskgit.transformer.parameters(),
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
        try:
            if self.trainer is not None:
                return int(self.trainer.estimated_stepping_batches)
        except RuntimeError:
            pass
        return None
