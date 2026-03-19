"""MaskGIT training task with automatic optimization."""

from typing import Any, cast

import torch
from omegaconf import DictConfig

from ..models.maskgit import MaskGIT
from ..models.vqvae import VQVAE
from ..training import MaskGITTrainingSteps
from .base_task import BaseTask


class MaskGITTask(BaseTask):
    """MaskGIT training task with automatic optimization.

    Uses standard automatic optimisation for single-stage training.
    Frozen VQVAE as tokenizer + Trainable Transformer.
    Supports sliding window inference for large images (handled by MaskGIT model).

    Args:
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
        self.save_hyperparameters(ignore=["vqvae_ckpt_path"])

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

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
        return self.training_steps.create_optimizers(
            optimizer_config=self.hparams.get("optimizer_config"),
            lr=self.lr,
            weight_decay=self.weight_decay,
            warmup_steps=self.hparams.get("warmup_steps", self.warmup_steps),
        )
