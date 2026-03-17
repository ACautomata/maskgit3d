"""MaskGIT training task with automatic optimization."""

from typing import Any

import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning import LightningModule
from omegaconf import DictConfig

from ..models.maskgit import MaskGIT
from ..models.vqvae import VQVAE


class MaskGITTask(LightningModule):
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

        self.vqvae = VQVAE()
        if vqvae_ckpt_path is not None:
            vqvae_state = torch.load(vqvae_ckpt_path, map_location="cpu", weights_only=True)
            if "state_dict" in vqvae_state:
                vqvae_state = vqvae_state["state_dict"]
            self.vqvae.load_state_dict(vqvae_state)
        self.vqvae.eval()
        self.vqvae.requires_grad_(False)

        if model_config is not None:
            self.maskgit = instantiate(model_config, vqvae=self.vqvae)
        else:
            self.maskgit = MaskGIT(
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
        if isinstance(batch, tuple | list):
            return batch[0]
        return batch

    def _compute_masked_loss(
        self, tokens: torch.Tensor, mask_ratio: float | None = None
    ) -> tuple[torch.Tensor, dict[str, int | float]]:
        """Compute masked prediction loss and return raw data for metrics.

        Args:
            tokens: Token indices [B, D, H, W]
            mask_ratio: Optional mask ratio (sampled if None)

        Returns:
            Tuple of (loss tensor, raw data dict for metrics computation)
        """
        B, D, H, W = tokens.shape
        tokens_flat = tokens.view(B, -1)

        effective_mask_ratio = (
            self.maskgit.mask_scheduler.sample_mask_ratio() if mask_ratio is None else mask_ratio
        )

        # Use transformer's predict_masked for masking logic
        masked_logits, masked_targets, mask = self.maskgit.transformer.predict_masked(
            tokens_flat, mask_ratio=effective_mask_ratio
        )

        loss = nn.functional.cross_entropy(masked_logits, masked_targets)

        # Compute accuracy in-step to avoid storing large tensors
        with torch.no_grad():
            predictions = masked_logits.argmax(dim=-1)
            correct = (predictions == masked_targets).sum().item()
            total = masked_targets.numel()

        # Return scalars for metrics (memory efficient)
        raw_data: dict[str, int | float] = {
            "correct": correct,
            "total": total,
            "mask_ratio": float(effective_mask_ratio),
        }
        return loss, raw_data

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, Any]:
        x = self._extract_input_tensor(batch)

        # Encode images to tokens using sliding window if enabled
        tokens = self.encode_images_to_tokens(x)

        # Compute loss and get raw data for metrics
        loss, raw_data = self._compute_masked_loss(tokens)

        # Return loss and raw data for callback processing
        return {
            "loss": loss,
            "log_data": raw_data,
        }

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x = self._extract_input_tensor(batch)

        # Encode images to tokens using sliding window if enabled
        tokens = self.encode_images_to_tokens(x)

        # Compute loss and get raw data for metrics
        loss, raw_data = self._compute_masked_loss(tokens)

        self.log(
            "val_loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
        )

        total = raw_data["total"]
        if total > 0:
            self.log(
                "val_mask_acc",
                raw_data["correct"] / total,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=x.shape[0],
            )

        self.log(
            "val_mask_ratio",
            raw_data["mask_ratio"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=x.shape[0],
        )

    def _decode_tokens_to_latent(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens to latent using VQVAE decoder with sliding window support.

        Args:
            tokens: Token indices [B, D, H, W]

        Returns:
            Decoded latent/images [B, C, D*16, H*16, W*16]
        """
        return self.maskgit.decode_tokens_to_latent(tokens)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, Any]:
        x = self._extract_input_tensor(batch)

        with torch.no_grad():
            tokens_shape = self.encode_images_to_tokens(x).shape

        with torch.no_grad():
            generated_images = self.maskgit.generate(
                shape=tokens_shape,
                temperature=1.0,
                num_iterations=12,
            )

        return {
            "generated_latent": generated_images,
            "input_shape": x.shape,
            "token_shape": tokens_shape,
        }

    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> dict[str, Any]:
        if self.hparams.get("optimizer_config") is not None:
            optimizer = instantiate(
                self.hparams["optimizer_config"],
                params=self.maskgit.parameters(),
            )
        else:
            optimizer = torch.optim.AdamW(
                self.maskgit.transformer.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        warmup_steps = self.hparams.get("warmup_steps", self.warmup_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
