"""MaskGIT training task with automatic optimization."""

from typing import Optional

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from ..models.maskgit import MaskGIT
from ..models.vqvae import VQVAE


class MaskGITTask(LightningModule):
    """MaskGIT training task with automatic optimization.

    Uses standard automatic optimization for single-stage training.
    Frozen VQVAE as tokenizer + Trainable Transformer.

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
    """

    def __init__(
        self,
        vqvae_ckpt_path: str,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        gamma_type: str = "cosine",
        lr: float = 2e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        # Load pretrained VQVAE
        vqvae_state = torch.load(vqvae_ckpt_path, map_location="cpu", weights_only=True)
        if "state_dict" in vqvae_state:
            vqvae_state = vqvae_state["state_dict"]

        self.vqvae = VQVAE()
        self.vqvae.load_state_dict(vqvae_state)
        self.vqvae.eval()
        self.vqvae.requires_grad_(False)

        # Build MaskGIT model
        self.maskgit = MaskGIT(
            vqvae=self.vqvae,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            gamma_type=gamma_type,
        )

        # Track training step for warmup
        self.training_step_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maskgit(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch if batch.dim() == 5 else batch[0]

        loss, metrics = self.maskgit.compute_loss(x)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/mask_acc", metrics["mask_acc"], prog_bar=True)
        self.log("train/mask_ratio", metrics["mask_ratio"], prog_bar=False)

        self.training_step_count += 1

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x = batch if batch.dim() == 5 else batch[0]

        loss, metrics = self.maskgit.compute_loss(x)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/mask_acc", metrics["mask_acc"], prog_bar=True)

        # Generate a sample every N batches
        if batch_idx == 0:
            with torch.no_grad():
                sample = self.maskgit.generate(shape=(1, 4, 4, 4), num_iterations=12)
                self.log("val/sample_shape", sample.shape[0])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.maskgit.transformer.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",
        #     },
        # }
