"""MaskGIT training task with automatic optimization."""

from typing import Any

import torch
import torch.nn as nn
from lightning import LightningModule
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms.croppad.array import DivisiblePad

from ..models.maskgit import MaskGIT
from ..models.vqvae import VQVAE


class MaskGITTask(LightningModule):
    """MaskGIT training task with automatic optimization.

    Uses standard automatic optimization for single-stage training.
    Frozen VQVAE as tokenizer + Trainable Transformer.
    Supports sliding window inference for large images.

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
        sliding_window: Sliding window configuration for VQVAE encoding
    """

    def __init__(
        self,
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

        # Sliding window configuration
        self.sliding_window_cfg = sliding_window or {}
        self._sliding_window_inferer: SlidingWindowInferer | None = None
        self._divisible_pad: DivisiblePad | None = None
        self._downsampling_factor = 16

        # Track training step for warmup
        self.training_step_count = 0

    def _get_sliding_window_inferer(self) -> SlidingWindowInferer | None:
        """Get or create sliding window inferer for VQVAE encoding."""
        if not self.sliding_window_cfg.get("enabled", False):
            return None
        if self._sliding_window_inferer is None:
            roi_size = tuple(self.sliding_window_cfg.get("roi_size", [64, 64, 64]))
            overlap = self.sliding_window_cfg.get("overlap", 0.25)
            mode = self.sliding_window_cfg.get("mode", "gaussian")
            sigma_scale = self.sliding_window_cfg.get("sigma_scale", 0.125)
            sw_batch_size = self.sliding_window_cfg.get("sw_batch_size", 1)
            self._sliding_window_inferer = SlidingWindowInferer(
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
                mode=mode,
                sigma_scale=sigma_scale,
                padding_mode="constant",
                cval=0.0,
            )
        return self._sliding_window_inferer

    def _get_divisible_pad(self) -> DivisiblePad:
        """Get or create divisible pad transform."""
        if self._divisible_pad is None:
            self._divisible_pad = DivisiblePad(k=self._downsampling_factor, mode="constant")
        return self._divisible_pad

    def encode_images_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to discrete tokens using VQVAE.

        Uses sliding window inference if enabled, otherwise standard encoding.

        Args:
            x: Input images [B, C, D, H, W]

        Returns:
            Token indices [B, D', H', W']
        """
        inferer = self._get_sliding_window_inferer()

        if inferer is None:
            # Standard encoding
            return self.maskgit.encode_tokens(x)

        # Sliding window encoding for large images
        original_shape = x.shape[2:]
        x_padded = self._get_divisible_pad()(x)

        # Encode patches and get indices
        def encode_fn(patch: torch.Tensor) -> torch.Tensor:
            """Encode patch and return one-hot indices."""
            z_q, _, indices = self.vqvae.encode(patch)
            # Return indices as float for MONAI compatibility
            return indices.float().unsqueeze(1)

        # Use sliding window to encode
        indices_padded = inferer(x_padded, encode_fn)

        # Crop back to original spatial dimensions
        # Indices spatial shape is smaller due to downsampling
        latent_d = original_shape[0] // self._downsampling_factor
        latent_h = original_shape[1] // self._downsampling_factor
        latent_w = original_shape[2] // self._downsampling_factor

        # Extract the center region
        B = x.shape[0]
        indices = indices_padded[:B, 0, :latent_d, :latent_h, :latent_w]  # type: ignore[index,misc,call-overload]

        # Convert to long and shift for transformer
        indices = indices.long()
        return (indices + 1) % self.maskgit.codebook_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.maskgit(x)
        return out

    def _extract_input_tensor(
        self,
        batch: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(batch, (tuple, list)):
            return batch[0]
        return batch

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, Any]:
        x = self._extract_input_tensor(batch)

        # Encode images to tokens using sliding window if enabled
        tokens = self.encode_images_to_tokens(x)

        # Compute loss using tokens
        loss, metrics = self._compute_loss_from_tokens(tokens)

        self.training_step_count += 1

        # Return loss and metrics for callback processing
        return {
            "loss": loss,
            "log_data": metrics,
        }

    def _compute_loss_from_tokens(
        self,
        tokens: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute MaskGIT loss from pre-encoded tokens.

        Args:
            tokens: Token indices [B, D, H, W]
            mask_ratio: Optional mask ratio

        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        from ..models.maskgit.scheduling import TrainingMaskScheduler

        B, D, H, W = tokens.shape
        tokens_flat = tokens.view(B, -1)
        n_total = tokens_flat.shape[1]

        # Sample mask ratio if not provided
        if mask_ratio is None:
            scheduler = TrainingMaskScheduler(gamma_type=self.maskgit.mask_scheduler.gamma_type)
            mask_ratio = scheduler.sample_mask_ratio()

        # Random masking
        mask = torch.rand(B, n_total, device=tokens.device) < mask_ratio

        # Ensure at least one token masked per sample
        for i in range(B):
            if not mask[i].any():
                mask[i, torch.randint(0, n_total, (1,), device=tokens.device)] = True

        # Get predictions
        logits = self.maskgit.transformer.forward(tokens_flat, mask_indices=mask)

        # Compute loss only on masked positions
        masked_logits = logits[mask]
        masked_targets = tokens_flat[mask]

        loss = nn.functional.cross_entropy(masked_logits, masked_targets)

        # Get accuracy on masked positions
        with torch.no_grad():
            preds = masked_logits.argmax(dim=-1)
            acc = (preds == masked_targets).float().mean()

        metrics = {
            "loss": loss.item(),
            "mask_acc": acc.item(),
            "mask_ratio": mask_ratio,
        }
        return loss, metrics

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, Any]:
        x = self._extract_input_tensor(batch)

        # Encode images to tokens using sliding window if enabled
        tokens = self.encode_images_to_tokens(x)

        # Compute loss
        loss, metrics = self._compute_loss_from_tokens(tokens)

        # Return loss and metrics for callback processing
        return {
            "loss": loss,
            "log_data": metrics,
        }

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, Any]:
        x = self._extract_input_tensor(batch)

        # Encode images to tokens using sliding window if enabled
        tokens = self.encode_images_to_tokens(x)

        # Compute loss
        loss, metrics = self._compute_loss_from_tokens(tokens)

        # Return loss and metrics for callback processing
        return {
            "loss": loss,
            "log_data": metrics,
        }

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
