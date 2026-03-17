"""Sample saving callback for saving PNG samples during evaluation.

This callback saves generated or reconstructed samples as PNG images during
validation and test phases. For 3D volumes, the middle depth slice is extracted
for 2D visualization.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from PIL import Image


def _normalize_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Normalize a CHW tensor to uint8 numpy array via per-slice min-max.

    Args:
        tensor: Shape [C, H, W].

    Returns:
        uint8 numpy array of shape [C, H, W].
    """
    tensor = tensor.detach().cpu().float()
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val - min_val > 1e-6:
        tensor = (tensor - min_val) / (max_val - min_val)
    else:
        tensor = torch.zeros_like(tensor)
    return (tensor * 255).clamp(0, 255).to(torch.uint8).numpy()


def _extract_middle_slice(tensor: torch.Tensor) -> torch.Tensor:
    """Extract the middle depth slice from a 5-D volume tensor.

    Args:
        tensor: Shape [B, C, D, H, W].

    Returns:
        Tensor of shape [B, C, H, W].
    """
    middle_idx = tensor.shape[2] // 2
    return tensor[:, :, middle_idx, :, :]


def _chw_to_pil(chw: np.ndarray) -> Image.Image:
    """Convert a [C, H, W] uint8 numpy array to a PIL Image.

    Single-channel arrays become greyscale; three-channel arrays become RGB.

    Args:
        chw: uint8 array of shape [C, H, W].

    Returns:
        PIL Image.
    """
    c, h, w = chw.shape
    if c == 1:
        return Image.fromarray(chw[0], mode="L")
    if c == 3:
        return Image.fromarray(chw.transpose(1, 2, 0), mode="RGB")
    return Image.fromarray(chw[0], mode="L")


class SampleSavingCallback(Callback):
    """Callback that saves PNG samples during validation and test.

    Supports two output modes determined by the keys present in the batch
    outputs dict:

    * **VQVAE** mode – expects ``x_real`` and ``x_recon`` keys; saves a
      side-by-side comparison (real on the left, reconstruction on the right).
    * **MaskGIT** mode – expects a ``generated_images`` key; saves each
      generated image individually.

    For 3D volumes with shape ``[B, C, D, H, W]``, the middle slice along the
    depth axis is extracted before saving.

    Args:
        output_dir: Base directory for saved PNG files.
        max_samples_per_batch: Maximum number of samples to save per batch.
        save_validation: Whether to save samples during the validation loop.
        save_test: Whether to save samples during the test loop.

    Example:
        >>> cb = SampleSavingCallback(output_dir="outputs/samples")
        >>> trainer = Trainer(callbacks=[cb])
    """

    def __init__(
        self,
        output_dir: str = "samples",
        max_samples_per_batch: int = 4,
        save_validation: bool = True,
        save_test: bool = True,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.max_samples_per_batch = max_samples_per_batch
        self.save_validation = save_validation
        self.save_test = save_test

    def _save_vqvae_batch(
        self,
        x_real: torch.Tensor,
        x_recon: torch.Tensor,
        stage: str,
        batch_idx: int,
    ) -> None:
        """Save side-by-side VQVAE real/reconstruction pairs.

        Args:
            x_real: Tensor [B, C, D, H, W] or [B, C, H, W].
            x_recon: Tensor [B, C, D, H, W] or [B, C, H, W].
            stage: "validate" or "test".
            batch_idx: Index of the current batch.
        """
        out_dir = self.output_dir / stage
        out_dir.mkdir(parents=True, exist_ok=True)

        if x_real.dim() == 5:
            x_real = _extract_middle_slice(x_real)
            x_recon = _extract_middle_slice(x_recon)

        n = min(x_real.shape[0], self.max_samples_per_batch)
        for i in range(n):
            real_arr = _normalize_to_uint8(x_real[i])
            recon_arr = _normalize_to_uint8(x_recon[i])

            real_img = _chw_to_pil(real_arr)
            recon_img = _chw_to_pil(recon_arr)

            total_width = real_img.width + recon_img.width
            combined = Image.new(real_img.mode, (total_width, real_img.height))
            combined.paste(real_img, (0, 0))
            combined.paste(recon_img, (real_img.width, 0))

            filename = out_dir / f"batch{batch_idx:04d}_sample{i:02d}.png"
            combined.save(filename)

    def _save_maskgit_batch(
        self,
        generated: torch.Tensor,
        stage: str,
        batch_idx: int,
    ) -> None:
        """Save MaskGIT generated images.

        Args:
            generated: Tensor [B, C, D, H, W] or [B, C, H, W].
            stage: "validate" or "test".
            batch_idx: Index of the current batch.
        """
        out_dir = self.output_dir / stage
        out_dir.mkdir(parents=True, exist_ok=True)

        if generated.dim() == 5:
            generated = _extract_middle_slice(generated)

        n = min(generated.shape[0], self.max_samples_per_batch)
        for i in range(n):
            arr = _normalize_to_uint8(generated[i])
            img = _chw_to_pil(arr)
            filename = out_dir / f"batch{batch_idx:04d}_sample{i:02d}.png"
            img.save(filename)

    def _process_outputs(
        self,
        outputs: Any,
        stage: str,
        batch_idx: int,
    ) -> None:
        """Dispatch to the correct saving routine based on output keys.

        Args:
            outputs: The dict returned by the LightningModule step.
            stage: "validate" or "test".
            batch_idx: Index of the current batch.
        """
        if outputs is None or not isinstance(outputs, dict):
            return

        x_real = outputs.get("x_real")
        x_recon = outputs.get("x_recon")
        generated = outputs.get("generated_images")

        if x_real is not None and x_recon is not None:
            self._save_vqvae_batch(x_real, x_recon, stage, batch_idx)
        elif generated is not None:
            self._save_maskgit_batch(generated, stage, batch_idx)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Save validation samples if enabled."""
        if not self.save_validation:
            return
        self._process_outputs(outputs, "validate", batch_idx)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Save test samples if enabled."""
        if not self.save_test:
            return
        self._process_outputs(outputs, "test", batch_idx)
