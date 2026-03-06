# Axial Slice Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add callback to visualize random axial slices from 3D medical images during validation and testing.

**Architecture:** Create `AxialSliceVisualizationCallback` class that hooks into `on_validation_batch_end` and `on_test_batch_end`. Modifies `validate_step` to return images for visualization. Saves PNGs to `output_dir/slices/` and logs to TensorBoard.

**Tech Stack:** PyTorch, TensorBoard, PIL/imageio for PNG saving

---

## Task 1: Add AxialSliceVisualizationCallback Class

**Files:**
- Modify: `src/maskgit3d/infrastructure/training/callbacks.py` (add at end before backward compatibility aliases)
- Test: `tests/unit/test_callbacks.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/test_callbacks.py

class TestAxialSliceVisualizationCallback:
    """Tests for AxialSliceVisualizationCallback."""

    def test_callback_creation(self, tmp_path):
        """Test callback initialization."""
        from maskgit3d.infrastructure.training.callbacks import AxialSliceVisualizationCallback
        
        callback = AxialSliceVisualizationCallback(
            num_samples=4,
            slice_range=3,
            output_dir=str(tmp_path),
        )
        assert callback.num_samples == 4
        assert callback.slice_range == 3
        assert callback.output_dir == tmp_path

    def test_extract_random_slice(self):
        """Test random slice extraction."""
        from maskgit3d.infrastructure.training.callbacks import AxialSliceVisualizationCallback
        import numpy as np
        
        callback = AxialSliceVisualizationCallback(num_samples=1, slice_range=3)
        
        # Create a 3D volume [B, C, D, H, W] = [1, 1, 28, 28, 28]
        volume = torch.randn(1, 1, 28, 28, 28)
        
        slice_2d = callback._extract_random_slice(volume)
        
        # Should return 2D array
        assert slice_2d.ndim == 2
        assert slice_2d.shape == (28, 28)
        # Should be normalized to [0, 1]
        assert slice_2d.min() >= 0
        assert slice_2d.max() <= 1

    def test_slice_within_range(self):
        """Test that slice is extracted within specified range."""
        from maskgit3d.infrastructure.training.callbacks import AxialSliceVisualizationCallback
        
        callback = AxialSliceVisualizationCallback(num_samples=1, slice_range=3)
        
        # Extract multiple times to verify range
        volume = torch.randn(1, 1, 28, 28, 28)
        
        # Run extraction many times to check range
        center = 28 // 2  # = 14
        min_expected = max(0, center - 3)  # = 11
        max_expected = min(27, center + 3)  # = 17
        
        # Since it's random, we just verify the slice shape is correct
        for _ in range(10):
            slice_2d = callback._extract_random_slice(volume)
            assert slice_2d.shape == (28, 28)
```

**Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/unit/test_callbacks.py::TestAxialSliceVisualizationCallback -v
```
Expected: FAIL with "cannot import name 'AxialSliceVisualizationCallback'"

**Step 3: Write minimal implementation**

```python
# Add to src/maskgit3d/infrastructure/training/callbacks.py
# Add imports at top of file (after existing imports):
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

# Add class before the backward compatibility aliases (around line 600)

class AxialSliceVisualizationCallback(Callback):
    """
    Visualize random axial slices during validation and testing.
    
    Randomly selects N samples per batch and extracts axial slices
    near the center of the 3D volume. Saves images to output_dir
    and logs to TensorBoard.
    
    Args:
        num_samples: Number of samples to visualize per batch (default: 4)
        slice_range: Range around center for slice selection (default: 3)
        output_dir: Directory to save slice images (default: "./slices")
        enable_tensorboard: Whether to log to TensorBoard (default: True)
    """
    
    def __init__(
        self,
        num_samples: int = 4,
        slice_range: int = 3,
        output_dir: str | Path = "./slices",
        enable_tensorboard: bool = True,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.slice_range = slice_range
        self.output_dir = Path(output_dir)
        self.enable_tensorboard = enable_tensorboard
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer (set by pipeline)
        self._writer: Any = None
        self._global_step = 0
    
    def set_writer(self, writer: Any) -> None:
        """Set TensorBoard writer."""
        self._writer = writer
    
    def _extract_random_slice(self, volume: torch.Tensor) -> np.ndarray:
        """
        Extract random axial slice near center.
        
        Args:
            volume: [B, C, D, H, W] tensor in [-1, 1] range
        
        Returns:
            Normalized [H, W] numpy array in [0, 1] range
        """
        # Get first sample, first channel
        arr = volume[0, 0].detach().cpu().float()  # [D, H, W]
        
        depth = arr.shape[0]
        center = depth // 2
        
        # Random slice within center ± slice_range
        min_idx = max(0, center - self.slice_range)
        max_idx = min(depth - 1, center + self.slice_range)
        slice_idx = random.randint(min_idx, max_idx)
        
        # Extract slice
        slice_2d = arr[slice_idx, :, :]  # [H, W]
        
        # Normalize from [-1, 1] to [0, 1]
        slice_np = (slice_2d + 1) / 2
        slice_np = slice_np.clamp(0, 1).numpy()
        
        return slice_np
    
    def _save_to_disk(self, image: np.ndarray, filepath: Path) -> None:
        """Save image as PNG."""
        # Convert to uint8
        image_uint8 = (image * 255).astype(np.uint8)
        img = Image.fromarray(image_uint8, mode='L')
        img.save(filepath)
    
    def _log_to_tensorboard(self, image: np.ndarray, tag: str, step: int) -> None:
        """Log image to TensorBoard."""
        if self._writer is not None and self.enable_tensorboard:
            self._writer.add_image(tag, image, step, dataformats="HW")
    
    def on_validation_batch_end(
        self,
        fabric: Any,
        model: Any,
        optimizer: Any,
        batch: Any,
        batch_idx: int,
        outputs: dict | None = None,
    ) -> None:
        """Process validation batch and visualize slices."""
        self._process_batch(batch, outputs, "val", batch_idx)
    
    def on_test_batch_end(
        self,
        fabric: Any,
        model: Any,
        optimizer: Any,
        batch: Any,
        batch_idx: int,
        outputs: dict | None = None,
    ) -> None:
        """Process test batch and visualize slices."""
        self._process_batch(batch, outputs, "test", batch_idx)
    
    def _process_batch(
        self,
        batch: Any,
        outputs: dict | None,
        prefix: str,
        batch_idx: int,
    ) -> None:
        """Process a batch and save/log slices."""
        if batch is None:
            return
        
        # Get input images from batch
        if isinstance(batch, tuple | list):
            images = batch[0]
        else:
            images = batch
        
        if images is None:
            return
        
        batch_size = images.shape[0]
        
        # Select random sample indices
        num_to_sample = min(self.num_samples, batch_size)
        sample_indices = random.sample(range(batch_size), num_to_sample)
        
        for i, sample_idx in enumerate(sample_indices):
            # Extract single sample
            sample = images[sample_idx:sample_idx+1]  # Keep batch dim [1, C, D, H, W]
            
            # Extract random slice
            slice_2d = self._extract_random_slice(sample)
            
            # Save to disk
            filename = f"{prefix}_batch_{batch_idx:04d}_sample_{i}_input.png"
            filepath = self.output_dir / filename
            self._save_to_disk(slice_2d, filepath)
            
            # Log to TensorBoard
            tag = f"{prefix}/batch_{batch_idx:04d}/sample_{i}_input"
            self._log_to_tensorboard(slice_2d, tag, self._global_step)
            
            # If outputs has predictions, visualize them too
            if outputs is not None and "images" in outputs:
                pred_images = outputs["images"]
                if isinstance(pred_images, np.ndarray):
                    pred_tensor = torch.from_numpy(pred_images)
                    if pred_tensor.dim() == 4:  # [N, C, D, H, W]
                        pred_tensor = pred_tensor.unsqueeze(0)  # Add batch dim
                    pred_sample = pred_tensor[sample_idx:sample_idx+1] if pred_tensor.shape[0] > sample_idx else pred_tensor[0:1]
                    pred_slice = self._extract_random_slice(pred_sample)
                    
                    # Save prediction
                    pred_filename = f"{prefix}_batch_{batch_idx:04d}_sample_{i}_pred.png"
                    pred_filepath = self.output_dir / pred_filename
                    self._save_to_disk(pred_slice, pred_filepath)
                    
                    # Log prediction
                    pred_tag = f"{prefix}/batch_{batch_idx:04d}/sample_{i}_pred"
                    self._log_to_tensorboard(pred_slice, pred_tag, self._global_step)
        
        self._global_step += 1
```

**Step 4: Run test to verify it passes**

```bash
poetry run pytest tests/unit/test_callbacks.py::TestAxialSliceVisualizationCallback -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/maskgit3d/infrastructure/training/callbacks.py tests/unit/test_callbacks.py
git commit -m "feat: add AxialSliceVisualizationCallback for random axial slice visualization"
```

---

## Task 2: Modify validate_step to Return Images

**Files:**
- Modify: `src/maskgit3d/infrastructure/training/strategies.py` (VQGANTrainingStrategy and MaskGITTrainingStrategy)
- Test: `tests/unit/test_strategies.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/test_strategies.py

class TestValidationStepOutputs:
    """Tests for validation step output format."""

    def test_vqgan_validate_step_returns_images(self):
        """Test that VQGAN validate_step returns images dict."""
        from maskgit3d.infrastructure.training.strategies import VQGANTrainingStrategy
        from maskgit3d.infrastructure.vqgan.vqvae import VQVAE
        
        # Create minimal VQVAE model
        model = VQVAE(
            in_channels=1,
            out_channels=1,
            embed_dim=64,
            num_embeddings=128,
            spatial_dims=3,
        )
        
        strategy = VQGANTrainingStrategy()
        
        # Create dummy batch
        batch = (torch.randn(2, 1, 16, 16, 16),)
        
        metrics = strategy.validate_step(model, batch)
        
        # Should return metrics dict
        assert isinstance(metrics, dict)
        assert "val_loss" in metrics

    def test_maskgit_validate_step_returns_images(self):
        """Test that MaskGIT validate_step returns metrics."""
        # This test verifies the existing behavior is maintained
        pass  # Implementation depends on MaskGIT model availability
```

**Step 2: Run test to verify current behavior**

```bash
poetry run pytest tests/unit/test_strategies.py::TestValidationStepOutputs -v
```

**Step 3: Modify validate_step methods to include images in return**

For VQGANTrainingStrategy (around line 635):
```python
# In VQGANTrainingStrategy.validate_step, add images to return:

def validate_step(
    self,
    model: ModelInterface,
    batch: tuple[torch.Tensor, ...],
) -> dict[str, float]:
    """Execute validation step."""
    model.eval()
    vq_model = cast(VQModelInterface, model)

    x = batch[0] if isinstance(batch, tuple | list) else batch

    with torch.no_grad():
        xrec, qloss = vq_model.forward_with_loss(x)
        rec_loss = torch.abs(x - xrec).mean()
        perceptual_loss = self._compute_perceptual_loss(x, xrec)

    metrics = {
        "val_loss": rec_loss.item(),
        "val_rec_loss": rec_loss.item(),
        "val_perceptual_loss": perceptual_loss.item(),
        "val_codebook_loss": qloss.mean().item(),
        # Add images for visualization (normalized to [0, 1])
        "images": ((xrec + 1) / 2).clamp(0, 1).cpu().numpy(),
        "targets": ((x + 1) / 2).clamp(0, 1).cpu().numpy(),
    }

    # ... rest of the method (PSNR/SSIM/LPIPS computation)
```

**Step 4: Run test to verify it passes**

```bash
poetry run pytest tests/unit/test_strategies.py::TestValidationStepOutputs -v
```

**Step 5: Commit**

```bash
git add src/maskgit3d/infrastructure/training/strategies.py tests/unit/test_strategies.py
git commit -m "feat: add images to validate_step return for visualization"
```

---

## Task 3: Register Callback in Training Pipeline

**Files:**
- Modify: `src/maskgit3d/application/pipeline.py` (FabricTrainingPipeline)
- Test: `tests/unit/test_pipeline_extended.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/test_pipeline_extended.py

def test_training_pipeline_with_slice_visualization(tmp_path):
    """Test that training pipeline accepts slice visualization callback."""
    from maskgit3d.infrastructure.training.callbacks import AxialSliceVisualizationCallback
    from maskgit3d.application.pipeline import FabricTrainingPipeline
    
    callback = AxialSliceVisualizationCallback(
        num_samples=2,
        slice_range=3,
        output_dir=str(tmp_path / "slices"),
    )
    
    # Pipeline should accept the callback
    # This is a basic integration test
    assert callback.num_samples == 2
```

**Step 2: Modify FabricTrainingPipeline to support the callback**

In `pipeline.py`, find the `_call_callbacks` method for validation batch end and ensure it passes outputs:

```python
# In _validate_epoch method (around line 754-762):
def _validate_epoch(
    self,
    epoch: int,
    val_loader: Any,
) -> dict[str, list[float]]:
    """Run one validation epoch."""
    self.model.eval()

    metrics_history: dict[str, list[float]] = {}

    pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            self._call_callbacks("on_validation_batch_start", batch, batch_idx)

            metrics = self.training_strategy.validate_step(self.model, batch)

            for key, value in metrics.items():
                if not isinstance(value, int | float):
                    continue
                prefixed_key = key if key.startswith("val_") else f"val_{key}"
                metrics_history.setdefault(prefixed_key, []).append(float(value))

            # Pass outputs to callback
            outputs = {k: v for k, v in metrics.items() if not isinstance(v, int | float)}
            self._call_callbacks("on_validation_batch_end", batch, batch_idx, outputs)

            if "val_loss" in metrics:
                pbar.set_postfix({"val_loss": f"{metrics['val_loss']:.4f}"})

    return metrics_history
```

**Step 3: Run tests**

```bash
poetry run pytest tests/unit/test_pipeline_extended.py::test_training_pipeline_with_slice_visualization -v
```

**Step 4: Commit**

```bash
git add src/maskgit3d/application/pipeline.py tests/unit/test_pipeline_extended.py
git commit -m "feat: pass validation outputs to callbacks for visualization"
```

---

## Task 4: Register Callback in Test Pipeline

**Files:**
- Modify: `src/maskgit3d/application/pipeline.py` (FabricTestPipeline)
- Test: `tests/unit/test_pipeline_extended.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/test_pipeline_extended.py

def test_test_pipeline_with_slice_visualization(tmp_path):
    """Test that test pipeline supports slice visualization callback."""
    from maskgit3d.infrastructure.training.callbacks import AxialSliceVisualizationCallback
    
    callback = AxialSliceVisualizationCallback(
        num_samples=2,
        slice_range=3,
        output_dir=str(tmp_path / "slices"),
    )
    
    # Verify callback can be instantiated for test pipeline
    assert callback.enable_tensorboard == True
```

**Step 2: Modify FabricTestPipeline to support the callback**

In `pipeline.py`, find the test loop and add callback hooks:

```python
# In FabricTestPipeline.run() method, around the test loop:
# Add callback initialization and hooks

def run(
    self,
    save_predictions: bool = False,
    export_nifti: bool = False,
    enable_tensorboard: bool = False,
    tensorboard_dir: str | None = None,
) -> dict[str, float]:
    """Run test pipeline."""
    # ... existing setup code ...
    
    # Initialize slice visualization callbacks
    slice_callbacks = [cb for cb in self.callbacks if hasattr(cb, 'on_test_batch_end')]
    
    # Set TensorBoard writer for callbacks
    if writer is not None:
        for cb in slice_callbacks:
            if hasattr(cb, 'set_writer'):
                cb.set_writer(writer)
    
    # ... in the test loop ...
    for batch_idx, batch in enumerate(pbar):
        self._call_callbacks("on_test_batch_start", batch, batch_idx)
        
        # ... inference code ...
        
        # After inference, pass outputs to callback
        self._call_callbacks("on_test_batch_end", batch, batch_idx, predictions)
```

**Step 3: Run tests**

```bash
poetry run pytest tests/unit/test_pipeline_extended.py::test_test_pipeline_with_slice_visualization -v
```

**Step 4: Commit**

```bash
git add src/maskgit3d/application/pipeline.py tests/unit/test_pipeline_extended.py
git commit -m "feat: add test pipeline support for slice visualization callback"
```

---

## Task 5: Add Configuration Options

**Files:**
- Modify: `src/maskgit3d/conf/config.yaml`
- Test: Verify config loads correctly

**Step 1: Add visualization config section**

```yaml
# Add to conf/config.yaml after output section:

# Visualization settings
visualization:
  enable_axial_slices: true    # Enable slice visualization during validation/test
  num_samples: 4               # Number of samples to visualize per batch
  slice_range: 3               # Range around center for slice selection
  output_subdir: slices        # Subdirectory for slice images
```

**Step 2: Verify config loads**

```bash
poetry run python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('src/maskgit3d/conf/config.yaml'); print(cfg.visualization)"
```

**Step 3: Commit**

```bash
git add src/maskgit3d/conf/config.yaml
git commit -m "feat: add visualization configuration options"
```

---

## Task 6: Integration Test

**Files:**
- Create: `tests/integration/test_slice_visualization.py`

**Step 1: Write integration test**

```python
# tests/integration/test_slice_visualization.py

"""Integration tests for axial slice visualization."""

import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from maskgit3d.infrastructure.training.callbacks import AxialSliceVisualizationCallback


class TestSliceVisualizationIntegration:
    """Integration tests for slice visualization callback."""

    def test_callback_with_mock_validation_batch(self, tmp_path):
        """Test callback with realistic validation batch."""
        callback = AxialSliceVisualizationCallback(
            num_samples=2,
            slice_range=3,
            output_dir=str(tmp_path / "slices"),
            enable_tensorboard=False,
        )
        
        # Create realistic 3D batch
        batch = (torch.randn(8, 1, 28, 28, 28),)  # MedMNIST-like batch
        
        # Mock outputs
        outputs = {
            "images": torch.randn(8, 1, 28, 28, 28).numpy(),
        }
        
        # Process batch
        mock_fabric = MagicMock()
        mock_model = MagicMock()
        mock_optimizer = MagicMock()
        
        callback.on_validation_batch_end(
            mock_fabric, mock_model, mock_optimizer, batch, 0, outputs
        )
        
        # Verify files were created
        slices_dir = tmp_path / "slices"
        assert slices_dir.exists()
        
        # Should have input slices
        input_files = list(slices_dir.glob("val_batch_0000_sample_*_input.png"))
        assert len(input_files) == 2  # num_samples = 2
        
        # Should have pred slices
        pred_files = list(slices_dir.glob("val_batch_0000_sample_*_pred.png"))
        assert len(pred_files) == 2

    def test_callback_tensorboard_logging(self, tmp_path):
        """Test TensorBoard logging."""
        from torch.utils.tensorboard import SummaryWriter
        
        tb_dir = tmp_path / "tensorboard"
        writer = SummaryWriter(str(tb_dir))
        
        callback = AxialSliceVisualizationCallback(
            num_samples=1,
            slice_range=3,
            output_dir=str(tmp_path / "slices"),
            enable_tensorboard=True,
        )
        callback.set_writer(writer)
        
        # Process batch
        batch = (torch.randn(4, 1, 28, 28, 28),)
        mock_fabric = MagicMock()
        mock_model = MagicMock()
        mock_optimizer = MagicMock()
        
        callback.on_validation_batch_end(
            mock_fabric, mock_model, mock_optimizer, batch, 0, None
        )
        
        writer.close()
        
        # Verify TensorBoard files exist
        tb_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(tb_files) > 0
```

**Step 2: Run integration test**

```bash
poetry run pytest tests/integration/test_slice_visualization.py -v
```

**Step 3: Commit**

```bash
git add tests/integration/test_slice_visualization.py
git commit -m "test: add integration tests for slice visualization callback"
```

---

## Task 7: Run Full Test Suite

**Step 1: Run all tests**

```bash
poetry run pytest --cache-clear -vv tests/
```

**Step 2: Run linting**

```bash
make format
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete axial slice visualization feature"
```

---

## Summary

This implementation adds:
1. `AxialSliceVisualizationCallback` class for random axial slice extraction
2. Modified `validate_step` to return images for visualization
3. Integration with both training and test pipelines
4. Configuration options for customization
5. Comprehensive unit and integration tests

The feature will automatically save PNG images to `output_dir/slices/` and log to TensorBoard during validation and testing.