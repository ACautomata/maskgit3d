# Axial Slice Visualization Design

**Date**: 2026-03-06
**Status**: Approved
**Author**: Design via brainstorming session

## Overview

Add functionality to visualize random axial slices from 3D medical images during validation and testing. Slices near the center of the volume will be extracted and saved to both `output_dir` (as PNG files) and TensorBoard.

## Requirements

### Functional Requirements
- Random selection of N samples per batch during validation/testing
- Extraction of axial slices near the center (center ± N slices)
- Save to output directory as PNG images
- Log to TensorBoard for real-time visualization

### Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_samples` | 4 | Number of samples to visualize per batch |
| `slice_range` | 3 | Range around center for random slice selection |
| `output_format` | PNG | Image format for saved slices |

## Architecture

### Component: `AxialSliceVisualizationCallback`

A new callback class that hooks into validation and test batch end events.

```
AxialSliceVisualizationCallback
├── __init__(num_samples, slice_range, output_dir, tensorboard_writer)
├── on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)
├── on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx)
├── _process_batch(batch, outputs, prefix, batch_idx, step)
├── _extract_random_slice(volume) -> np.ndarray
├── _save_to_disk(image, filepath)
└── _log_to_tensorboard(image, tag, step)
```

### Data Flow

```
Validation/Test Batch
        │
        ▼
validate_step() / test inference
        │
        ▼
on_validation_batch_end / on_test_batch_end
        │
        ├── Random sample selection (4 samples)
        │
        ├── Random slice extraction (center ± 3)
        │
        ├── Normalize to [0, 1]
        │
        ├── Save PNG to output_dir/slices/
        │
        └── Log to TensorBoard (if enabled)
```

## Implementation Details

### Slice Extraction Algorithm

```python
def _extract_random_slice(self, volume: torch.Tensor) -> np.ndarray:
    """
    Extract random axial slice near center.
    
    Args:
        volume: [B, C, D, H, W] tensor in [-1, 1] range
    
    Returns:
        Normalized [H, W] numpy array in [0, 1] range
    """
    depth = volume.shape[2]
    center = depth // 2
    
    # Random slice within center ± slice_range
    min_idx = max(0, center - self.slice_range)
    max_idx = min(depth - 1, center + self.slice_range)
    slice_idx = random.randint(min_idx, max_idx)
    
    # Extract and normalize
    slice_2d = volume[0, 0, slice_idx, :, :].detach().cpu()
    slice_np = (slice_2d + 1) / 2  # [-1,1] -> [0,1]
    return slice_np.clamp(0, 1).numpy()
```

### Output File Structure

```
output_dir/
├── tensorboard/
│   └── events.out.tfevents...
└── slices/
    ├── val_batch_000_sample_0_input.png
    ├── val_batch_000_sample_0_pred.png
    ├── val_batch_000_sample_0_target.png
    ├── test_batch_001_sample_2_input.png
    └── ...
```

### TensorBoard Tags

```
validation/
├── batch_000/
│   ├── sample_0_input
│   ├── sample_0_prediction
│   └── sample_0_target
└── ...

test/
├── batch_000/
│   ├── sample_0_input
│   ├── sample_0_prediction
│   └── sample_0_target
└── ...
```

## Configuration

Add to `conf/config.yaml`:

```yaml
visualization:
  enable_axial_slices: true    # Enable slice visualization
  num_samples: 4               # Samples per batch
  slice_range: 3               # Range around center
  output_subdir: slices        # Subdirectory for PNG files
```

## Files to Modify

| File | Change Description |
|------|-------------------|
| `src/maskgit3d/infrastructure/training/callbacks.py` | Add `AxialSliceVisualizationCallback` class |
| `src/maskgit3d/infrastructure/training/strategies.py` | Modify `validate_step()` to return images dict |
| `src/maskgit3d/application/pipeline.py` | Register callback in pipelines, pass TensorBoard writer |
| `src/maskgit3d/conf/config.yaml` | Add visualization configuration section |

## Integration Points

### Validation Pipeline (`FabricTrainingPipeline`)

1. Create callback in `__init__` or `_setup_callbacks()`
2. Callback hooks into `on_validation_batch_end`
3. Access outputs from `validate_step()` return value

### Test Pipeline (`FabricTestPipeline`)

1. Create callback in `run()` method
2. Callback hooks into `on_test_batch_end`
3. Access outputs from inference results

## Error Handling

- Handle missing TensorBoard writer gracefully (skip logging)
- Handle empty/invalid batches (skip processing)
- Ensure output directory exists before saving

## Testing Strategy

1. Unit tests for slice extraction logic
2. Integration tests with mock pipelines
3. Visual verification with actual model outputs

## Dependencies

- `torch` - Tensor operations
- `numpy` - Array manipulation
- `PIL` or `imageio` - PNG saving
- `tensorboard` - TensorBoard logging (already available)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Performance impact on validation | Process only first N batches by default |
| Disk space for PNG files | Implement max file limit or cleanup |
| TensorBoard memory usage | Log only every N batches |

## References

- Existing `_centre_slice()` function in `pipeline.py:471-482`
- Existing `FID2p5DCallback` pattern in `fid_callback.py`
- Callback system in `callbacks.py`