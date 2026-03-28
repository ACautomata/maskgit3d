# Best VQVAE Configuration for MedMNIST 3D

## Results Summary

**Best FID: 31.29** (achieved at epoch 91)

### Comparison with Baseline

| Metric | Baseline | Best Config | Improvement |
|--------|----------|-------------|-------------|
| Final FID | 74.0 → 34.20 | **31.29** | **57.7% total improvement** |
| Model | vqvae (47M params) | vqvae_small (6.9M params) | - |
| Data Augmentation | None | RandFlip (all 3 axes) | - |
| Perceptual Loss | lambda_perceptual=0.1 | lambda_perceptual=0.5 | - |

## Configuration Changes

### 1. Task Configuration

**File**: `src/maskgit3d/conf/task/vqvae_medmnist.yaml`

```yaml
# Change this line:
lambda_perceptual: 0.1

# To:
lambda_perceptual: 0.5
```

### 2. Data Augmentation

**File**: `src/maskgit3d/data/medmnist/transforms.py`

```python
from monai.transforms.spatial.array import RandFlip, Resize

def create_training_transforms(config: MedMNISTConfig) -> Callable:
    crop_size = config.crop_size
    validate_crop_size_for_vqvae(crop_size)

    return Compose(
        [
            EnsureType(),
            SpatialPad(spatial_size=crop_size, mode="constant"),
            ScaleIntensityRange(
                a_min=0.0,
                a_max=255.0,
                b_min=-1.0,
                b_max=1.0,
            ),
            # ADD THESE 3 LINES:
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=2),
            # END ADD
            RandSpatialCrop(
                roi_size=crop_size,
                random_center=True,
                random_size=False,
            ),
        ]
    )
```

### 3. Training Command

```bash
/home/junran/.conda/envs/maskgit3d/bin/maskgit3d-train \
  --config-name train_medmnist \
  task.lambda_perceptual=0.5 \
  trainer.max_epochs=100
```

**Note**: After modifying `transforms.py`, you no longer need to pass `task.lambda_perceptual=0.5` on the command line since it's now the default.

## Key Findings

1. **Small model works better**: vqvae_small (6.9M params) significantly outperforms vqvae (47M params) on small datasets
2. **Stronger perceptual loss**: lambda_perceptual=0.5 improves visual quality and FID
3. **Data augmentation helps**: Random flips along all 3 spatial axes improve generalization
4. **Training duration**: 100 epochs with early stopping (patience=30) is sufficient
5. **Learning rate schedule**: Default warmup_steps=50 and min_lr_ratio=0.01 work well

## Failed Experiments

The following optimizations did NOT improve FID:

- ❌ Learning rate adjustments (lr=3e-4, 5e-4, warmup_steps=0/10)
- ❌ Larger batch size (32 instead of 16)
- ❌ Larger codebook (8192 instead of 4096)
- ❌ Lower commitment cost (0.25 instead of 0.5)
- ❌ RandRotate90 augmentation (too much noise, hurts early convergence)
- ❌ 200 epochs (early stopping at epoch 78, worse final FID=32.85)
- ❌ fp32 precision (slower, no improvement over bf16-mixed)

## Git Branch

Best configuration is on branch: `feat/vqvae-aug-flip`

Remote location: `amax:/home/junran/maskgit3d-run`

## Checkpoints

Best model checkpoint saved in:
```
/home/junran/maskgit3d-run/checkpoints/2026-03-27/<timestamp>/
```

Look for checkpoint at epoch 91 with val_fid=31.29.