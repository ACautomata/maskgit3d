# maskgit-3d v2 Architecture Design

**Date**: 2026-03-07
**Author**: Design Team
**Status**: Approved

## Overview

v2 is a complete refactoring of the maskgit-3d project following PyTorch Lightning and Hydra best practices. No backward compatibility with v1.

## Design Principles

1. **Pure Lightning Architecture**: LightningModule + Trainer, remove Fabric
2. **Model-Task Separation**: nn.Module for networks, LightningModule for training logic
3. **Minimal Abstraction**: Only Protocol, BaseTask, DataModule
4. **Hydra Composition**: Configuration-driven instantiation via defaults
5. **MONAI Integration**: Keep MONAI components (Metrics, MAISI VAE structure)

## Architecture

### Project Structure

```
src/maskgit3d/
├── models/                    # Pure nn.Module
│   ├── __init__.py
│   ├── vqvae/
│   │   ├── __init__.py
│   │   ├── encoder.py        # MaisiEncoder wrapper
│   │   ├── decoder.py        # MaisiDecoder wrapper
│   │   └── quantizer.py      # VectorQuantizer
│   ├── discriminator/
│   │   ├── __init__.py
│   │   └── patch_discriminator.py
│   └── maskgit/
│       ├── __init__.py
│       └── transformer.py    # MaskGIT Transformer
│
├── tasks/                     # LightningModule
│   ├── __init__.py
│   ├── base_task.py          # Thin base class
│   ├── vqvae_task.py         # VQVAE training (manual optimization)
│   └── maskgit_task.py       # MaskGIT training (automatic optimization)
│
├── data/                      # LightningDataModule
│   ├── __init__.py
│   ├── medmnist3d.py
│   └── brats.py
│
├── losses/                    # Loss functions
│   ├── __init__.py
│   ├── gan_loss.py
│   └── vq_loss.py
│
├── metrics/                   # MONAI metrics wrapper
│   ├── __init__.py
│   └── image_metrics.py      # SSIM, PSNR, Dice
│
├── callbacks/                 # Lightning Callbacks
│   ├── __init__.py
│   └── ema_callback.py
│
├── utils/
│   ├── __init__.py
│   └── instantiators.py
│
├── train.py                   # Training entry
├── eval.py                    # Evaluation entry
└── inference.py               # Inference entry
│
configs/
├── train.yaml                 # Main config
├── eval.yaml
├── inference.yaml
├── model/
│   ├── vqvae.yaml
│   └── maskgit.yaml
├── task/
│   ├── vqvae.yaml
│   └── maskgit.yaml
├── data/
│   ├── medmnist3d.yaml
│   └── brats.yaml
├── optimizer/
│   ├── adam.yaml
│   └── adamw.yaml
├── scheduler/
│   ├── cosine.yaml
│   └── none.yaml
├── trainer/
│   └── default.yaml
├── callbacks/
│   └── default.yaml
└── logger/
    └── tensorboard.yaml
```

### Layer Boundaries

| Layer | Allowed | Forbidden |
|-------|---------|-----------|
| `models/` (nn.Module) | Network structure, forward pass | optimizer, scheduler, log, metric, trainer state |
| `tasks/` (LightningModule) | Training/validation/test logic, optimizers | Data download, path scanning, complex preprocessing |
| `data/` (DataModule) | Data loading, preprocessing, augmentation | Model logic, loss weights, class priors |
| `train.py` | Config assembly, instantiation, Trainer launch | Business logic |

## Key Components

### 1. Models Layer

**VQVAE** (`models/vqvae/`):
- `encoder.py`: MONAI MaisiEncoder wrapper
- `decoder.py`: MONAI MaisiDecoder wrapper
- `quantizer.py`: VectorQuantizer (from original implementation)

**Discriminator** (`models/discriminator/`):
- 3D PatchGAN discriminator for adversarial training

**MaskGIT Transformer** (`models/maskgit/`):
- Token embedding + Transformer encoder
- Masked token prediction

### 2. Tasks Layer

**BaseTask**:
- Thin base class with `save_hyperparameters()`
- `configure_optimizers()` using Hydra instantiation

**VQVAETask**:
- `automatic_optimization = False` for GAN training
- Dual optimizers (Generator + Discriminator)
- Manual optimization loop in `training_step()`

**MaskGITTask**:
- Standard automatic optimization
- Frozen VQVAE as tokenizer
- Single optimizer

### 3. Data Layer

**LightningDataModule pattern**:
- `setup()`: Create train/val/test datasets
- `train_dataloader()`, `val_dataloader()`, `test_dataloader()`
- MONAI transforms integration

### 4. Losses Layer

**GAN Loss**:
- Adversarial loss for Generator and Discriminator
- LSGAN or HingeGAN options

**VQ Loss**:
- Codebook loss
- Commitment loss

### 5. Metrics Layer

**Image Metrics** (MONAI wrappers):
- SSIM (Structural Similarity)
- PSNR (Peak Signal-to-Noise Ratio)
- Dice (for segmentation tasks)

### 6. Callbacks Layer

- `EMACallback`: Exponential Moving Average
- `ImageLoggingCallback`: Log generated images
- `CheckpointCallback`: Model checkpointing

## Hydra Configuration

### Main Config

```yaml
# configs/train.yaml
defaults:
  - _self_
  - model: vqvae
  - task: vqvae
  - data: medmnist3d
  - optimizer: adam
  - scheduler: cosine
  - trainer: default
  - callbacks: default
  - logger: tensorboard

seed: 42
ckpt_path: null
```

### Instantiable Configs

All configs use `_target_` for Hydra instantiation:

```yaml
# configs/model/vqvae.yaml
_target_: src.maskgit3d.models.vqvae.VQVAE
encoder_cfg:
  spatial_dims: 3
  in_channels: 1
  ...
```

### Task Config with Model Injection

```yaml
# configs/task/vqvae.yaml
_target_: src.maskgit3d.tasks.vqvae_task.VQVAETask
model_cfg: ${model}  # Reference to model config
discriminator_cfg:
  _target_: src.maskgit3d.models.discriminator.PatchDiscriminator3D
  in_channels: 1
optimizer_cfg: ${optimizer}
scheduler_cfg: ${scheduler}
```

## Entry Points

### train.py

```python
@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.data)
    task = instantiate(cfg.task)
    callbacks = instantiate(cfg.callbacks)
    logger = instantiate(cfg.logger)
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    
    trainer.fit(task, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
```

### eval.py

```python
@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.data)
    task = instantiate(cfg.task)
    task = task.load_from_checkpoint(cfg.ckpt_path)
    
    trainer = instantiate(cfg.trainer)
    trainer.validate(task, datamodule=datamodule)
```

### inference.py

```python
@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    task = instantiate(cfg.task)
    task = task.load_from_checkpoint(cfg.ckpt_path)
    
    # Inference logic
    ...
```

## Training Pipeline

### Stage 1: VQVAE Training

```bash
python src/maskgit3d/train.py task=vqvae data=medmnist3d
```

- GAN training with manual optimization
- Dual optimizers: Generator + Discriminator
- Losses: L1 reconstruction + VQ loss + Adversarial loss

### Stage 2: MaskGIT Training

```bash
python src/maskgit3d/train.py task=maskgit data=medmnist3d \
    task.vqvae_ckpt_path=./checkpoints/vqvae/best.ckpt
```

- Standard automatic optimization
- Frozen VQVAE tokenizer
- Loss: Cross-entropy on masked tokens

## Migration from v1

### Removed Components

1. **Fabric**: Replaced with LightningModule + Trainer
2. **injector**: Replaced with Hydra instantiation
3. **TrainingStrategy**: Merged into LightningModule
4. **DataProvider**: Replaced with LightningDataModule
5. **domain/interfaces.py**: Removed (no longer needed)

### Preserved Components

1. **MONAI Networks**: MaisiEncoder, MaisiDecoder
2. **MONAI Metrics**: SSIM, PSNR, Dice
3. **MONAI Transforms**: Data augmentation
4. **Core Algorithms**: VQVAE, MaskGIT Transformer

## Testing Strategy

1. **Unit Tests**: Each model component, loss function, metric
2. **Integration Tests**: Training loop, evaluation pipeline
3. **Config Tests**: Hydra instantiation validation

## Success Criteria

1. [ ] All v1 functionality preserved (not API-compatible)
2. [ ] VQVAE training converges with same quality
3. [ ] MaskGIT training converges with same quality
4. [ ] Test coverage > 80%
5. [ ] No Fabric, no injector dependencies
6. [ ] Hydra config validation passes
7. [ ] Lightning Trainer features work (logging, checkpointing, callbacks)