# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **⚠️ ARCHITECTURE UPDATE**: Project migrated from Fabric + dependency injection to PyTorch Lightning + builder pattern. Old `cli/`, `config/`, `domain/`, `application/`, `infrastructure/` directories removed. See AGENTS.md for current architecture.

## Project Overview

**maskgit-3d** is a 3D medical image generation framework using PyTorch + Lightning + MONAI with two-stage training.

- **Two-Stage Training**: VQVAE (Stage 1) + MaskGIT (Stage 2)
- **Datasets**: MedMNIST-3D, BraTS
- **Configuration**: Hydra config composition with builder pattern
- **Architecture**: PyTorch Lightning tasks with training steps abstraction

## Runtime Environment

### Conda Environment

```bash
# Activate existing environment
conda activate maskgit3d

# Or create new environment
conda create -n maskgit3d python=3.10 -y
conda activate maskgit3d
```

### Installation

```bash
# Install with dependencies
pip install -e .

# With dev dependencies
pip install -e .[dev]
```

## Common Commands

### Training

```bash
# VQVAE training (Stage 1) - default
maskgit3d-train

# With overrides
maskgit3d-train \
    trainer.max_epochs=10 \
    task.lr_g=1e-4 \
    task.lr_d=1e-4 \
    data.batch_size=4 \
    data.num_workers=0

# Resume from checkpoint
maskgit3d-train ckpt_path=./checkpoints/last.ckpt

# MaskGIT training (Stage 2, requires VQVAE checkpoint)
maskgit3d-train \
    task=maskgit \
    task.vqvae_ckpt_path=./checkpoints/vqvae.ckpt \
    trainer.max_epochs=10
```

### Evaluation

```bash
# Validation from checkpoint
maskgit3d-test ckpt_path=./checkpoints/model.ckpt

# Test mode
maskgit3d-test \
    task=maskgit \
    ckpt_path=./checkpoints/model.ckpt \
    mode=test
```

### Testing

```bash
# Run all tests
pytest --cache-clear -vv tests

# Specific test file
pytest tests/unit/tasks/test_vqvae_task.py -v

# Skip slow tests
pytest -m "not slow" tests
```

### Formatting

```bash
# Format code
ruff format src/maskgit3d/ tests/
ruff check --fix src/maskgit3d/ tests/
```

## Architecture

### Current Project Structure

```
src/maskgit3d/
├── conf/               # Hydra configs (inside package for editable install)
│   ├── task/           # Task configs (vqvae.yaml, maskgit.yaml)
│   ├── model/          # Model configs (not directly wired)
│   ├── data/           # Data configs (medmnist3d.yaml, brats.yaml)
│   ├── callbacks/      # Callback configs
│   └── trainer/        # Trainer configs
├── tasks/              # LightningModule implementations
│   ├── vqvae_task.py   # VQVAETask (Stage 1)
│   └── maskgit_task.py # MaskGITTask (Stage 2)
├── models/             # Model architectures
│   ├── vqvae/          # VQVAE (encoder, quantizer, decoder)
│   ├── maskgit/        # MaskGIT transformer
│   └── discriminator/  # Discriminator for GAN
├── training/           # Training step orchestration
│   ├── vqvae_steps.py  # VQVAETrainingSteps
│   ├── maskgit_steps.py# MaskGITTrainingSteps
│   └── gan_strategy.py # GANTrainingStrategy
├── data/               # DataModules
│   ├── medmnist/       # MedMNIST3DDataModule
│   └── brats/          # BraTSDataModule
├── losses/             # Loss implementations
│   └── vq_perceptual_loss.py
├── callbacks/          # Lightning callbacks
├── runtime/            # Builder pattern
│   └── composition.py  # build_vqvae_task, build_maskgit_task
├── interfaces/         # Protocols
├── train.py            # CLI entry: maskgit3d-train
└── eval.py             # CLI entry: maskgit3d-test
```

### Key Components

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `VQVAETask` | Class | `tasks/vqvae_task.py` | Stage 1 training (VQVAE) |
| `MaskGITTask` | Class | `tasks/maskgit_task.py` | Stage 2 training (MaskGIT) |
| `build_vqvae_task` | Function | `runtime/composition.py` | Constructs VQVAETask from config |
| `build_maskgit_task` | Function | `runtime/composition.py` | Constructs MaskGITTask from config |
| `VQVAE` | Class | `models/vqvae/vqvae.py` | Encoder + Quantizer + Decoder |
| `VQVAETrainingSteps` | Class | `training/vqvae_steps.py` | Training logic abstraction |
| `GANTrainingStrategy` | Class | `training/gan_strategy.py` | Optimizer stepping with gradient clipping |
| `VQPerceptualLoss` | Class | `losses/vq_perceptual_loss.py` | L1 + perceptual + VQ + GAN loss |
| `MedMNIST3DDataModule` | Class | `data/medmnist/datamodule.py` | MedMNIST-3D data loading |

### Builder Pattern

Tasks are constructed via builder functions in `runtime/composition.py`:

```python
def build_vqvae_task(cfg: DictConfig) -> VQVAETask:
    model = create_vqvae_model(cfg.model)
    loss_fn = VQPerceptualLoss(...)
    training_steps = VQVAETrainingSteps(...)
    return VQVAETask(model=model, loss_fn=loss_fn, ...)
```

### Training Steps Abstraction

Training logic is separated from Lightning tasks:
- `VQVAETrainingSteps.training_step()` - Full training loop
- `VQVAETask.training_step()` delegates to training steps
- Enables independent testing of training logic

## VQVAE Pipeline

### Training Flow

```
Hydra Config (conf/task/vqvae.yaml)
    ↓
Builder Pattern (runtime/composition.py)
    ├─> VQVAE model
    ├─> VQPerceptualLoss
    ├─> GANTrainingStrategy
    └─> VQVAETrainingSteps
    ↓
Data Loading (MedMNIST3DDataModule)
    ↓
Training Loop
    ├─> Generator: vqvae(x) → loss_g → opt_g.step()
    └─> Discriminator: x_recon.detach() → loss_d → opt_d.step()
```

### Key Configuration (conf/task/vqvae.yaml)

```yaml
# Learning rates
lr_g: 1.0e-4
lr_d: 1.0e-4

# Loss weights
lambda_l1: 1.0
lambda_vq: 1.0
lambda_gan: 0.1
lambda_perceptual: 0.1

# Discriminator warmup
disc_start: 2000
use_adaptive_weight: true
adaptive_weight_max: 100.0

# Gradient clipping
gradient_clip_enabled: true
gradient_clip_val: 1.0
```

## Conventions

### Config Location
- Hydra configs in `src/maskgit3d/conf/` (inside package, not project root)
- Required for editable install via `config_path="conf"`

### Task-Based Config
- `conf/task/*.yaml` defines complete task config
- Model/optimizer configs exist but not directly wired to train.py
- Construction via builder functions only

### Manual Optimization
- VQVAETask uses `automatic_optimization = False` for GAN training
- Alternating generator/discriminator steps per batch

## Important Notes

1. **Stage Dependencies**: Stage 2 (MaskGIT) requires Stage 1 (VQVAE) checkpoint
2. **Data Path**: Use absolute paths (Hydra doesn't expand `~`)
3. **Crop Size**: Must be divisible by 16 for VQVAE encoder
4. **Checkpoints**: Saved to `./checkpoints/` by default

## Testing

- Coverage requirement: >80%
- Test markers: `@pytest.mark.slow`, `@pytest.mark.gpu`, `@pytest.mark.integration`
- Unit tests: `tests/unit/` (mirrors src structure)
- Integration tests: `tests/integration/`

## References

- **Primary**: AGENTS.md (current architecture)
- **Outdated**: This file previously described Fabric + injector architecture (removed)
