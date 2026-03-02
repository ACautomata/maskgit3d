# CLI Usage Guide

## Installation

After installing the package, the following CLI commands will be available:

```bash
pip install -e .
```

Available commands:
- `maskgit3d-train` - Training command
- `maskgit3d-test` - Testing command
- `maskgit3d` - Main CLI with subcommands

## Quick Start

### Training

```bash
# Train with default configuration
maskgit3d-train

# Train with specific model and dataset
maskgit3d-train model=maskgit dataset=medmnist3d

# Train VQGAN with BraTS dataset
maskgit3d-train model=vqgan dataset=brats

# Override specific parameters
maskgit3d-train model=maskgit dataset=medmnist3d training.num_epochs=50 dataset.batch_size=8

# Use experiment configuration
maskgit3d-train --config-name=experiment/maskgit_medmnist
```

### Testing

```bash
# Test with a checkpoint
maskgit3d-test model=maskgit dataset=medmnist3d checkpoint.load_from=./checkpoints/best.ckpt

# Test and save predictions
maskgit3d-test model=vqgan dataset=brats checkpoint.load_from=./checkpoints/best.ckpt output.save_predictions=true
```

## Configuration Structure

The configuration is organized hierarchically:

```
conf/
├── config.yaml          # Main configuration
├── model/
│   ├── maskgit.yaml     # MaskGIT model config
│   ├── vqgan.yaml       # VQGAN model config
│   └── maisi_vq.yaml    # MAISI VQ config
├── dataset/
│   ├── medmnist3d.yaml  # MedMNIST 3D dataset
│   ├── brats.yaml       # BraTS dataset
│   └── simple.yaml      # Simple/synthetic dataset
├── training/
│   └── default.yaml     # Training configuration
├── system/
│   └── default.yaml     # System configuration
└── experiment/
    ├── maskgit_medmnist.yaml  # Experiment presets
    ├── vqgan_brats.yaml
    └── maisi_simple.yaml
```

## Switching Datasets

To switch between datasets, simply change the `dataset` parameter:

```bash
# Use MedMNIST 3D
maskgit3d-train dataset=medmnist3d

# Use BraTS
maskgit3d-train dataset=brats

# Use simple synthetic data
maskgit3d-train dataset=simple
```

You can also create custom dataset configurations in `conf/dataset/`.

## Configuration Overrides

Hydra allows flexible configuration overrides from command line:

```bash
# Change batch size
maskgit3d-train dataset.batch_size=16

# Change learning rate
maskgit3d-train training.optimizer.lr=1.0e-3

# Change number of epochs
maskgit3d-train training.num_epochs=200

# Change device
maskgit3d-train system.device=cuda:0

# Enable Fabric for distributed training
maskgit3d-train training.fabric.enabled=true training.fabric.devices=2

# Multiple overrides
maskgit3d-train model=maskgit dataset=brats training.num_epochs=100 system.seed=123
```

## Working with Experiment Presets

Experiment presets combine multiple configurations:

```bash
# Use experiment preset
maskgit3d-train --config-name=experiment/maskgit_medmnist

# Override experiment preset
maskgit3d-train --config-name=experiment/vqgan_brats training.num_epochs=300
```

## Output Structure

Hydra automatically creates output directories:

```
outputs/
├── 2025-03-02/
│   ├── 10-30-15_maskgit_medmnist3d/
│   │   ├── .hydra/           # Hydra configuration snapshots
│   │   ├── checkpoints/      # Model checkpoints
│   │   └── logs/             # Training logs
│   └── 11-45-22_vqgan_brats/
└── multirun/                 # For sweeps and hyperparameter tuning
```

## Advanced Usage

### Resume Training

```bash
maskgit3d-train checkpoint.resume_from=./checkpoints/checkpoint_epoch_50.ckpt
```

### Distributed Training with Fabric

```bash
# Multi-GPU training
maskgit3d-train training.fabric.enabled=true training.fabric.devices=2 training.fabric.strategy=ddp

# Mixed precision training
maskgit3d-train training.fabric.enabled=true training.fabric.precision=16-mixed
```

### Custom Configuration Directory

```bash
maskgit3d --config-dir=/path/to/custom/configs train model=my_model
```

## Creating Custom Configurations

### Custom Dataset Configuration

Create `conf/dataset/my_dataset.yaml`:

```yaml
# @package _global_

dataset:
  name: my_dataset
  type: simple
  data_dir: ./data/my_dataset
  batch_size: 4
  image_size: 64
  in_channels: 1
  # ... other parameters
```

Then use it:

```bash
maskgit3d-train dataset=my_dataset
```

### Custom Model Configuration

Create `conf/model/my_model.yaml`:

```yaml
# @package _global_

model:
  name: my_model
  type: maskgit
  in_channels: 1
  image_size: 64
  # ... other parameters
```

## Tips

1. **Start Simple**: Use `dataset=simple` for quick testing
2. **Check Config**: Use `--cfg job` to print resolved configuration
3. **Dry Run**: Test configuration without running training
4. **Version Control**: Track your experiment configurations in git
