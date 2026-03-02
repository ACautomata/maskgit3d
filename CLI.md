# CLI Usage Guide

The `maskgit3d` package provides command-line interfaces for training and testing 3D medical image segmentation models using Hydra configuration management.

## Installation

```bash
pip install -e .
```

This installs three CLI commands:
- `maskgit3d` - Main CLI with subcommands
- `maskgit3d-train` - Direct training command
- `maskgit3d-test` - Direct testing command

## Quick Start

### Training

```bash
# Train with default config (maskgit model, simple dataset)
maskgit3d train

# Train with specific model and dataset
maskgit3d train model=maskgit dataset=medmnist3d

# Train VQGAN model with BraTS dataset
maskgit3d train model=vqgan dataset=brats

# Override training parameters
maskgit3d train model=maskgit dataset=medmnist3d training.num_epochs=50 training.lr=0.0001
```

### Testing

```bash
# Test with checkpoint
maskgit3d test model=maskgit dataset=medmnist3d checkpoint.load_from=./checkpoints/best.ckpt

# Test with output saving
maskgit3d test model=vqgan dataset=brats output.save_predictions=true output.output_dir=./predictions
```

### View Configuration

```bash
# View resolved config for training
python src/maskgit3d/cli/train.py --cfg job dataset=brats model=vqgan

# View resolved config for testing
python src/maskgit3d/cli/test.py --cfg job dataset=medmnist3d
```

## Configuration Structure

All configurations are stored in the `conf/` directory:

```
conf/
├── config.yaml              # Main configuration file
├── model/
│   ├── maskgit.yaml         # MaskGIT model config
│   ├── vqgan.yaml           # VQGAN model config
│   └── maisi_vq.yaml        # MAISI VQ model config
├── dataset/
│   ├── simple.yaml          # Simple synthetic dataset
│   ├── medmnist3d.yaml      # MedMNIST3D dataset
│   └── brats.yaml           # BraTS dataset
├── training/
│   └── default.yaml         # Training parameters
└── system/
    └── default.yaml         # System settings
```

## Switching Datasets

To switch datasets, use the `dataset=<name>` override:

```bash
# Use simple synthetic dataset
maskgit3d train dataset=simple

# Use MedMNIST3D dataset
maskgit3d train dataset=medmnist3d

# Use BraTS dataset
maskgit3d train dataset=brats
```

## Adding Custom Datasets

1. Create a new YAML file in `conf/dataset/`:

```yaml
# conf/dataset/my_dataset.yaml
name: my_dataset
type: my_dataset
data_dir: ./data/my_dataset
batch_size: 2
num_workers: 4
image_size: 128
in_channels: 1

# Dataset-specific parameters
custom_param: value
```

2. Use it with:

```bash
maskgit3d train dataset=my_dataset
```

## Configuration Overrides

Hydra allows overriding any config value from command line:

```bash
# Override model parameters
maskgit3d train model.codebook_size=2048 model.embed_dim=512

# Override training parameters
maskgit3d train training.num_epochs=200 training.batch_size=8

# Override system parameters
maskgit3d train system.seed=123 system.device=cuda

# Multiple overrides
maskgit3d train model=maskgit dataset=brats training.num_epochs=50 training.lr=0.0001
```

## Python API

You can also use the training/testing pipelines directly in Python:

```python
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from injector import Injector

from maskgit3d.cli.train import create_module_from_config
from maskgit3d.application.pipeline import TrainingPipeline

# Initialize Hydra
with initialize_config_dir(config_dir="./conf", version_base=None):
    # Compose config
    cfg = compose(
        config_name="config",
        overrides=["model=maskgit", "dataset=medmnist3d"]
    )
    
    # Create DI module
    module = create_module_from_config(cfg)
    
    # Create injector and get pipeline
    injector = Injector([module])
    pipeline = injector.get(TrainingPipeline)
    
    # Run training
    pipeline.run(num_epochs=cfg.training.num_epochs)
```

## Environment Variables

- `HYDRA_FULL_ERROR=1` - Show full stack traces for Hydra errors
- `CUDA_VISIBLE_DEVICES=0` - Specify GPU device

## Troubleshooting

### Issue: `Primary config module 'conf' not found`

Make sure you're running the command from the project root directory where the `conf/` folder exists.

### Issue: Python 3.14 compatibility

If you encounter issues with Python 3.14, use Python 3.12 instead:

```bash
conda create -n maskgit3d_py312 python=3.12
conda activate maskgit3d_py312
pip install -e .
```
