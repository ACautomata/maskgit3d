# maskgit-3d

A PyTorch + Lightning + MONAI deep learning framework with dependency injection architecture.

## Installation

```bash
conda create -n maskgit3d python=3.10 -y
conda activate maskgit3d
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pytorch-lightning monai injector
pip install -e .
```

## Quick Start

```python
from injector import Injector
from maskgit3d.config.modules import create_segmentation_module
from maskgit3d.application.pipeline import TrainingPipeline

# Create module with default config
module = create_segmentation_module(
    in_channels=1,
    out_channels=2,
    spatial_size=(64, 64, 64),
    batch_size=4,
    lr=1e-4,
)

# Create injector
injector = Injector([module])

# Get pipeline
pipeline = injector.get(TrainingPipeline)

# Train
pipeline.run(num_epochs=10)
```
