# maskgit-3d

A PyTorch + Lightning + MONAI deep learning framework with dependency injection architecture.

## Installation

```bash
conda create -n maskgit3d python=3.10 -y
conda activate maskgit3d
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install lightning monai injector
pip install -e .
```

## Quick Start

### Vanilla PyTorch Training

```python
from injector import Injector
from maskgit3d.config.modules import create_maskgit_module
from maskgit3d.application.pipeline import TrainingPipeline

# Create module with default config
module = create_maskgit_module(
    in_channels=1,
    image_size=64,
    codebook_size=1024,
    embed_dim=256,
    latent_channels=256,
    lr=1e-4,
)

# Create injector
injector = Injector([module])

# Get pipeline
pipeline = injector.get(TrainingPipeline)

# Train
pipeline.run(num_epochs=10)
```

### Lightning Fabric Training (Distributed/Mixed Precision)

```python
from injector import Injector
from maskgit3d.config.modules import create_maskgit_module, create_fabric_pipeline
from maskgit3d.application.pipeline import FabricTrainingPipeline

# Create module with default config
module = create_maskgit_module(
    in_channels=1,
    image_size=64,
    codebook_size=1024,
    embed_dim=256,
    latent_channels=256,
    lr=1e-4,
)

# Create injector
injector = Injector([module])

# Get components
model = injector.get(ModelInterface)
data_provider = injector.get(DataProvider)
training_strategy = injector.get(TrainingStrategy)
optimizer_factory = injector.get(OptimizerFactory)

# Create Fabric pipeline with custom configuration
pipeline = create_fabric_pipeline(
    model=model,
    data_provider=data_provider,
    training_strategy=training_strategy,
    optimizer_factory=optimizer_factory,
    accelerator="cuda",
    devices=2,
    strategy="ddp",
    precision="16-mixed",
)

# Train with distributed training
pipeline.run(num_epochs=10)
```

### Training MaskGit with Pretrained VQGAN

MaskGit (Stage 2) requires pretrained VQGAN weights from Stage 1:

```bash
# Train MaskGit with pretrained VQGAN (frozen by default)
maskgit3d-train model=maskgit model.pretrained_vqgan_path=./checkpoints/vqgan/best.ckpt

# Fine-tune without freezing VQGAN
maskgit3d-train model=maskgit model.pretrained_vqgan_path=./checkpoints/vqgan/best.ckpt model.freeze_vqgan=false
```
