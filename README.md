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

## One-Click Training & Testing Commands

All commands below use the `maskgit3d-train` / `maskgit3d-test` CLI entry points powered by [Hydra](https://hydra.cc/).

> **Prerequisites**: conda environment activated (`conda activate maskgit3d`) and package installed (`pip install -e .`).

---

### Stage 1 — VQGAN Training

Train the VQ-VAE/VQGAN encoder-decoder with a vector-quantised codebook.

```bash
# Default: MedMNIST 3D dataset
maskgit3d-train model=vqgan dataset=medmnist3d

# With BraTS dataset
maskgit3d-train model=vqgan dataset=brats dataset.data_dir=/path/to/brats

# Custom epochs & learning rate
maskgit3d-train model=vqgan dataset=medmnist3d \
    training.num_epochs=100 \
    training.optimizer.lr=1e-4 \
    dataset.batch_size=4

# Resume from a checkpoint
maskgit3d-train model=vqgan dataset=medmnist3d \
    checkpoint.resume_from=./checkpoints/checkpoint_epoch_50.ckpt

# Mixed-precision training (recommended for GPU)
maskgit3d-train model=vqgan dataset=medmnist3d \
    training.fabric.precision=16-mixed
```

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.codebook_size` | `1024` | VQ codebook vocabulary size |
| `model.embed_dim` | `256` | Embedding dimension |
| `model.latent_channels` | `256` | Latent space channels |
| `training.num_epochs` | `100` | Number of training epochs |
| `training.optimizer.lr` | `1e-4` | Learning rate |
| `dataset.batch_size` | `4` | Batch size |
| `checkpoint.save_dir` | `./checkpoints` | Checkpoint save directory |

---

### Stage 1 — VQGAN Testing

Evaluate a trained VQGAN checkpoint. Outputs reconstruction metrics, TensorBoard visualisations, and `.nii.gz` sample files.

```bash
# Basic evaluation
maskgit3d-test model=vqgan dataset=medmnist3d \
    checkpoint.load_from=./checkpoints/vqgan/best.ckpt

# Save reconstructed volumes as .nii.gz for offline inspection
maskgit3d-test model=vqgan dataset=medmnist3d \
    checkpoint.load_from=./checkpoints/vqgan/best.ckpt \
    output.export_nifti=true \
    output.output_dir=./outputs/vqgan_test

# Enable TensorBoard sample visualisation
maskgit3d-test model=vqgan dataset=medmnist3d \
    checkpoint.load_from=./checkpoints/vqgan/best.ckpt \
    output.enable_tensorboard=true \
    output.tensorboard_dir=./outputs/vqgan_test/tensorboard

# Full test with all outputs
maskgit3d-test model=vqgan dataset=brats \
    checkpoint.load_from=./checkpoints/vqgan/best.ckpt \
    output.save_predictions=true \
    output.export_nifti=true \
    output.enable_tensorboard=true \
    output.output_dir=./outputs/vqgan_brats_test
```

**View TensorBoard logs:**

```bash
tensorboard --logdir=./outputs/vqgan_test/tensorboard
# Then open http://localhost:6006 in your browser
```

---

### Stage 2 — MaskGit Training

Train the MaskGit transformer. **Requires a pretrained Stage 1 VQGAN checkpoint.**

```bash
# Train MaskGit with pretrained VQGAN (VQGAN frozen by default)
maskgit3d-train model=maskgit dataset=medmnist3d \
    model.pretrained_vqgan_path=./checkpoints/vqgan/best.ckpt

# Fine-tune with VQGAN unfrozen (end-to-end)
maskgit3d-train model=maskgit dataset=medmnist3d \
    model.pretrained_vqgan_path=./checkpoints/vqgan/best.ckpt \
    model.freeze_vqgan=false

# Custom transformer configuration
maskgit3d-train model=maskgit dataset=medmnist3d \
    model.pretrained_vqgan_path=./checkpoints/vqgan/best.ckpt \
    model.transformer_hidden=768 \
    model.transformer_layers=12 \
    model.transformer_heads=12 \
    training.num_epochs=200 \
    training.optimizer.lr=1e-4

# With BraTS dataset
maskgit3d-train model=maskgit dataset=brats \
    model.pretrained_vqgan_path=./checkpoints/vqgan/best.ckpt \
    dataset.data_dir=/path/to/brats

# Resume training
maskgit3d-train model=maskgit dataset=medmnist3d \
    model.pretrained_vqgan_path=./checkpoints/vqgan/best.ckpt \
    checkpoint.resume_from=./checkpoints/maskgit/checkpoint_epoch_50.ckpt
```

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.pretrained_vqgan_path` | `null` | **Required**: path to Stage 1 checkpoint |
| `model.freeze_vqgan` | `true` | Freeze VQGAN weights during Stage 2 training |
| `model.transformer_hidden` | `768` | Transformer hidden dimension |
| `model.transformer_layers` | `12` | Number of transformer layers |
| `model.transformer_heads` | `12` | Number of attention heads |
| `model.mask_ratio` | `0.5` | Masking ratio for MaskGit training |

---

### Stage 2 — MaskGit Testing

Evaluate a trained MaskGit checkpoint. Outputs generation/reconstruction metrics, TensorBoard visualisations, and `.nii.gz` samples.

```bash
# Basic evaluation
maskgit3d-test model=maskgit dataset=medmnist3d \
    checkpoint.load_from=./checkpoints/maskgit/best.ckpt

# Save generated volumes as .nii.gz
maskgit3d-test model=maskgit dataset=medmnist3d \
    checkpoint.load_from=./checkpoints/maskgit/best.ckpt \
    output.export_nifti=true \
    output.output_dir=./outputs/maskgit_test

# Enable TensorBoard sample visualisation
maskgit3d-test model=maskgit dataset=medmnist3d \
    checkpoint.load_from=./checkpoints/maskgit/best.ckpt \
    output.enable_tensorboard=true \
    output.tensorboard_dir=./outputs/maskgit_test/tensorboard

# Full test with all outputs
maskgit3d-test model=maskgit dataset=brats \
    checkpoint.load_from=./checkpoints/maskgit/best.ckpt \
    output.save_predictions=true \
    output.export_nifti=true \
    output.enable_tensorboard=true \
    output.output_dir=./outputs/maskgit_brats_test
```

---

## Common Notes & Caveats

### GPU Selection

```bash
# Use a specific GPU
CUDA_VISIBLE_DEVICES=0 maskgit3d-train model=vqgan dataset=medmnist3d

# Multi-GPU with DDP
maskgit3d-train model=vqgan dataset=medmnist3d \
    training.fabric.devices=2 \
    training.fabric.strategy=ddp
```

### Output Directory Layout

Hydra automatically timestamps each run:

```
outputs/
└── 2025-03-02/
    └── 10-30-15_vqgan_medmnist3d/
        ├── .hydra/           # Hydra config snapshot
        ├── checkpoints/      # Saved checkpoints
        ├── tensorboard/      # TensorBoard event files
        └── predictions/      # nii.gz / npy outputs
```

### NIfTI Output Files

When `output.export_nifti=true`, the following files are saved per batch:

```
outputs/
├── predictions_batch_0.nii.gz    # Predicted masks / reconstructions
├── probabilities_batch_0.nii.gz  # Predicted probabilities
└── ...
```

Open `.nii.gz` files with [ITK-SNAP](http://www.itksnap.org), [3D Slicer](https://www.slicer.org/), or nibabel:

```python
import nibabel as nib
img = nib.load("predictions_batch_0.nii.gz")
data = img.get_fdata()  # shape: (D, H, W) or (C, D, H, W)
```

### TensorBoard Visualisations

The test pipeline logs the following per batch to TensorBoard:

| Tag | Content |
|-----|---------|
| `test/input` | Input 3D volume (centre slice, greyscale) |
| `test/prediction` | Model output (centre slice, greyscale) |
| `test/target` | Ground-truth label (centre slice, greyscale), if available |

---

## Quick Start (Python API)

### Lightning Fabric Training (Distributed/Mixed Precision)

```python
from injector import Injector
from maskgit3d.config.modules import create_maskgit_module
from maskgit3d.application.pipeline import FabricTrainingPipeline
from maskgit3d.domain.interfaces import (
    DataProvider, ModelInterface, OptimizerFactory, TrainingStrategy,
)

module = create_maskgit_module(
    in_channels=1,
    image_size=64,
    codebook_size=1024,
    embed_dim=256,
    latent_channels=256,
    lr=1e-4,
)
injector = Injector([module])

pipeline = FabricTrainingPipeline(
    model=injector.get(ModelInterface),
    data_provider=injector.get(DataProvider),
    training_strategy=injector.get(TrainingStrategy),
    optimizer_factory=injector.get(OptimizerFactory),
    accelerator="cuda",
    devices=2,
    strategy="ddp",
    precision="16-mixed",
)
pipeline.run(num_epochs=100)
```

### Training MaskGit with Pretrained VQGAN

```bash
# Train MaskGit with pretrained VQGAN (frozen by default)
maskgit3d-train model=maskgit model.pretrained_vqgan_path=./checkpoints/vqgan/best.ckpt

# Fine-tune without freezing VQGAN
maskgit3d-train model=maskgit model.pretrained_vqgan_path=./checkpoints/vqgan/best.ckpt model.freeze_vqgan=false
```
