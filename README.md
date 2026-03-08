# maskgit-3d

PyTorch + Lightning + MONAI based 3D medical image generation project.

The current codebase is organized around Lightning tasks, DataModules, and Hydra config composition. It does not use the older dependency-injection / Fabric layout that earlier docs described.

## Current Status

- Stage 1 is implemented as `VQVAETask` in `src/maskgit3d/tasks/vqvae_task.py`
- Stage 2 is implemented as `MaskGITTask` in `src/maskgit3d/tasks/maskgit_task.py`
- Training entrypoint: `src/maskgit3d/train.py`
- Evaluation entrypoint: `src/maskgit3d/eval.py`
- Default dataset wiring: `src/maskgit3d/data/medmnist/datamodule.py`

## Install

```bash
conda create -n maskgit3d python=3.10 -y
conda activate maskgit3d

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

For development:

```bash
pip install -e .[dev]
```

## Project Layout

```text
maskgit-3d/
├── src/maskgit3d/
│   ├── conf/          # Hydra config groups
│   ├── data/          # datasets, transforms, LightningDataModule
│   ├── losses/        # loss implementations
│   ├── metrics/       # evaluation metrics
│   ├── models/        # VQVAE and MaskGIT model code
│   ├── tasks/         # LightningModule training/eval logic
│   ├── train.py       # Hydra + Lightning training entrypoint
│   └── eval.py        # Hydra + Lightning evaluation entrypoint
├── tests/
└── pyproject.toml
```

## Active Config Surface

The entrypoints currently instantiate:

- `cfg.data`
- `cfg.task`
- `cfg.trainer`
- optional `cfg.callbacks`
- optional `cfg.logger`

Current config files live under `src/maskgit3d/conf/`:

- `train.yaml`
- `eval.yaml`
- `task/vqvae.yaml`
- `task/maskgit.yaml`
- `data/medmnist3d.yaml`
- `trainer/default.yaml`

Note: `conf/model/` and `conf/optimizer/` exist in the repo, but `train.py` does not currently thread those groups into task construction. The runtime configuration surface is therefore task-centric today.

## Training

Default VQVAE training:

```bash
maskgit3d-train
```

Override common settings through Hydra:

```bash
maskgit3d-train \
    trainer.max_epochs=10 \
    task.lr_g=1e-4 \
    task.lr_d=1e-4 \
    data.batch_size=4 \
    data.num_workers=0
```

Resume training from a Lightning checkpoint:

```bash
maskgit3d-train ckpt_path=./checkpoints/last.ckpt
```

Train MaskGIT using a pretrained VQVAE checkpoint:

```bash
maskgit3d-train \
    task=maskgit \
    task.vqvae_ckpt_path=./checkpoints/vqvae.ckpt \
    trainer.max_epochs=10 \
    data.batch_size=4
```

## Evaluation

Validation from checkpoint (default `task=vqvae`):

```bash
maskgit3d-test ckpt_path=./checkpoints/model.ckpt
```

Validation for a MaskGIT checkpoint:

```bash
maskgit3d-test \
    task=maskgit \
    ckpt_path=./checkpoints/maskgit.ckpt
```

Run test loop instead of validation:

```bash
maskgit3d-test \
    task=maskgit \
    ckpt_path=./checkpoints/model.ckpt \
    mode=test
```

## Data Location

The MedMNIST datamodule reads `data.data_dir` from Hydra config. You can set it either way:

```bash
export DATA_DIR=/absolute/path/to/data
maskgit3d-train
```

or:

```bash
maskgit3d-train data.data_dir=/absolute/path/to/data
```

## Development

Run tests:

```bash
python -m pytest --cache-clear -vv tests
```

Run focused tests without project-wide coverage addopts:

```bash
python -m pytest --override-ini addopts='' tests/unit/tasks/test_maskgit_task.py -q
```

Run the project test target:

```bash
make test
```

Format and lint:

```bash
make format
```

## Architecture Notes

What already lines up well with Lightning/Hydra best practice:

- `train.py` and `eval.py` stay thin and only assemble runtime objects
- `tasks/` contains LightningModule logic
- `models/` contains model implementations
- `data/` contains DataModule and data pipeline logic

Current caveats:

- `MaskGITTask` and `VQVAETask` still own most optimizer and model-construction details directly
- `conf/model/` and `conf/optimizer/` are not fully wired into `train.py` yet
- the config story is therefore partially split between Hydra groups and task constructor defaults

If you continue evolving the architecture, the next useful step is to make task construction consume explicit model and optimizer config groups instead of keeping those parameters only inside task YAML files.
