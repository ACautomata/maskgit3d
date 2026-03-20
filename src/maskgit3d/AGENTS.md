# maskgit3d Package

## OVERVIEW

Core package implementing 3D medical image generation with VQVAE + MaskGIT architecture. Uses Lightning tasks, builder pattern construction, and Hydra configuration.

## STRUCTURE

```
maskgit3d/
├── conf/               # Hydra configs (30 YAML files)
├── tasks/              # LightningModules (VQVAETask, MaskGITTask)
├── models/             # Neural network architectures
├── runtime/            # Builders, factories, composition logic
├── training/           # Training step orchestration
├── data/               # DataModules and transforms
├── callbacks/          # Lightning callbacks
├── losses/             # Loss functions
├── interfaces/         # Protocol definitions
├── train.py            # Entry: maskgit3d-train
└── eval.py             # Entry: maskgit3d-test
```

## WHERE TO LOOK

| Component | Directory | Key Files |
|-----------|-----------|-----------|
| Training tasks | `tasks/` | `vqvae_task.py`, `maskgit_task.py` |
| Model architectures | `models/` | `vqvae/vqvae.py`, `maskgit/maskgit.py` |
| Task construction | `runtime/` | `composition.py` (builders) |
| Training steps | `training/` | `vqvae_steps.py`, `maskgit_steps.py` |
| Hydra configs | `conf/` | `train.yaml`, `task/*.yaml` |
| Data loading | `data/` | `medmnist/datamodule.py` |

## KEY PATTERNS

### Task Construction (Builder Pattern)
```python
# runtime/composition.py
task = build_vqvae_task(cfg)  # or build_maskgit_task(cfg)
# Internally: builds model, loss, optimizer, training_steps
```

### Training Steps Abstraction
```python
# Tasks delegate to step classes
class VQVAETrainingSteps:
    def generator_step(self, x_real, vq_loss, ...) -> dict
    def discriminator_step(self, x_real, x_recon, ...) -> dict
```

### Hydra Config Structure
```yaml
# conf/task/vqvae.yaml - single source for task params
defaults:
  - /model@model: vqvae
lr_g: 1e-4
lr_d: 1e-4
# ... all training params
```

## CONVENTIONS

- Configs in `conf/` are packaged with wheel (for editable install)
- Model/optimizer configs exist but construction via builder functions
- Use `instantiate(cfg.data)` for DataModules, not manual construction
- Callbacks configured via `conf/callbacks/` and instantiated in train.py

## ANTI-PATTERNS

- Don't directly instantiate tasks — use `build_*_task()` functions
- Don't assume protocols are enforced — they're advisory contracts
- Don't put training logic in models — belongs in `training/` steps