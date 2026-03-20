# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-19
**Commit:** 10eda4e
**Branch:** task-decoupling-refactor

## OVERVIEW

maskgit-3d is a 3D medical image generation framework using PyTorch + Lightning + MONAI with two-stage training (VQGAN → MaskGIT). Uses Hydra config composition and builder pattern for task construction.

## STRUCTURE

```
maskgit-3d/
├── src/maskgit3d/          # Main package
│   ├── conf/               # Hydra configs (inside package for editable install)
│   ├── tasks/              # LightningModule implementations (VQVAETask, MaskGITTask)
│   ├── models/             # VQVAE, MaskGIT, Discriminator architectures
│   ├── runtime/            # Builder pattern (composition.py), factories
│   ├── training/           # Training step orchestration (VQVAETrainingSteps, etc.)
│   ├── data/               # DataModules (MedMNIST, BraTS)
│   ├── callbacks/          # Lightning callbacks (metrics, stability, sample saving)
│   ├── losses/             # VQ perceptual loss, LPIPS
│   ├── interfaces/         # Protocols (VQTokenizerProtocol, OptimizerFactoryProtocol)
│   ├── train.py            # CLI entry: maskgit3d-train
│   └── eval.py             # CLI entry: maskgit3d-test
├── tests/                  # Unit + integration tests (mirrors src structure)
└── automation/             # Generated automation files (ignore)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add/modify training logic | `src/maskgit3d/tasks/` | VQVAETask, MaskGITTask |
| Add new model architecture | `src/maskgit3d/models/` | vqvae/, maskgit/, discriminator/ |
| Change training steps | `src/maskgit3d/training/` | vqvae_steps.py, maskgit_steps.py |
| Add callbacks | `src/maskgit3d/callbacks/` | Register in `conf/callbacks/` |
| Modify data pipeline | `src/maskgit3d/data/` | MedMNIST, BraTS datamodules |
| Understand task construction | `src/maskgit3d/runtime/composition.py` | Builder functions |
| Add Hydra config | `src/maskgit3d/conf/` | task/, model/, data/, trainer/ |
| Fix tests | `tests/unit/` | Mirrors src structure |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `VQVAETask` | Class | `tasks/vqvae_task.py` | Stage 1 training (autoencoder + VQ) |
| `MaskGITTask` | Class | `tasks/maskgit_task.py` | Stage 2 training (transformer on tokens) |
| `build_vqvae_task` | Function | `runtime/composition.py` | Constructs VQVAETask from Hydra config |
| `build_maskgit_task` | Function | `runtime/composition.py` | Constructs MaskGITTask from Hydra config |
| `VQVAE` | Class | `models/vqvae/vqvae.py` | Encoder + Quantizer + Decoder |
| `MaskGIT` | Class | `models/maskgit/maskgit.py` | Transformer for token prediction |
| `VQPerceptualLoss` | Class | `losses/vq_perceptual_loss.py` | GAN + L1 + VQ + perceptual loss |
| `MedMNISTDataModule` | Class | `data/medmnist/datamodule.py` | MedMNIST-3D data loading |

## CONVENTIONS

### Config Location (Non-Standard)
- Hydra configs in `src/maskgit3d/conf/` (inside package, not project root)
- Required for editable install to find configs via `config_path="conf"`

### Dual Config Strategy
- `conf/task/*.yaml` defines complete task (model + optimizer + training params)
- `conf/model/` and `conf/optimizer/` exist but NOT wired into train.py directly
- Model/optimizer construction via `runtime/composition.py` builder functions

### Training Steps Abstraction
- Training logic split into `training/` classes (VQVAETrainingSteps, MaskGITTrainingSteps)
- Tasks delegate to these step classes instead of inline training_step
- Enables testing training logic independently

### Protocol Usage (Partial)
- `interfaces/` defines protocols (VQTokenizerProtocol, OptimizerFactoryProtocol)
- NOT consistently applied across all components
- Use as contracts when implementing new models/strategies

### Callback Payload Mechanism
- `BaseTask.save_callback_payload()` / `get_callback_payload()`
- Custom mechanism for task-to-callback data flow
- Use in custom callbacks that need training data

## ANTI-PATTERNS (THIS PROJECT)

### DO NOT
- **Never** assume `conf/model/` configs are directly instantiated — use builder functions
- **Never** reference `cli/` directory — it doesn't exist (entrypoints are `train.py`, `eval.py`)
- **Never** use `injector` library — dependency declared but NOT used (use builder pattern)
- **Never** assume Makefile exists — README references it but it's missing

### Configuration Pitfalls
- `dataset.data_dir` must use absolute paths (Hydra doesn't expand `~`)
- `crop_size` must be divisible by 16 for VQVAE encoder
- Stage 2 requires `task.vqvae_ckpt_path` pointing to Stage 1 checkpoint

## UNIQUE STYLES

### Builder Pattern
```python
# In runtime/composition.py
def build_vqvae_task(cfg: DictConfig) -> VQVAETask:
    model = build_vqvae_model(cfg.model)
    loss_fn = build_loss_fn(cfg)
    training_steps = VQVAETrainingSteps(...)
    return VQVAETask(model=model, loss_fn=loss_fn, ...)
```

### Task-Based Config
```yaml
# conf/task/vqvae.yaml defines everything needed
defaults:
  - /model@model: vqvae
  - /optimizer@optimizer: adam
# All params in one place
lr_g: 1e-4
lr_d: 1e-4
lambda_l1: 1.0
```

## COMMANDS

```bash
# Install
pip install -e .
pip install -e .[dev]  # With dev dependencies

# Training
maskgit3d-train                                    # VQVAE with defaults
maskgit3d-train task=maskgit task.vqvae_ckpt_path=...  # MaskGIT

# Testing
pytest --cache-clear -vv tests                    # All tests
pytest tests/unit/tasks/test_vqvae_task.py -v     # Specific file
pytest -m "not slow" tests                        # Skip slow tests

# Formatting (if Makefile created)
ruff format src/maskgit3d/ tests/
ruff check --fix src/maskgit3d/ tests/
```

## NOTES

### Architecture Evolution
- Project migrated from Fabric + dependency injection to Lightning + builder pattern
- Old `config/`, `domain/`, `application/`, `infrastructure/` directories removed
- CLAUDE.md references outdated architecture — trust README.md and this file

### Stage Dependencies
- Stage 1 (VQVAE) must complete before Stage 2 (MaskGIT)
- Stage 2 freezes VQVAE by default
- Checkpoints saved to `./checkpoints/` by default

### Testing Markers
- `@pytest.mark.slow` — Long-running tests
- `@pytest.mark.gpu` — Requires GPU
- `@pytest.mark.integration` — End-to-end tests

### Data Location
- MedMNIST downloads to `~/.medmnist` by default
- Override with `data.data_dir=/absolute/path` in Hydra