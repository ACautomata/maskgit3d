# maskgit-3d — Autonomous Research

This is an autonomous ML research setup for maskgit-3d: 3D medical image generation using PyTorch + Lightning + MONAI with two-stage training (VQVAE Stage 1 + MaskGIT Stage 2).

## Architecture

Two scopes of code, inspired by karpathy/autoresearch:

- **prepare.py scope** (read-only): data loading, evaluation metrics, callbacks, utilities
- **train.py scope** (modifiable): model architectures, training logic, losses, optimizers

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar27`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main/master.
3. **Read the in-scope files**: Read these for full context:
   - Project docs (`README.md`, `CLAUDE.md`, `AGENTS.md`) — project context and architecture
   - **prepare.py scope** (read-only, do NOT modify):
     - `src/maskgit3d/data/` — MedMNIST3D and BraTS datasets, transforms, datamodules, collators
     - `src/maskgit3d/metrics/` — FID and image quality metrics (SSIM, PSNR)
     - `src/maskgit3d/callbacks/` — Training monitoring (FID logging, gradient norm, NaN detection, etc.)
     - `src/maskgit3d/inference/` — VQVAE reconstruction utilities with sliding window
     - `src/maskgit3d/interfaces/` — Protocol definitions (VQTokenizerProtocol, TokenGeneratorProtocol)
     - `src/maskgit3d/utils/` — Geometry validation, sliding window helpers
     - `src/maskgit3d/eval.py` — Evaluation entry point
     - `src/maskgit3d/runtime/checkpoints.py` — Checkpoint loading utilities
     - `src/maskgit3d/runtime/callback_selection.py` — Callback filtering by task type
     - `src/maskgit3d/config/` — Configuration schemas
     - `src/maskgit3d/conf/` — Hydra YAML configs (all subdirectories)
   - **train.py scope** (modifiable):
     - `src/maskgit3d/models/vqvae/` — VQVAE encoder, decoder, quantizer, FSQ
     - `src/maskgit3d/models/maskgit/` — MaskGIT transformer, sampling, scheduling
     - `src/maskgit3d/models/incontext/` — InContextMaskGIT, tokenizer, sequence builder
     - `src/maskgit3d/models/discriminator/` — 3D patch discriminator
     - `src/maskgit3d/tasks/` — VQVAETask, MaskGITTask, InContextTask (LightningModules)
     - `src/maskgit3d/training/` — VQVAETrainingSteps, MaskGITTrainingSteps, InContextTrainingSteps
     - `src/maskgit3d/losses/` — VQPerceptualLoss, PerceptualLoss, MaskWeightedCrossEntropyLoss
     - `src/maskgit3d/runtime/composition.py` — Task builder functions
     - `src/maskgit3d/runtime/model_factory.py` — Model construction from config
     - `src/maskgit3d/runtime/optimizer_factory.py` — Optimizer/LR scheduler factories
     - `src/maskgit3d/runtime/scheduler_factory.py` — Cosine warmup scheduler
     - `src/maskgit3d/train.py` — Training entry point
4. **Verify data/environment**: Check that MedMNIST data is available. Run:
   ```bash
   conda activate maskgit3d
   python -c "from medmnist import INFO; print('MedMNIST available')"
   ```
   Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`
5. **Initialize results.tsv**: Create `results.tsv` with the header row:
   ```
   commit	val_rec_loss	memory_gb	status	description
   ```
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs via:
```bash
maskgit3d-train task=vqvae trainer.max_epochs=10 data.batch_size=4
```
Training runs for a configured number of epochs (default 10). Override via `trainer.max_epochs`.

For MaskGIT Stage 2 (requires VQVAE checkpoint):
```bash
maskgit3d-train task=maskgit task.vqvae_ckpt_path=./checkpoints/vqvae.ckpt trainer.max_epochs=10
```

**What you CAN do:**
- Modify files in train.py scope:
  - `models/vqvae/` — encoder/decoder architecture, quantizer (VQ/FSQ), channel splitting
  - `models/maskgit/` — transformer architecture, mask scheduling, sampling strategy
  - `models/incontext/` — in-context learning architecture, sequence building
  - `models/discriminator/` — discriminator architecture for GAN
  - `tasks/` — LightningModule training logic, optimizer configuration, gradient strategies
  - `training/` — training step implementations, loss computation, forward/backward logic
  - `losses/` — loss functions, loss weighting, adaptive weighting
  - `runtime/composition.py` — how tasks are assembled from components
  - `runtime/model_factory.py` — model construction logic
  - `runtime/optimizer_factory.py` — optimizer types, hyperparameters
  - `runtime/scheduler_factory.py` — LR schedule design
  - `train.py` — training entry point configuration

**What you CANNOT do:**
- Modify files in prepare.py scope. They are read-only.
- Install new packages. Use only existing dependencies (PyTorch, Lightning, MONAI, etc.).
- Modify the evaluation harness. FID, SSIM, reconstruction loss computation is ground truth.

**The goal: get the lowest val_rec_loss** (for VQVAE Stage 1) or **lowest val_loss** (for MaskGIT Stage 2).

For **VQVAE Stage 1**, the primary metric is `val_rec_loss` — the L1 reconstruction loss on the validation set, computed by the `ReconstructionLossCallback`. Lower is better. Secondary metric: `val_fid` (Fréchet Inception Distance, logged by `FIDCallback`). Lower FID means more realistic reconstructions.

For **MaskGIT Stage 2**, the primary metric is `val_loss` — masked cross-entropy on held-out tokens. Lower is better. Secondary metric: `val_mask_acc` (mask prediction accuracy). Higher is better.

**Memory** is a soft constraint. Some increase is acceptable for meaningful metric gains, but it should not blow up dramatically. 3D medical images are memory-intensive — watch out for OOM.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code and getting equal or better results is a great outcome.

**The first run**: Always establish the baseline first — run training as-is with no modifications.

## Output format

Lightning logs metrics to console (progress bar) and optionally TensorBoard. After each validation epoch, look for:

```
Epoch 0: val_rec_loss=0.123, val_fid=45.6
```

Extract the key metric:
```bash
grep "val_rec_loss" run.log | tail -1
```

For MaskGIT:
```bash
grep "val_loss" run.log | tail -1
```

If using TensorBoard, check:
```bash
tensorboard --logdir lightning_logs/
```

## Logging results

Log each experiment to `results.tsv` (tab-separated, NOT comma-separated).

Columns:
```
commit	val_rec_loss	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_rec_loss achieved — use 99.0 for crashes
3. peak memory in GB, round to .1f — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of the experiment

Example:
```
commit	val_rec_loss	memory_gb	status	description
a1b2c3d	0.150	8.2	keep	baseline
b2c3d4e	0.132	8.5	keep	increase encoder channels from 64 to 128
c3d4e5f	0.155	8.2	discard	switch to FSQ quantizer (4 levels)
d4e5f6g	99.0	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar27`).

LOOP FOREVER:
1. Look at the git state: the current branch/commit
2. Modify files in **train.py scope** with an experimental idea
3. git commit
4. Run the experiment: `maskgit3d-train task=vqvae trainer.max_epochs=10 > run.log 2>&1`
5. Read out the results: `grep "val_rec_loss" run.log | tail -1`
6. If results can't be found, the run crashed. Check logs: `tail -n 50 run.log` and attempt a fix. If you can't fix after a few attempts, give up.
7. Record the results in `results.tsv` (do NOT commit results.tsv — leave it untracked)
8. If val_rec_loss improved (lower is better), advance the branch
9. If val_rec_loss is equal or worse, git reset back to where you started

**Timeout**: Each experiment should take ~10-30 minutes depending on dataset and hardware. If a run exceeds 60 minutes, kill it and treat it as failure.

**Crashes**: If a run crashes (OOM, bug), fix simple issues and re-run. If the idea is fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the experiment loop begins, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the code, try combining previous near-misses, try radical changes. The loop runs until the human interrupts you.

## Research Ideas

1. **Quantizer comparison**: Swap VectorQuantizer for FSQ with different level configurations (e.g., [8,5,5,5] vs [6,5,5,5]) and compare reconstruction quality
2. **Encoder depth**: Increase encoder depth (more residual blocks) and observe effect on reconstruction loss
3. **Channel splitting**: Experiment with `num_splits` parameter to split high-dimensional latent maps into manageable chunks for quantization
4. **Loss weight tuning**: Adjust `lambda_l1`, `lambda_vq`, `lambda_gan`, `lambda_perceptual` — the relative weighting of L1, VQ commitment, adversarial, and perceptual losses
5. **Discriminator warmup**: Change `disc_start` from 2000 to earlier/later — when the discriminator starts training affects GAN stability
6. **Adaptive weight ceiling**: Modify `adaptive_weight_max` (default 100.0) to control how aggressively the GAN loss weight adapts
7. **Learning rate schedule**: Replace ReduceLROnPlateau with cosine annealing, or try different warmup schedules
8. **Gradient checkpointing**: Enable/disable gradient checkpointing to trade memory for compute and potentially increase batch size
9. **Mask schedule**: For MaskGIT Stage 2, try different `gamma_type` (cosine vs linear vs arccos) for mask ratio scheduling
10. **Sliding window inference**: Enable sliding window inference for larger crops and evaluate the impact on reconstruction quality

## File Scope Reference

### prepare.py scope (DO NOT MODIFY)
| Directory/File | Role |
|---|---|
| `data/medmnist/` | MedMNIST3D dataset, datamodule, transforms, downloader, validators |
| `data/brats/` | BraTS2023 dataset, datamodule, transforms, config |
| `data/collators/` | In-context batch collation |
| `data/transforms.py` | Global transform utilities |
| `metrics/fid.py` | Fréchet Inception Distance (2.5D for 3D images) |
| `metrics/image_metrics.py` | SSIM, PSNR metrics |
| `callbacks/fid_logging.py` | FID computation during validation |
| `callbacks/reconstruction_loss.py` | Reconstruction loss logging |
| `callbacks/mask_accuracy.py` | Mask prediction accuracy tracking |
| `callbacks/masked_cross_entropy.py` | Masked CE loss logging |
| `callbacks/vqvae_training_losses.py` | VQVAE loss component logging |
| `callbacks/gradient_norm.py` | Gradient norm monitoring |
| `callbacks/nan_detection.py` | NaN/Inf detection |
| `callbacks/training_stability.py` | Training stability monitoring |
| `callbacks/training_time.py` | Training time tracking |
| `callbacks/cuda_memory.py` | GPU memory monitoring |
| `callbacks/sample_saving.py` | Sample visualization during eval |
| `inference/reconstruction.py` | VQVAE reconstruction with sliding window |
| `interfaces/models.py` | VQTokenizerProtocol, TokenGeneratorProtocol |
| `interfaces/training.py` | OptimizerFactoryProtocol, SchedulerFactoryProtocol |
| `utils/geometry.py` | Crop size validation, padding computation |
| `utils/sliding_window.py` | MONAI sliding window wrapper |
| `eval.py` | Evaluation entry point |
| `runtime/checkpoints.py` | Checkpoint loading utilities |
| `runtime/callback_selection.py` | Callback filtering by task type |
| `config/schemas.py` | Configuration schema definitions |
| `conf/` | Hydra YAML configs (model, task, data, optimizer, etc.) |

### train.py scope (MODIFIABLE)
| Directory/File | Role |
|---|---|
| `models/vqvae/vqvae.py` | Main VQVAE model (encoder + quantizer + decoder) |
| `models/vqvae/encoder.py` | 3D convolutional encoder with residual blocks |
| `models/vqvae/decoder.py` | 3D convolutional decoder with residual blocks |
| `models/vqvae/quantizer.py` | Vector quantizer with EMA codebook updates |
| `models/vqvae/fsq.py` | Finite Scalar Quantizer alternative |
| `models/vqvae/splitting.py` | Channel splitting for large latent maps |
| `models/vqvae/protocol.py` | Quantizer protocol interface |
| `models/maskgit/maskgit.py` | MaskGIT model (frozen VQVAE + transformer) |
| `models/maskgit/transformer.py` | Bidirectional transformer for masked prediction |
| `models/maskgit/sampling.py` | Iterative decoding sampler |
| `models/maskgit/scheduling.py` | Mask ratio scheduler (cosine, linear, arccos) |
| `models/incontext/incontext_maskgit.py` | Multi-modal in-context MaskGIT |
| `models/incontext/tokenizer.py` | Multi-modal tokenizer with modality embeddings |
| `models/incontext/transformer.py` | Variable-length transformer |
| `models/incontext/sequence_builder.py` | Transformer input sequence construction |
| `models/discriminator/patch_discriminator.py` | 3D patch discriminator for GAN |
| `tasks/base_task.py` | Base LightningModule |
| `tasks/vqvae_task.py` | VQVAE training task with GAN optimization |
| `tasks/maskgit_task.py` | MaskGIT training task |
| `tasks/incontext_task.py` | InContextMaskGIT training task |
| `tasks/gan_training_strategy.py` | GAN optimizer stepping with gradient clipping |
| `tasks/output_contracts.py` | TypedDict contracts for step outputs |
| `training/vqvae_steps.py` | VQVAE training step logic |
| `training/maskgit_steps.py` | MaskGIT training step logic |
| `training/incontext_steps.py` | InContext training step logic |
| `losses/vq_perceptual_loss.py` | Combined L1 + perceptual + VQ + GAN loss |
| `losses/perceptual_loss.py` | Perceptual loss with pretrained features |
| `losses/mask_weighted_ce.py` | Masked cross-entropy loss |
| `runtime/composition.py` | Task builder functions (build_vqvae_task, etc.) |
| `runtime/model_factory.py` | Model construction from config |
| `runtime/optimizer_factory.py` | Optimizer factories (AdamW, GAN, Transformer) |
| `runtime/scheduler_factory.py` | LR scheduler factory (cosine warmup) |
| `train.py` | Training entry point |
