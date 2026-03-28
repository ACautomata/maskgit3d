# maskgit-3d — Autonomous Research

This is an autonomous ML research setup for maskgit-3d: 3D medical image generation using VQVAE + MaskGIT with PyTorch Lightning + Hydra.

## Architecture

Two scopes of code, inspired by karpathy/autoresearch:

- **prepare.py scope** (read-only): data loading, evaluation metrics, callbacks, utilities
- **train.py scope** (modifiable): model architectures, training logic, losses, optimizers

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar28`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main/master.
3. **Read the in-scope files**: Read these for full context:
   - Project docs (`README.md`, `CLAUDE.md`, `AGENTS.md`) — project context and architecture
   - **prepare.py scope** (read-only, do NOT modify):
     - `src/maskgit3d/data/` — MedMNIST & BraTS datamodules, datasets, transforms, collators
     - `src/maskgit3d/callbacks/` — FID logging, reconstruction loss, gradient norm, NaN detection, etc.
     - `src/maskgit3d/metrics/` — FID (2.5D InceptionV3), image metrics
     - `src/maskgit3d/inference/` — VQVAEReconstructor (sliding window inference)
     - `src/maskgit3d/interfaces/` — Protocol definitions for models and training
     - `src/maskgit3d/config/` — Config schemas (VQVAEModelConfig, MaskGITModelConfig)
     - `src/maskgit3d/utils/` — Geometry utilities, sliding window helpers
     - `src/maskgit3d/eval.py` — Evaluation entry point
     - `src/maskgit3d/conf/` — All Hydra YAML configs (data, callbacks, trainer, etc.)
     - `src/maskgit3d/runtime/callback_selection.py` — Callback selection logic
     - `src/maskgit3d/runtime/checkpoints.py` — Checkpoint loading
     - `src/maskgit3d/runtime/modules.py` — Module registration
   - **train.py scope** (modifiable):
     - `src/maskgit3d/models/vqvae/` — Encoder, Decoder, VectorQuantizer, FSQQuantizer, VQVAE
     - `src/maskgit3d/models/discriminator/` — PatchDiscriminator3D
     - `src/maskgit3d/models/maskgit/` — MaskGIT, Transformer, scheduling, sampling
     - `src/maskgit3d/models/incontext/` — InContextMaskGIT, sequence builder, tokenizer
     - `src/maskgit3d/tasks/` — VQVAETask, MaskGITTask, InContextTask, GANTrainingStrategy
     - `src/maskgit3d/training/` — VQVAETrainingSteps, MaskGITTrainingSteps, InContextSteps
     - `src/maskgit3d/losses/` — VQPerceptualLoss, PerceptualLoss, MaskWeightedCE
     - `src/maskgit3d/runtime/composition.py` — Builder pattern (build_vqvae_task, etc.)
     - `src/maskgit3d/runtime/model_factory.py` — Model construction from config
     - `src/maskgit3d/runtime/optimizer_factory.py` — GANOptimizerFactory, TransformerOptimizerFactory
     - `src/maskgit3d/runtime/scheduler_factory.py` — Cosine warmup scheduler creation
     - `src/maskgit3d/train.py` — Training entry point
4. **Verify data/environment**: Run `conda activate maskgit3d && python -c "from maskgit3d.data.medmnist.datamodule import MedMNIST3DDataModule; print('OK')"` to verify dependencies. For MedMNIST, data auto-downloads to `${DATA_DIR}` or `./data`.
5. **Initialize results.tsv**: Create `results.tsv` with the header row:
   ```
   commit	val_fid	memory_gb	status	description
   ```
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs via:
```bash
maskgit3d-train --config-name train_medmnist trainer.max_epochs=100
```
Training runs for 100 epochs (~1500 steps at batch_size=16, ~15 steps/epoch on MedMNIST).

**What you CAN do:**
- Modify files in train.py scope:
  - Change model architectures (encoder/decoder channels, attention levels, quantizer type)
  - Adjust training logic (GAN strategy, gradient clipping, optimizer betas)
  - Modify loss functions (lambda weights, adaptive weight calculation, discriminator loss type)
  - Change optimizer/scheduler configuration (LR, weight decay, warmup steps)
  - Modify builder/composition code (how components are wired from config)
  - Change model factory logic
  - Adjust training entry point

**What you CANNOT do:**
- Modify files in prepare.py scope. They are read-only.
- Install new packages. Use only existing dependencies.
- Modify the evaluation harness. The metric code is ground truth.

**The goal: get the lowest val_fid.**
val_fid (FID — Fréchet Inception Distance using 2.5D InceptionV3 features) measures the distributional distance between real and reconstructed images. Lower is better. Current best: 31.29 on MedMNIST 3D.

**Memory** is a soft constraint. Some increase is acceptable for meaningful gains.

**Simplicity criterion**: All else being equal, simpler is better. For this project, typical run-to-run FID variance on MedMNIST is ~1.0 point (small dataset, ~1000 training samples). A 0.5 val_fid improvement that adds 15 lines of hacky code is probably not worth it. A 0.5 val_fid improvement from deleting code is definitely worth keeping. An improvement of ~0 but much simpler code is also worth keeping. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Always establish the baseline first — run training as-is with no modifications.

## Output format

When training finishes, Lightning prints epoch summaries and the FIDCallback logs val_fid at each validation epoch. Example output:
```
Epoch 99: 100%|██████████| 15/15 [00:30<00:00]
Validation: |██████████| 2/2 [00:05<00:00, val_rec_loss=0.078, val_fid=31.29]
`Trainer.fit` stopped: `max_epochs=100` reached.
```

You can extract the key metric from the log file:
```bash
grep "val_fid" run.log | tail -1
```

Additional useful metric extractions:
```bash
# Best FID across all epochs
grep "val_fid" run.log | awk -F'val_fid=' '{print $2}' | awk -F'[,\"]' '{print $1}' | sort -n | head -1

# Final reconstruction loss
grep "val_rec_loss" run.log | tail -1

# Peak GPU memory (from CUDA memory callback)
grep "peak_memory" run.log | tail -1

# Training loss curve (generator)
grep "train/total_loss" run.log | tail -5

# Discriminator loss
grep "train/disc_loss" run.log | tail -5

# Gradient norm
grep "grad_norm" run.log | tail -5

# All training losses at epoch end
grep "train/" run.log | tail -1
```

## Logging results

Log each experiment to `results.tsv` (tab-separated, NOT comma-separated).

Columns:
```
commit	val_fid	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_fid achieved — use 999.0 for crashes
3. peak memory in GB, round to .1f — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of the experiment

Example:
```
commit	val_fid	memory_gb	status	description
a1b2c3d	34.20	6.2	keep	baseline (vqvae_small, lambda_perceptual=0.5, 100 epochs)
b2c3d4e	31.29	6.5	keep	increase perceptual weight to 0.5 + randflip augmentation
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar28`).

LOOP FOREVER:
1. Look at the git state: the current branch/commit
2. Modify files in **train.py scope** with an experimental idea
3. git commit
4. Run the experiment: `maskgit3d-train --config-name train_medmnist trainer.max_epochs=100 > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "val_fid" run.log | tail -1`
6. If results can't be found, the run crashed. Check logs and attempt a fix. If you can't fix after a few attempts, give up.
7. Record the results in `results.tsv` (do NOT commit results.tsv — leave it untracked)
8. If val_fid improved (lower is better), advance the branch
9. If val_fid is equal or worse, git reset back to where you started

**Timeout**: Each experiment should take ~5 minutes per epoch on single GPU. 100 epochs = ~8 hours wall clock. If a run exceeds 12 hours, kill it and treat it as failure.

**Crashes**: If a run crashes (OOM, bug), fix simple issues and re-run. If the idea is fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the experiment loop begins, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the code, try combining previous near-misses, try radical changes. The loop runs until the human interrupts you.

## Research Ideas

### VQVAE Architecture

1. **Deeper encoder, shallower decoder** — Add an extra residual block to the encoder while keeping the decoder unchanged. Asymmetric depth may improve latent quality without increasing reconstruction artifacts.

2. **Attention at multiple levels** — Currently only the deepest level has attention (`attention_levels: [false, false, true]`). Try enabling attention at level 1 as well (`[false, true, true]`) to capture mid-range spatial dependencies in 3D.

3. **Larger embedding dimension with projection** — Increase `embedding_dim` to 512 but add a projection layer (1x1x1 Conv3d) in `quant_conv`/`post_quant_conv` to keep the quantizer input at 256. This gives richer latent representations without changing the codebook.

4. **Residual quantization (multi-codebook)** — Replace the single VQ pass with 2-4 rounds of residual VQ. After the first quantization, compute the residual and quantize again. This dramatically increases effective codebook size without increasing single-codebook size.

5. **Exponential moving average decay tuning** — The current EMA decay is 0.99. Try 0.95 (faster codebook adaptation) or 0.999 (more stable codebook) and measure FID impact. The decay directly controls how quickly the codebook adapts to the data distribution.

6. **Straight-through estimator variants** — Replace `z + (z_q - z).detach()` with the gradient-scaling variant `z_q + (1 - scale) * (z - z_q).detach()` where scale is a learnable or fixed parameter (e.g., 0.5). This can improve gradient flow through the quantization bottleneck.

### Loss Function

7. **GAN loss type ablation** — Current discriminator uses hinge loss. Try `vanilla` (LSGAN-style) loss, which may produce smoother gradients for 3D patch discrimination. The `VQPerceptualLoss` already supports this via `disc_loss: vanilla`.

8. **Adaptive weight clamping** — The adaptive weight max is 100.0. Try lower values (10.0, 30.0) to prevent the GAN loss from dominating reconstruction early in training. Also try removing the clamp entirely.

9. **Progressive discriminator warmup** — Instead of `disc_start=150` (hard cutoff), implement a linear warmup: `disc_factor = min(1.0, (global_step - disc_start) / warmup_length)`. This avoids the sharp transition that can destabilize training.

10. **Commitment cost scheduling** — Start with high commitment_cost (0.5) and decay it linearly over training. Early strong commitment forces the encoder to produce vectors near codebook entries; later relaxation allows finer-grained reconstruction.

### Optimizer & Training Dynamics

11. **Separate discriminator LR schedule** — Currently both generator and discriminator share the same cosine warmup schedule. Try a higher LR for the discriminator (3x generator LR) with its own warmup, which can improve the discriminator's ability to provide useful gradients.

12. **Gradient penalty on discriminator** — Add a small R1 gradient penalty (gradient penalty on real images only) to stabilize GAN training. `loss_d += gamma * grad_penalty`. This is a common stabilization technique for patch discriminators.

13. **Accumulated generator steps** — Try `accumulate_grad_batches=2` with halved batch_size, effectively doubling the effective batch size for generator updates without increasing memory.

## File Scope Reference

### prepare.py scope (DO NOT MODIFY)
| Directory/File | Role |
|---|---|
| `src/maskgit3d/data/medmnist/` | MedMNIST 3D datamodule, dataset, downloader, transforms, config, validators |
| `src/maskgit3d/data/brats/` | BraTS 2023 datamodule, dataset, transforms, config |
| `src/maskgit3d/data/collators/` | InContext sample list collator, incontext collator |
| `src/maskgit3d/data/transforms.py` | General data transforms |
| `src/maskgit3d/callbacks/fid_logging.py` | FID callback for validation (2.5D InceptionV3) |
| `src/maskgit3d/callbacks/reconstruction_loss.py` | L1 reconstruction loss callback |
| `src/maskgit3d/callbacks/cuda_memory.py` | CUDA memory monitoring |
| `src/maskgit3d/callbacks/gradient_norm.py` | Gradient norm monitoring |
| `src/maskgit3d/callbacks/mask_accuracy.py` | MaskGIT mask accuracy callback |
| `src/maskgit3d/callbacks/masked_cross_entropy.py` | Masked cross-entropy callback |
| `src/maskgit3d/callbacks/nan_detection.py` | NaN detection callback |
| `src/maskgit3d/callbacks/sample_saving.py` | Sample image saving callback |
| `src/maskgit3d/callbacks/training_stability.py` | Training stability monitoring |
| `src/maskgit3d/callbacks/training_time.py` | Training time tracking |
| `src/maskgit3d/callbacks/vqvae_training_losses.py` | VQVAE training losses logging |
| `src/maskgit3d/metrics/fid.py` | FID metric with 2.5D InceptionV3 feature extraction |
| `src/maskgit3d/metrics/image_metrics.py` | Image metric utilities |
| `src/maskgit3d/inference/reconstruction.py` | VQVAEReconstructor (sliding window) |
| `src/maskgit3d/interfaces/models.py` | Model protocol definitions |
| `src/maskgit3d/interfaces/training.py` | Training protocol definitions |
| `src/maskgit3d/config/schemas.py` | Config schema validation (VQVAEModelConfig, MaskGITModelConfig) |
| `src/maskgit3d/utils/geometry.py` | Geometry utilities |
| `src/maskgit3d/utils/sliding_window.py` | Sliding window inference helpers |
| `src/maskgit3d/eval.py` | Evaluation entry point (maskgit3d-test) |
| `src/maskgit3d/conf/` | All Hydra YAML configs |
| `src/maskgit3d/runtime/callback_selection.py` | Callback selection logic |
| `src/maskgit3d/runtime/checkpoints.py` | Checkpoint loading |
| `src/maskgit3d/runtime/modules.py` | Module registration |

### train.py scope (MODIFIABLE)
| Directory/File | Role |
|---|---|
| `src/maskgit3d/models/vqvae/encoder.py` | 3D encoder (MONAI MaisiEncoder wrapper) |
| `src/maskgit3d/models/vqvae/decoder.py` | 3D decoder (MONAI MaisiDecoder wrapper) |
| `src/maskgit3d/models/vqvae/quantizer.py` | VectorQuantizer with EMA codebook updates |
| `src/maskgit3d/models/vqvae/fsq.py` | FSQ (Finite Scalar Quantization) quantizer |
| `src/maskgit3d/models/vqvae/vqvae.py` | VQVAE model (encoder + quantizer + decoder) |
| `src/maskgit3d/models/vqvae/protocol.py` | Quantizer protocol definition |
| `src/maskgit3d/models/vqvae/splitting.py` | Channel splitting for data parallelism |
| `src/maskgit3d/models/discriminator/patch_discriminator.py` | 3D Patch Discriminator for GAN loss |
| `src/maskgit3d/models/maskgit/maskgit.py` | MaskGIT model (VQVAE tokenizer + transformer) |
| `src/maskgit3d/models/maskgit/transformer.py` | Bidirectional transformer for masked prediction |
| `src/maskgit3d/models/maskgit/scheduling.py` | Mask ratio scheduling (cosine, linear, etc.) |
| `src/maskgit3d/models/maskgit/sampling.py` | Iterative decoding sampler |
| `src/maskgit3d/models/incontext/incontext_maskgit.py` | InContext MaskGIT model |
| `src/maskgit3d/models/incontext/sequence_builder.py` | Token sequence builder |
| `src/maskgit3d/models/incontext/tokenizer.py` | InContext tokenizer |
| `src/maskgit3d/models/incontext/transformer.py` | InContext transformer |
| `src/maskgit3d/tasks/base_task.py` | Base Lightning task |
| `src/maskgit3d/tasks/vqvae_task.py` | VQVAE training task (manual optimization, GAN) |
| `src/maskgit3d/tasks/maskgit_task.py` | MaskGIT training task (automatic optimization) |
| `src/maskgit3d/tasks/incontext_task.py` | InContext MaskGIT training task |
| `src/maskgit3d/tasks/gan_training_strategy.py` | GAN optimizer stepping + gradient clipping |
| `src/maskgit3d/tasks/output_contracts.py` | Typed output dictionaries for training steps |
| `src/maskgit3d/training/vqvae_steps.py` | VQVAE training step logic |
| `src/maskgit3d/training/maskgit_steps.py` | MaskGIT training step logic |
| `src/maskgit3d/training/incontext_steps.py` | InContext training step logic |
| `src/maskgit3d/losses/vq_perceptual_loss.py` | VQPerceptualLoss (L1 + perceptual + VQ + GAN) |
| `src/maskgit3d/losses/perceptual_loss.py` | Perceptual loss (LPIPS via MONAI) |
| `src/maskgit3d/losses/mask_weighted_ce.py` | Mask-weighted cross-entropy loss |
| `src/maskgit3d/runtime/composition.py` | Builder pattern (build_vqvae_task, build_maskgit_task) |
| `src/maskgit3d/runtime/model_factory.py` | Model construction from config |
| `src/maskgit3d/runtime/optimizer_factory.py` | GANOptimizerFactory, TransformerOptimizerFactory |
| `src/maskgit3d/runtime/scheduler_factory.py` | Cosine warmup scheduler creation |
| `src/maskgit3d/train.py` | Training entry point (maskgit3d-train) |
