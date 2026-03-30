# MaskGIT v3 Implementation Progress (2026-03-26)

## ✅ Completed Work

### Core v3 Implementation (all tests passing)

| Component | File | Tests |
|-----------|------|-------|
| MaskWeightedCrossEntropyLoss | `losses/mask_weighted_ce.py` | 7 |
| InContextTokenizer | `models/incontext/tokenizer.py` | 31 |
| VariableLengthMaskGITTransformer | `models/incontext/transformer.py` | 11 |
| InContextSequenceBuilder | `models/incontext/sequence_builder.py` | 19 |
| InContextMaskGIT Model | `models/incontext/incontext_maskgit.py` | 23 |
| InContextTrainingSteps | `training/incontext_steps.py` | 9 |
| InContextMaskGITTask | `tasks/incontext_task.py` | 13 |
| Hydra Configs | `conf/task/incontext_maskgit.yaml` | - |
| Builder Function | `runtime/composition.py` | 1 |
| Integration Tests | `tests/integration/test_incontext_training.py` | 17 |

### any2one Support

| Component | File | Purpose |
|-----------|------|---------|
| InContextSample | `models/incontext/types.py` | Raw sample dataclass |
| PreparedInContextBatch | `models/incontext/types.py` | Prepared token tensors |
| build_sample() | `sequence_builder.py` | Per-sample sequence building |
| prepare_batch() | `incontext_maskgit.py` | GPU-side encoding + padding |
| compute_loss_from_prepared() | `incontext_maskgit.py` | Token-level loss computation |
| training_step_any2one() | `incontext_steps.py` | Any2one training step |

### Sliding Window Support (Fixed 2026-03-26)

| Component | File | Purpose |
|-----------|------|---------|
| `encode_with_modality()` | `tokenizer.py` | Batch encoding with modality embedding + sliding window |
| `_decode_tokens_to_images()` | `incontext_maskgit.py` | Sliding window decode for large images |
| `_get_sliding_window_inferer()` | `incontext_maskgit.py` | Cached sliding window inferer |

**Critical fixes**:
1. Added `encode_with_modality()` to tokenizer - encodes batch with modality embedding AND sliding window
2. Fixed `prepare_batch()` to use `encode_with_modality()` instead of `encode_images_to_latents()` - ensures modality embeddings are added during any2one training
3. Fixed `generate()` to use `_decode_tokens_to_images()` for sliding window decoding
4. Fixed sliding window decode test to actually trigger the branch (`latent_size > roi_size/downsampling`)

**Encoding path consistency verified**: Both `compute_loss()` and `prepare_batch()` + `compute_loss_from_prepared()` now produce equivalent results.

**In-context test totals**: 17 integration tests + 84+ unit tests in `tests/unit/models/incontext/` (including 5 new `encode_with_modality` tests + 5 sliding window tests)

## 📐 Architecture Decisions

### Token ID Allocation
- Real vocab: `0` to `codebook_size-1`
- CLS: `codebook_size`
- SEP: `codebook_size+1`
- MOD_LABEL_i: `codebook_size+2+i`
- MASK: `codebook_size+2+num_modalities`

### Sequence Format
```
[CLS] [MOD0] [LAT0...] [MOD1] [LAT1...] [MOD_TARGET] [TARGET...] [SEP]
```

### Masking Strategy
- Only target positions are masked
- Context positions stay visible (provide conditioning)
- Mask ratio per sample can vary

### Batch API Evolution
- **Old**: `context_images: list[Tensor]`, `target_image: Tensor` (same structure for all samples)
- **New any2one**: `samples: list[InContextSample]` (different context count per sample)

## 🚀 Usage Commands

### Run Tests
```bash
# Unit tests
python -m pytest tests/unit/models/incontext/ tests/unit/training/test_incontext_steps.py -v

# Integration tests
python -m pytest tests/integration/test_incontext_training.py -v

# Full test suite
python -m pytest tests/ -q --tb=short

# Sliding window tests
python -m pytest tests/unit/models/incontext/test_incontext_maskgit.py::TestWithSlidingWindow -v

# encode_with_modality tests
python -m pytest tests/unit/models/incontext/test_tokenizer.py::TestEncodeWithModality -v
```

### Training
```bash
# Basic training
maskgit3d-train task=incontext_maskgit task.vqvae_ckpt_path=/path/to/vqvae.ckpt

# With sliding window for large images
maskgit3d-train \
  task=incontext_maskgit \
  task.vqvae_ckpt_path=./checkpoints/vqvae.ckpt \
  task.sliding_window.enabled=true \
  task.sliding_window.roi_size=[32,32,32] \
  task.sliding_window.overlap=0.25

# With custom parameters
maskgit3d-train \
  task=incontext_maskgit \
  task.vqvae_ckpt_path=./checkpoints/vqvae.ckpt \
  task.num_modalities=4 \
  task.hidden_size=768 \
  trainer.max_epochs=10
```

### Python API
```python
from maskgit3d.models.incontext.types import InContextSample
from maskgit3d.models.incontext.incontext_maskgit import InContextMaskGIT

# Create samples with different context counts
samples = [
    InContextSample(
        context_images=[t1_image],  # 1 context
        context_modality_ids=[0],
        target_image=t2_image,
        target_modality_id=1,
        mask_ratio=0.5,
    ),
    InContextSample(
        context_images=[t1_image, flair_image],  # 2 contexts
        context_modality_ids=[0, 2],
        target_image=t2_image,
        target_modality_id=1,
        mask_ratio=0.3,
    ),
]

# Prepare batch and compute loss
prepared = model.prepare_batch(samples)
loss, metrics = model.compute_loss_from_prepared(prepared)
```

## 📁 File Structure

```
src/maskgit3d/
├── models/incontext/
│   ├── types.py                    # InContextSample, PreparedInContextBatch
│   ├── tokenizer.py                # InContextTokenizer (+ encode_with_modality)
│   ├── sequence_builder.py         # InContextSequenceBuilder
│   ├── transformer.py              # VariableLengthMaskGITTransformer
│   └── incontext_maskgit.py        # InContextMaskGIT
├── training/
│   └── incontext_steps.py          # InContextTrainingSteps
├── tasks/
│   └── incontext_task.py           # InContextMaskGITTask
├── losses/
│   └── mask_weighted_ce.py         # MaskWeightedCrossEntropyLoss
└── conf/
    ├── task/incontext_maskgit.yaml
    └── model/incontext_maskgit.yaml

tests/
├── unit/models/incontext/          # 84+ tests
├── unit/training/
└── integration/test_incontext_training.py  # 17 tests
```

## 🎯 Next Steps

1. ~~Fix sliding window in prepare_batch()~~ ✅ Done
2. ~~Fix sliding window in generate()~~ ✅ Done
3. ~~Fix encoding path mismatch (modality embeddings)~~ ✅ Done
4. **Test with real large medical images** - Verify memory usage with 256x256x256 volumes
5. **Create DataModule for in-context training** - Wrap dataset to produce InContextSample
6. **Add evaluation metrics** - FID, LPIPS for target modality generation quality