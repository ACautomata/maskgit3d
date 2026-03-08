# Task: Refactor Metrics into Callbacks

## Goal
将 maskgit-3d 项目中的 metrics 计算从 Task 类迁移到独立的 Callback 中，实现关注点分离。Task 类只保留核心的前向传播、反向传播和优化过程。

## Context

### 当前架构问题
- **Task 类职责过重**: `VQVAETask` 和 `MaskGITTask` 中混合了训练逻辑（forward/backward/optimize）和指标计算（logging metrics）
- **Metrics 分散在各处**: 
  - Task 内部：`train/loss`, `val/loss_l1`, `train/mask_acc` 等在 `training_step`/`validation_step` 中直接计算并 log
  - Callbacks 中：`time/*`, `gradients/*` 已经在 callbacks 中实现
  - Loss 模块：`vq_perceptual_loss.py` 返回 log_dict 供 Task 使用

### 当前 Metrics 分布
1. **VQVAETask** (`src/maskgit3d/tasks/vqvae_task.py`):
   - `train/total_loss`, `train/nll_loss`, `train/rec_loss`, `train/p_loss`, `train/g_loss`, `train/vq_loss`
   - `train/d_weight`, `train/disc_factor`, `train/disc_loss`, `train/logits_real`, `train/logits_fake`
   - `val_loss`, `val/loss_l1`, `val/loss_vq`, `val/loss_perceptual`
   - `test/loss_l1`, `test/loss_vq`, `test/inference_time`, `test/peak_memory_mb`

2. **MaskGITTask** (`src/maskgit3d/tasks/maskgit_task.py`):
   - `train/loss`, `train/mask_acc`, `train/mask_ratio`
   - `val_loss`, `val/mask_acc`, `val/sample_shape`
   - `test/loss`, `test/mask_acc`, `test/sliding_window_enabled`

3. **已有 Callbacks** (`src/maskgit3d/callbacks/`):
   - `TrainingTimeCallback`: `time/epoch_train_seconds`, `time/etc_seconds`, etc.
   - `GradientNormCallback`: `gradients/total_norm`, `gradients/{layer_name}`
   - `NaNDetectionCallback`: `train/nan_detected`, `train/nan_count`

### 当前日志格式
```python
# 当前格式: 分层命名
self.log("train/loss", loss, prog_bar=True)
self.log("val/loss_l1", loss_l1, prog_bar=True)
self.log("time/epoch_train_seconds", epoch_time)

# 期望格式: metric:value 形式
self.log("loss:train", loss, prog_bar=True)
self.log("loss_l1:val", loss_l1, prog_bar=True)
self.log("epoch_train_seconds:time", epoch_time)
```

## Requirements

### 1. Metrics 迁移到 Callbacks
为每个 Task 创建对应的 Metrics Callback:

**VQVAEMetricsCallback** (`src/maskgit3d/callbacks/vqvae_metrics.py`):
- 继承 `pytorch_lightning.callbacks.Callback`
- 在 `on_train_batch_end` 中计算并 log VQVAE 训练指标
- 在 `on_validation_batch_end` 中计算并 log 验证指标
- 在 `on_test_batch_end` 中计算并 log 测试指标
- 需要的输入：从 `outputs` 或 `batch` 中获取必要数据

**MaskGITMetricsCallback** (`src/maskgit3d/callbacks/maskgit_metrics.py`):
- 继承 `pytorch_lightning.callbacks.Callback`
- 在 `on_train_batch_end` 中计算并 log MaskGIT 训练指标
- 在 `on_validation_batch_end`/`on_test_batch_end` 中计算验证/测试指标
- 需要的输入：从 `outputs` 或 `batch` 中获取必要数据

### 2. Task 类简化

**VQVAETask** 修改后只保留:
- `forward(x)` - 前向传播
- `training_step()` - 计算 loss 并返回（不再直接 log metrics）
- `validation_step()` - 返回必要数据供 callback 使用
- `test_step()` - 返回必要数据供 callback 使用
- `configure_optimizers()` - 优化器配置

**MaskGITTask** 修改后只保留:
- `forward(x)` - 前向传播
- `training_step()` - 计算 loss 并返回
- `validation_step()` - 返回必要数据
- `test_step()` - 返回必要数据
- `configure_optimizers()` - 优化器配置

### 3. 日志格式规范化
统一使用 `metric_name:split` 格式:
- `train/loss` → `loss:train`
- `val/loss_l1` → `loss_l1:val`
- `time/epoch_train_seconds` → `epoch_train_seconds:time`
- `gradients/total_norm` → `total_norm:gradients`

**转换规则**:
- 原格式: `{split}/{metric_name}` 或 `{category}/{metric_name}`
- 新格式: `{metric_name}:{split}` 或 `{metric_name}:{category}`

### 4. 数据传递机制
Task 和 Callback 之间通过 `outputs` 传递数据:
```python
# Task.training_step
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    # 返回 dict 供 callback 使用
    return {
        "loss": loss,  # required by Lightning
        "log_data": {
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            # ... 其他需要 log 的数据
        }
    }

# Callback.on_train_batch_end
def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    if "log_data" in outputs:
        for name, value in outputs["log_data"].items():
            pl_module.log(f"{name}:train", value, prog_bar=True)
```

## File Changes

### New Files
1. `src/maskgit3d/callbacks/vqvae_metrics.py` - VQVAE metrics callback
2. `src/maskgit3d/callbacks/maskgit_metrics.py` - MaskGIT metrics callback
3. `src/maskgit3d/callbacks/metric_utils.py` - 共享的 metric 工具函数

### Modified Files
1. `src/maskgit3d/tasks/vqvae_task.py` - 移除 metric logging，返回 log_data
2. `src/maskgit3d/tasks/maskgit_task.py` - 移除 metric logging，返回 log_data
3. `src/maskgit3d/callbacks/training_time.py` - 更新日志格式
4. `src/maskgit3d/callbacks/gradient_norm.py` - 更新日志格式
5. `src/maskgit3d/callbacks/nan_detection.py` - 更新日志格式
6. `src/maskgit3d/conf/callbacks/default.yaml` - 添加新的 callbacks

## Verification Steps

1. **Run training test**:
   ```bash
   poetry run pytest tests/ -v -k "training" --no-header -q 2>&1 | head -50
   ```

2. **Check logs format**:
   ```bash
   # 运行一个短训练并检查日志输出
   maskgit3d-train model=vqgan dataset=medmnist3d \
     dataset.data_dir=/tmp/medmnist \
     training.num_epochs=1 \
     dataset.batch_size=2 \
     experiment_name=test_metrics 2>&1 | grep -E "(loss:|time:|gradients:)"
   ```

3. **Verify Callbacks are called**:
   - 检查 TensorBoard logs/ 目录下有正确的 metric 记录
   - 检查控制台输出包含新格式的 metric 名称

## Definition of Done

- [ ] VQVAEMetricsCallback 实现完成，所有原 VQVAETask 的 metrics 在 callback 中计算和记录
- [ ] MaskGITMetricsCallback 实现完成，所有原 MaskGITTask 的 metrics 在 callback 中计算和记录
- [ ] VQVAETask 和 MaskGITTask 中的 `self.log()` 调用全部移除，改为返回 `log_data`
- [ ] 所有日志格式统一为 `metric_name:split` 格式
- [ ] 所有现有 tests 通过 (`poetry run pytest tests/`)
- [ ] 新 tests 覆盖 callback 的核心功能

## References

- PyTorch Lightning Callbacks: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks
- Current Task implementations:
  - `src/maskgit3d/tasks/vqvae_task.py`
  - `src/maskgit3d/tasks/maskgit_task.py`
- Current Callback implementations:
  - `src/maskgit3d/callbacks/training_time.py`
  - `src/maskgit3d/callbacks/gradient_norm.py`
  - `src/maskgit3d/callbacks/nan_detection.py`
