# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**maskgit-3d** 是一个基于 PyTorch + Lightning + MONAI 的 3D 医学图像生成深度学习框架，采用依赖注入架构。

- **双阶段训练**: VQGAN (Stage 1) + MaskGIT (Stage 2)
- **数据集**: MedMNIST-3D、BraTS 等医学影像数据集
- **配置管理**: Hydra + OmegaConf
- **依赖注入**: injector 库

## Runtime Environment

### Conda 环境

```bash
# 激活已有环境
conda activate maskgit3d

# 或创建新环境
conda create -n maskgit3d python=3.10 -y
conda activate maskgit3d
```

### 依赖安装

```bash
# 使用 poetry 安装
poetry install

# 或手动安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install lightning monai injector hydra-core omegaconf medmnist nibabel einops
pip install -e .
```

## Common Commands

```bash
# 安装开发依赖
poetry install

# 运行全部测试
poetry run pytest --cache-clear -vv tests

# 运行单个测试
poetry run pytest tests/path/to/test_file.py::test_function_name -v

# 运行带覆盖率测试
make test

# 代码格式化
make format
```

### 训练与测试

```bash
# VQGAN 训练 (Stage 1)
maskgit3d-train model=vqgan dataset=medmnist3d

# MaskGIT 训练 (Stage 2，需要先训练 VQGAN)
maskgit3d-train model=maskgit dataset=medmnist3d \
    model.pretrained_vqgan_path=./checkpoints/vqgan/best.ckpt

# 模型测试
maskgit3d-test model=vqgan dataset=medmnist3d \
    checkpoint.load_from=./checkpoints/vqgan/best.ckpt
```

## Architecture

### 项目结构

```
src/maskgit3d/
├── cli/              # 命令行接口 (train/test)
├── config/           # 依赖注入模块 (modules.py)
├── domain/           # 领域接口定义 (interfaces.py)
├── application/      # Fabric 训练管道 (pipeline.py)
├── infrastructure/  # 具体实现
│   ├── vqgan/        # VQVAE/VQGAN 实现
│   ├── maskgit/      # MaskGIT Transformer 实现
│   ├── data/         # 数据提供者 (MedMNIST, BraTS)
│   └── training/     # 训练/推理策略
└── conf/             # Hydra 配置文件
```

### 依赖注入模块 (config/modules.py)

| 模块 | 职责 |
|------|------|
| `ModelModule` | 模型实现 (VQVAE、MaskGIT) |
| `DataModule` | 数据加载器 (MedMNIST-3D、BraTS) |
| `TrainingModule` | 训练策略与优化器 |
| `InferenceModule` | 推理策略与指标 |
| `SystemModule` | 系统级配置 (device) |

### 领域接口 (domain/interfaces.py)

- `ModelInterface` - 模型基接口
- `VQModelInterface` - VQ 模型特化接口
- `MaskGITModelInterface` -特化接口
 MaskGIT 模型- `DataProvider` - 数据提供者接口
- `TrainingStrategy` - 训练策略接口
- `InferenceStrategy` - 推理策略接口
- `OptimizerFactory` - 优化器工厂接口

### CLI 入口 (cli/train.py)

使用 Hydra 的 `@hydra.main` 装饰器，通过配置文件创建依赖注入模块：
1. 解析配置文件提取模型、数据、训练参数
2. 创建 `CompositeModule` 配置注入绑定
3. 使用 `Injector` 创建各组件实例
4. 构建 `FabricTrainingPipeline` 并执行训练

## Key Configuration Files

| 文件 | 用途 |
|------|------|
| `conf/config.yaml` | 主配置 |
| `conf/model/vqgan.yaml` | VQGAN 模型配置 |
| `conf/model/maskgit.yaml` | MaskGIT 模型配置 |
| `conf/dataset/medmnist3d.yaml` | MedMNIST 数据集配置 |
| `conf/dataset/brats.yaml` | BraTS 数据集配置 |
| `conf/training/default.yaml` | 训练配置 |

### Dataset Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `crop_size` | Random crop size for training (D, H, W). Must be divisible by 16 for VQVAE. |
| `roi_size` | ROI size for sliding window inference (D, H, W). |
| `image_size` | Legacy parameter, used as default for crop_size/roi_size if not specified. |

## Important Notes

1. **Stage 2 依赖 Stage 1**: MaskGIT 训练需要先完成 VQGAN 训练
2. **VQGAN 冻结**: 默认情况下 Stage 2 会冻结 VQGAN 参数
3. **数据格式**: 支持 NIfTI (.nii.gz) 和 NumPy 数组格式

## Testing

- 测试覆盖要求: >80%
- 测试标记: `@pytest.mark.slow`, `@pytest.mark.gpu`, `@pytest.mark.integration`
- 单元测试: `tests/unit/`
- 集成测试: `tests/integration/`
