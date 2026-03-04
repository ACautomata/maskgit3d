# maskgit-3d

基于 PyTorch + Lightning + MONAI 的 3D 医学图像生成深度学习框架，采用依赖注入架构。

## 功能特性

- **双阶段训练**: VQGAN (Stage 1) + MaskGIT (Stage 2)
- **多数据集支持**: MedMNIST-3D、BraTS 等医学影像数据集
- **依赖注入**: 基于 `injector` 库的清晰架构
- **灵活配置**: 基于 Hydra 的配置管理系统
- **高性能训练**: Lightning Fabric 支持多 GPU 与混合精度

## 快速开始

### 安装

```bash
# 创建 conda 环境
conda create -n maskgit3d python=3.10 -y
conda activate maskgit3d

# 安装 PyTorch (CPU 或 CUDA 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装核心依赖
pip install lightning monai injector

# 安装项目
pip install -e .
```

### 一键训练与测试

> **前置条件**: conda 环境已激活 + 项目已安装

#### Stage 1 — VQGAN 训练

```bash
# 默认: MedMNIST-3D 数据集
maskgit3d-train model=vqgan dataset=medmnist3d

# 自定义参数
maskgit3d-train model=vqgan dataset=medmnist3d \
    training.num_epochs=100 \
    training.optimizer.lr=1e-4 \
    dataset.batch_size=4
```

#### Stage 2 — MaskGIT 训练

```bash
# 需要先有 Stage 1 的 VQGAN 预训练模型
maskgit3d-train model=maskgit dataset=medmnist3d \
    model.pretrained_vqgan_path=./checkpoints/vqgan/best.ckpt
```

#### 模型测试

```bash
# VQGAN 测试
maskgit3d-test model=vqgan dataset=medmnist3d \
    checkpoint.load_from=./checkpoints/vqgan/best.ckpt

# MaskGIT 测试
maskgit3d-test model=maskgit dataset=medmnist3d \
    checkpoint.load_from=./checkpoints/maskgit/best.ckpt
```

## 项目结构

```
maskgit-3d/
├── src/maskgit3d/
│   ├── cli/              # 命令行接口 (train/test)
│   ├── config/           # 配置模块与依赖注入
│   ├── domain/           # 领域接口定义
│   ├── application/      # 训练管道
│   ├── infrastructure/   # 模型、数据、训练策略实现
│   └── conf/            # Hydra 配置文件
├── tests/                # 测试套件 (>80% 覆盖率)
└── pyproject.toml       # 项目配置
```

## 核心配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model.codebook_size` | `1024` | VQ 码本大小 |
| `model.embed_dim` | `256` | 嵌入维度 |
| `model.latent_channels` | `256` | 潜在空间通道数 |
| `model.transformer_hidden` | `768` | Transformer 隐藏维度 |
| `model.transformer_layers` | `12` | Transformer 层数 |
| `training.num_epochs` | `100` | 训练轮数 |
| `training.optimizer.lr` | `1e-4` | 学习率 |
| `dataset.batch_size` | `4` | 批大小 |

## 开发命令

```bash
# 安装开发依赖
poetry install

# 运行全部测试
poetry run pytest --cache-clear -vv tests

# 运行带覆盖率测试
make test

# 代码格式化
make format
```

## GPU 使用

```bash
# 指定 GPU
CUDA_VISIBLE_DEVICES=0 maskgit3d-train model=vqgan dataset=medmnist3d

# 多 GPU DDP
maskgit3d-train model=vqgan dataset=medmnist3d \
    training.fabric.devices=2 \
    training.fabric.strategy=ddp
```

---

<!-- Agent 文档开始 -->

<details>
<summary><h2>📖 Agent 完整文档 (点击展开)</h2></summary>

### 架构概览

本项目采用**依赖注入 (Dependency Injection)** 架构，核心模块位于 `src/maskgit3d/config/modules.py`。

#### 依赖注入模块

| 模块 | 职责 |
|------|------|
| `ModelModule` | 提供模型实现 (VQVAE、MaskGIT) |
| `DataModule` | 提供数据加载器 (MedMNIST-3D、BraTS) |
| `TrainingModule` | 提供训练策略与优化器 |
| `InferenceModule` | 提供推理策略与指标 |
| `SystemModule` | 提供系统级配置 (device) |

#### 领域接口 (`domain/interfaces.py`)

```python
ModelInterface          # 模型基接口
VQModelInterface        # VQ 模型特化接口
MaskGITModelInterface   # MaskGIT 模型特化接口
DataProvider            # 数据提供者接口
TrainingStrategy       # 训练策略接口
InferenceStrategy      # 推理策略接口
OptimizerFactory       # 优化器工厂接口
Metrics                # 指标接口
```

### 模型架构

#### VQ-VAE / VQGAN (Stage 1)

- **编码器**: 3D 卷积网络，将输入体积映射到潜在空间
- **向量量化**: 使用码本 (codebook) 对潜在表示进行量化
- **解码器**: 3D 卷积网络，从量化潜在重建图像
- **判别器** (可选): GAN 对抗训练

**配置文件**: `conf/model/vqgan.yaml`

#### MaskGIT (Stage 2)

- **VQGAN 编码器**: 冻结 (默认)，将图像编码为 token 序列
- **Transformer**: 双向 Transformer (BERT-style) 预测 masked tokens
- **VQGAN 解码器**: 将 token 序列解码回图像

**配置文件**: `conf/model/maskgit.yaml`

### 数据集支持

| 数据集 | 配置 | 说明 |
|--------|------|------|
| MedMNIST-3D | `dataset=medmnist3d` | 28x28x28 医学图像分类数据集 |
| BraTS | `dataset=brats` | 多模态脑肿瘤 MRI 数据集 |

**数据提供者** (`infrastructure/data/`):

- `MedMnist3DDataProvider`: MedMNIST-3D 数据加载
- `BraTSDataProvider`: BraTS 数据加载
- `SimpleDataProvider`: 简单数据加载

### 训练策略

#### VQGAN 训练策略

- **损失函数**:
  - Reconstruction Loss (L1/L2)
  - Perceptual Loss (MONAI PerceptualLoss)
  - Codebook Loss (VQ 承诺损失)
  - GAN Loss (判别器对抗损失)

#### MaskGIT 训练策略

- **损失函数**:
  - Masked Prediction Loss
  - Reconstruction Loss
- **Masking**: 随机 mask 50% (默认) 的 tokens

### 推理策略

- **VQGAN**: 直接重建
- **MaskGIT**: 迭代式解码 (默认 12 步)

### 输出格式

#### Checkpoint 结构

```python
{
    "epoch": int,
    "global_step": int,
    "model_state_dict": {...},
    "optimizer_state_dict": {...},
    "metrics": {...}
}
```

#### NIfTI 输出

当 `output.export_nifti=true` 时，保存为 `.nii.gz` 格式：

```
outputs/
├── predictions_batch_0.nii.gz
├── probabilities_batch_0.nii.gz
└── ...
```

### 配置系统

项目使用 **Hydra** 进行配置管理，配置文件位于 `src/maskgit3d/conf/`：

```
conf/
├── config.yaml          # 主配置
├── model/
│   ├── vqgan.yaml
│   └── maskgit.yaml
├── dataset/
│   ├── medmnist3d.yaml
│   └── brats.yaml
├── training/
│   └── default.yaml
└── experiment/
    ├── vqgan_brats.yaml
    └── maskgit_medmnist.yaml
```

### 测试覆盖

- **单元测试**: `tests/unit/`
- **集成测试**: `tests/integration/`
- **覆盖率**: >80%

### 关键文件索引

| 文件 | 职责 |
|------|------|
| `cli/train.py` | 训练 CLI 入口 |
| `cli/test.py` | 测试 CLI 入口 |
| `application/pipeline.py` | Fabric 训练管道 |
| `config/modules.py` | 依赖注入模块 |
| `infrastructure/vqgan/vqvae.py` | VQVAE 实现 |
| `infrastructure/maskgit/maskgit_model.py` | MaskGIT 实现 |
| `infrastructure/maskgit/transformer.py` | Transformer 实现 |
| `infrastructure/training/strategies.py` | 训练/推理策略 |

### Python API 示例

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

### 依赖项

```
torch>=2.0.0
lightning>=2.0.0
monai>=1.0.0
injector>=0.21.0
hydra-core>=1.3.0
omegaconf>=2.3.0
medmnist>=3.0.0
nibabel>=5.0.0
einops>=0.7.0
```

### 注意事项

1. **Stage 2 依赖 Stage 1**: MaskGIT 训练需要先完成 VQGAN 训练
2. **VQGAN 冻结**: 默认情况下 Stage 2 会冻结 VQGAN 参数
3. **数据格式**: 支持 NIfTI (.nii.gz) 和 NumPy 数组格式

</details>

<!-- Agent 文档结束 -->
