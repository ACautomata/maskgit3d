# 实施计划：MAISI 架构魔改版 VQGAN

## 目标
将 MONAI `AutoencoderKlMaisi` 的 Encoder/Decoder 架构移植到项目中，移除 KL 正则化，接入 VQ 量化器。

---

## 架构对比

### 原 VAE 架构 (要移除的部分)
```
Encoder → quant_conv_mu → z_mu ─┐
           quant_conv_log_sigma → z_sigma ─┼→ sampling(z_mu, z_sigma) → z → Decoder
                                              (重参数化: z = mu + eps*sigma)
```

### 新 VQ 架构 (目标)
```
Encoder → quant_conv → Quantizer → z_q → post_quant_conv → Decoder
                             ↓
                         codebook indices
```

---

## 实施步骤

### Step 1: 创建 MaisiVQModel3D 类
**文件**: `src/maskgit3d/infrastructure/vqgan/maisi_vq_model.py`

```python
class MaisiVQModel3D(nn.Module, VQModelInterface):
    """
    基于 MAISI 架构的 VQGAN 模型。

    使用 MONAI 的 MaisiEncoder/MaisiDecoder + VectorQuantizer2。
    """

    def __init__(
        self,
        in_channels: int = 1,
        codebook_size: int = 1024,
        embed_dim: int = 256,
        latent_channels: int = 4,
        num_channels: Sequence[int] = (64, 128, 256),
        num_res_blocks: Sequence[int] = (2, 2, 2),
        attention_levels: Sequence[bool] = (False, False, False),
        norm_num_groups: int = 32,
        with_encoder_nonlocal_attn: bool = False,
        with_decoder_nonlocal_attn: bool = False,
        num_splits: int = 4,
        dim_split: int = 1,
    ):
        # Encoder (MAISI)
        self.encoder = MaisiEncoder(...)

        # VQ 量化层
        self.quantize = VectorQuantizer2(n_embed=codebook_size, embed_dim=embed_dim)
        self.quant_conv = nn.Conv3d(latent_channels, embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(embed_dim, latent_channels, 1)

        # Decoder (MAISI)
        self.decoder = MaisiDecoder(...)
```

### Step 2: 实现核心方法

```python
def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
    """编码 + 量化"""
    h = self.encoder(x)          # 确定性编码，无随机采样
    h = self.quant_conv(h)
    quant, emb_loss, info = self.quantize(h)
    return quant, emb_loss, info

def decode(self, quant: torch.Tensor) -> torch.Tensor:
    """解码"""
    quant = self.post_quant_conv(quant)
    return self.decoder(quant)
```

### Step 3: 更新配置模块
**文件**: `src/maskgit3d/config/modules.py`

添加新的工厂函数:
```python
def create_maisi_vq_module(
    image_size: int = 64,
    in_channels: int = 1,
    codebook_size: int = 1024,
    embed_dim: int = 256,
    latent_channels: int = 4,
    num_channels: Sequence[int] = (64, 128, 256),
    ...
) -> VQGANModule:
    ...
```

### Step 4: 更新训练策略
**文件**: `src/maskgit3d/infrastructure/training/strategies.py`

- 移除 KL loss 计算
- 保留 VQ commitment loss
- 保留重构 loss + 对抗 loss

---

## 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 新增 | `src/maskgit3d/infrastructure/vqgan/maisi_vq_model.py` | 主模型类 |
| 修改 | `src/maskgit3d/infrastructure/vqgan/__init__.py` | 导出新类 |
| 修改 | `src/maskgit3d/config/modules.py` | 添加配置 |
| 修改 | `src/maskgit3d/infrastructure/training/strategies.py` | 移除 KL loss |
| 新增 | `tests/unit/test_maisi_vq_model.py` | 单元测试 |

---

## 关键参数对照

| 参数 | MAISI 默认值 | 说明 |
|------|-------------|------|
| `latent_channels` | 4 | Encoder 输出通道数 |
| `num_channels` | [64, 128, 256] | 各层通道数 |
| `num_res_blocks` | [2, 2, 2] | 各层残差块数量 |
| `attention_levels` | [False, False, False] | 各层是否使用注意力 |
| `num_splits` | 4 | 内存优化：分割数量 |
| `dim_split` | 1 | 内存优化：分割维度 |

---

## 预期效果

1. **更小的 latent space**: 4 通道 vs 原来的 256 通道
2. **内存优化**: MAISI 的 `num_splits` 特性支持大 volume 处理
3. **简洁架构**: 无 KL 正则化，纯 VQ 量化