# maskgit-3d v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor maskgit-3d to use PyTorch Lightning + Hydra best practices with pure Lightning architecture.

**Architecture:** Four-layer architecture (models/tasks/data/train.py). Remove Fabric, injector, TrainingStrategy. Keep MONAI components (Metrics, MAISI VAE structure). Model-Task separation.

**Tech Stack:** PyTorch, Lightning (LightningModule+Trainer), Hydra, MONAI, OmegaConf

---

## Phase 1: Foundation & Project Structure

### Task 1: Create v2 Directory Structure

**Files:**
- Create: `src/maskgit3d/models/__init__.py`
- Create: `src/maskgit3d/models/vqvae/__init__.py`
- Create: `src/maskgit3d/models/discriminator/__init__.py`
- Create: `src/maskgit3d/models/maskgit/__init__.py`
- Create: `src/maskgit3d/tasks/__init__.py`
- Create: `src/maskgit3d/data/__init__.py`
- Create: `src/maskgit3d/losses/__init__.py`
- Create: `src/maskgit3d/metrics/__init__.py`
- Create: `src/maskgit3d/callbacks/__init__.py`
- Create: `src/maskgit3d/utils/__init__.py`
- Create: `configs/model/` directory
- Create: `configs/task/` directory
- Create: `configs/data/` directory
- Create: `configs/optimizer/` directory
- Create: `configs/scheduler/` directory
- Create: `configs/trainer/` directory
- Create: `configs/callbacks/` directory
- Create: `configs/logger/` directory

**Step 1: Create all directories and __init__.py files**

```bash
mkdir -p src/maskgit3d/models/vqvae
mkdir -p src/maskgit3d/models/discriminator
mkdir -p src/maskgit3d/models/maskgit
mkdir -p src/maskgit3d/tasks
mkdir -p src/maskgit3d/data
mkdir -p src/maskgit3d/losses
mkdir -p src/maskgit3d/metrics
mkdir -p src/maskgit3d/callbacks
mkdir -p src/maskgit3d/utils
mkdir -p configs/model
mkdir -p configs/task
mkdir -p configs/data
mkdir -p configs/optimizer
mkdir -p configs/scheduler
mkdir -p configs/trainer
mkdir -p configs/callbacks
mkdir -p configs/logger
touch src/maskgit3d/models/__init__.py
touch src/maskgit3d/models/vqvae/__init__.py
touch src/maskgit3d/models/discriminator/__init__.py
touch src/maskgit3d/models/maskgit/__init__.py
touch src/maskgit3d/tasks/__init__.py
touch src/maskgit3d/data/__init__.py
touch src/maskgit3d/losses/__init__.py
touch src/maskgit3d/metrics/__init__.py
touch src/maskgit3d/callbacks/__init__.py
touch src/maskgit3d/utils/__init__.py
```

**Step 2: Verify structure created**

Run: `find src/maskgit3d -type d | sort`
Expected: All directories listed

**Step 3: Commit**

```bash
git add .
git commit -m "feat(v2): create directory structure"
```

---

### Task 2: Migrate VectorQuantizer

**Files:**
- Read: `src/maskgit3d/infrastructure/vqgan/vector_quantizer.py`
- Create: `src/maskgit3d/models/vqvae/quantizer.py`
- Test: `tests/unit/models/vqvae/test_quantizer.py`

**Step 1: Write the failing test**

```python
# tests/unit/models/vqvae/test_quantizer.py
import pytest
import torch
from src.maskgit3d.models.vqvae.quantizer import VectorQuantizer


def test_vector_quantizer_forward():
    """Test VectorQuantizer forward pass returns correct shapes."""
    num_embeddings = 8192
    embedding_dim = 256
    batch_size = 2
    spatial_dims = (4, 4, 4)
    
    quantizer = VectorQuantizer(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
    )
    
    # Input: (B, C, D, H, W)
    z = torch.randn(batch_size, embedding_dim, *spatial_dims)
    
    z_q, vq_loss, indices = quantizer(z)
    
    # Check shapes
    assert z_q.shape == z.shape, f"z_q shape mismatch: {z_q.shape} vs {z.shape}"
    assert indices.shape == (batch_size, *spatial_dims), f"indices shape mismatch"
    assert vq_loss.dim() == 0, "vq_loss should be scalar"


def test_vector_quantizer_straight_through_estimator():
    """Test that gradients flow through straight-through estimator."""
    quantizer = VectorQuantizer(num_embeddings=100, embedding_dim=16)
    z = torch.randn(1, 16, 2, 2, 2, requires_grad=True)
    
    z_q, vq_loss, _ = quantizer(z)
    loss = z_q.sum() + vq_loss
    loss.backward()
    
    assert z.grad is not None, "Gradient should flow through quantizer"
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/models/vqvae/test_quantizer.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Copy and adapt VectorQuantizer from v1**

Read original file and copy to new location, keeping only the core quantizer logic:

```python
# src/maskgit3d/models/vqvae/quantizer.py
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector Quantization layer for VQ-VAE.
    
    Args:
        num_embeddings: Number of codebook entries
        embedding_dim: Dimension of each embedding vector
        commitment_cost: Weight for commitment loss (default: 0.25)
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize continuous latent vectors.
        
        Args:
            z: Input tensor of shape (B, C, D, H, W)
        
        Returns:
            z_q: Quantized tensor (B, C, D, H, W)
            vq_loss: Scalar tensor with VQ loss
            indices: Quantized indices (B, D, H, W)
        """
        # Flatten spatial dimensions
        # z: (B, C, D, H, W) -> (B, D, H, W, C)
        z = z.permute(0, 2, 3, 4, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)
        
        # Calculate distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z^T e
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        
        # Get closest codebook entry
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Quantize
        z_q = self.embedding(encoding_indices).view(z.shape)
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        # Compute loss
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Reshape indices
        indices = encoding_indices.view(z.shape[:-1])
        
        # Permute back to (B, C, D, H, W)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
        
        return z_q, vq_loss, indices
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/models/vqvae/test_quantizer.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/maskgit3d/models/vqvae/quantizer.py tests/unit/models/vqvae/test_quantizer.py
git commit -m "feat(v2): migrate VectorQuantizer with tests"
```

---

### Task 3: Migrate VQVAE Encoder (MONAI MaisiEncoder)

**Files:**
- Read: `src/maskgit3d/infrastructure/vqgan/vqvae.py`
- Create: `src/maskgit3d/models/vqvae/encoder.py`
- Test: `tests/unit/models/vqvae/test_encoder.py`

**Step 1: Write the failing test**

```python
# tests/unit/models/vqvae/test_encoder.py
import pytest
import torch
from src.maskgit3d.models.vqvae.encoder import Encoder


def test_encoder_forward():
    """Test Encoder forward pass."""
    in_channels = 1
    out_channels = 256  # embedding_dim
    spatial_dims = 3
    
    encoder = Encoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    
    # Input: (B, C, D, H, W)
    x = torch.randn(2, in_channels, 32, 32, 32)
    z = encoder(x)
    
    # Output should have spatial dims reduced by factor of 16
    assert z.shape[0] == 2, "Batch size mismatch"
    assert z.shape[1] == out_channels, f"Channel mismatch: {z.shape[1]} vs {out_channels}"
    assert z.shape[2] == 2, f"Spatial dim D mismatch: {z.shape[2]}"
    assert z.shape[3] == 2, f"Spatial dim H mismatch: {z.shape[3]}"
    assert z.shape[4] == 2, f"Spatial dim W mismatch: {z.shape[4]}"
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/models/vqvae/test_encoder.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create Encoder wrapper around MONAI MaisiEncoder**

```python
# src/maskgit3d/models/vqvae/encoder.py
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from monai.networks.nets import MaisiEncoder


class Encoder(nn.Module):
    """Encoder network using MONAI's MaisiEncoder.
    
    Args:
        spatial_dims: Number of spatial dimensions (2 or 3)
        in_channels: Number of input channels
        out_channels: Number of output channels (embedding dimension)
        channels: Base number of channels (default: 32)
        strides: Strides for each downsampling layer (default: (2, 2, 2, 2))
        kernel_size: Convolution kernel size (default: 3)
        up_kernel_size: Transposed conv kernel size (default: 3)
        num_res_units: Number of residual units (default: 2)
    """
    
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int] = (32, 64, 128, 256),
        strides: Sequence[int] = (2, 2, 2, 2),
        kernel_size: int = 3,
        up_kernel_size: int = 3,
        num_res_units: int = 2,
    ):
        super().__init__()
        
        self.encoder = MaisiEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            num_res_units=num_res_units,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W) for 3D
        
        Returns:
            Latent representation of shape (B, out_channels, D/16, H/16, W/16)
        """
        return self.encoder(x)
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/models/vqvae/test_encoder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/maskgit3d/models/vqvae/encoder.py tests/unit/models/vqvae/test_encoder.py
git commit -m "feat(v2): migrate Encoder with MONAI MaisiEncoder wrapper"
```

---

### Task 4: Migrate VQVAE Decoder (MONAI MaisiDecoder)

**Files:**
- Create: `src/maskgit3d/models/vqvae/decoder.py`
- Test: `tests/unit/models/vqvae/test_decoder.py`

**Step 1: Write the failing test**

```python
# tests/unit/models/vqvae/test_decoder.py
import pytest
import torch
from src.maskgit3d.models.vqvae.decoder import Decoder


def test_decoder_forward():
    """Test Decoder forward pass."""
    in_channels = 256  # embedding_dim
    out_channels = 1
    spatial_dims = 3
    
    decoder = Decoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    
    # Input: (B, C, D, H, W) - latent space
    z = torch.randn(2, in_channels, 2, 2, 2)
    x_recon = decoder(z)
    
    # Output should have spatial dims expanded by factor of 16
    assert x_recon.shape[0] == 2, "Batch size mismatch"
    assert x_recon.shape[1] == out_channels, f"Channel mismatch"
    assert x_recon.shape[2] == 32, f"Spatial dim D mismatch: {x_recon.shape[2]}"
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/models/vqvae/test_decoder.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create Decoder wrapper**

```python
# src/maskgit3d/models/vqvae/decoder.py
from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.nets import MaisiDecoder


class Decoder(nn.Module):
    """Decoder network using MONAI's MaisiDecoder.
    
    Args:
        spatial_dims: Number of spatial dimensions (2 or 3)
        in_channels: Number of input channels (embedding dimension)
        out_channels: Number of output channels
        channels: Base number of channels (default: (256, 128, 64, 32))
        strides: Strides for each upsampling layer (default: (2, 2, 2, 2))
        kernel_size: Convolution kernel size (default: 3)
        up_kernel_size: Transposed conv kernel size (default: 3)
        num_res_units: Number of residual units (default: 2)
    """
    
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int] = (256, 128, 64, 32),
        strides: Sequence[int] = (2, 2, 2, 2),
        kernel_size: int = 3,
        up_kernel_size: int = 3,
        num_res_units: int = 2,
    ):
        super().__init__()
        
        self.decoder = MaisiDecoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            num_res_units=num_res_units,
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction.
        
        Args:
            z: Latent tensor of shape (B, C, D, H, W)
        
        Returns:
            Reconstructed tensor of shape (B, out_channels, D*16, H*16, W*16)
        """
        return self.decoder(z)
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/models/vqvae/test_decoder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/maskgit3d/models/vqvae/decoder.py tests/unit/models/vqvae/test_decoder.py
git commit -m "feat(v2): migrate Decoder with MONAI MaisiDecoder wrapper"
```

---

### Task 5: Create VQVAE Model

**Files:**
- Create: `src/maskgit3d/models/vqvae/vqvae.py`
- Update: `src/maskgit3d/models/vqvae/__init__.py`
- Test: `tests/unit/models/vqvae/test_vqvae.py`

**Step 1: Write the failing test**

```python
# tests/unit/models/vqvae/test_vqvae.py
import pytest
import torch
from src.maskgit3d.models.vqvae import VQVAE


def test_vqvae_forward():
    """Test VQVAE full forward pass."""
    model = VQVAE(
        spatial_dims=3,
        in_channels=1,
        embedding_dim=64,
        num_embeddings=512,
    )
    
    x = torch.randn(2, 1, 16, 16, 16)
    x_recon, vq_loss, indices = model(x)
    
    assert x_recon.shape == x.shape, f"Reconstruction shape mismatch"
    assert vq_loss.dim() == 0, "vq_loss should be scalar"
    # Indices shape: (B, D/16, H/16, W/16)
    assert indices.shape == (2, 1, 1, 1), f"indices shape mismatch: {indices.shape}"


def test_vqvae_encode_decode():
    """Test VQVAE encode/decode separately."""
    model = VQVAE(
        spatial_dims=3,
        in_channels=1,
        embedding_dim=64,
        num_embeddings=512,
    )
    
    x = torch.randn(2, 1, 16, 16, 16)
    z_q, vq_loss, indices = model.encode(x)
    x_recon = model.decode(z_q)
    
    assert z_q.shape[1] == 64, "Latent channel mismatch"
    assert x_recon.shape == x.shape, "Reconstruction shape mismatch"
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/models/vqvae/test_vqvae.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create VQVAE model**

```python
# src/maskgit3d/models/vqvae/vqvae.py
from typing import Tuple

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .quantizer import VectorQuantizer


class VQVAE(nn.Module):
    """Vector Quantized Variational Autoencoder.
    
    Combines Encoder, VectorQuantizer, and Decoder for 3D medical image
    compression and generation.
    
    Args:
        spatial_dims: Number of spatial dimensions (2 or 3)
        in_channels: Number of input channels
        embedding_dim: Dimension of latent embeddings
        num_embeddings: Number of codebook entries
        commitment_cost: Weight for commitment loss (default: 0.25)
        encoder_channels: Encoder channel progression (default: (32, 64, 128, 256))
        decoder_channels: Decoder channel progression (default: (256, 128, 64, 32))
    """
    
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        embedding_dim: int = 256,
        num_embeddings: int = 8192,
        commitment_cost: float = 0.25,
        encoder_channels: Tuple[int, ...] = (32, 64, 128, 256),
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32),
    ):
        super().__init__()
        
        self.encoder = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=embedding_dim,
            channels=encoder_channels,
        )
        
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )
        
        self.decoder = Decoder(
            spatial_dims=spatial_dims,
            in_channels=embedding_dim,
            out_channels=in_channels,
            channels=decoder_channels,
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to quantized latent.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        
        Returns:
            z_q: Quantized latent (B, embedding_dim, D/16, H/16, W/16)
            vq_loss: Scalar VQ loss
            indices: Codebook indices (B, D/16, H/16, W/16)
        """
        z = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z)
        return z_q, vq_loss, indices
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction.
        
        Args:
            z_q: Quantized latent (B, embedding_dim, D, H, W)
        
        Returns:
            Reconstruction (B, C, D*16, H*16, W*16)
        """
        return self.decoder(z_q)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode -> quantize -> decode.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        
        Returns:
            x_recon: Reconstructed input (B, C, D, H, W)
            vq_loss: Scalar VQ loss
            indices: Codebook indices (B, D/16, H/16, W/16)
        """
        z_q, vq_loss, indices = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, vq_loss, indices
    
    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from codebook indices only.
        
        Args:
            indices: Codebook indices (B, D, H, W)
        
        Returns:
            Reconstruction (B, C, D*16, H*16, W*16)
        """
        z_q = self.quantizer.embedding(indices)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
        return self.decode(z_q)
```

**Step 4: Update __init__.py**

```python
# src/maskgit3d/models/vqvae/__init__.py
from .vqvae import VQVAE
from .encoder import Encoder
from .decoder import Decoder
from .quantizer import VectorQuantizer

__all__ = ["VQVAE", "Encoder", "Decoder", "VectorQuantizer"]
```

**Step 5: Run test to verify it passes**

Run: `poetry run pytest tests/unit/models/vqvae/test_vqvae.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/maskgit3d/models/vqvae/ tests/unit/models/vqvae/test_vqvae.py
git commit -m "feat(v2): create VQVAE model combining Encoder, Quantizer, Decoder"
```

---

## Phase 2: Discriminator & Losses

### Task 6: Create Discriminator Model

**Files:**
- Create: `src/maskgit3d/models/discriminator/patch_discriminator.py`
- Test: `tests/unit/models/discriminator/test_patch_discriminator.py`

**Step 1: Write the failing test**

```python
# tests/unit/models/discriminator/test_patch_discriminator.py
import pytest
import torch
from src.maskgit3d.models.discriminator import PatchDiscriminator3D


def test_patch_discriminator_forward():
    """Test PatchDiscriminator3D forward pass."""
    discriminator = PatchDiscriminator3D(
        in_channels=1,
        ndf=64,
        n_layers=3,
    )
    
    # Input: (B, C, D, H, W)
    x = torch.randn(2, 1, 16, 16, 16)
    output = discriminator(x)
    
    # Output should be a list of feature maps
    assert isinstance(output, list), "Output should be a list"
    assert len(output) > 0, "Output should not be empty"
    # Each output is a tuple of (feature_map, None)
    assert all(isinstance(o, tuple) for o in output), "Each output should be a tuple"
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/models/discriminator/test_patch_discriminator.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create PatchDiscriminator3D**

```python
# src/maskgit3d/models/discriminator/patch_discriminator.py
from typing import List, Tuple

import torch
import torch.nn as nn


class PatchDiscriminator3D(nn.Module):
    """3D PatchGAN Discriminator for adversarial training.
    
    Args:
        in_channels: Number of input channels
        ndf: Base number of discriminator features (default: 64)
        n_layers: Number of convolutional layers (default: 3)
        norm_layer: Normalization layer type (default: "instance")
    """
    
    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: str = "instance",
    ):
        super().__init__()
        
        use_bias = norm_layer == "instance"
        
        # Build convolutional layers
        layers = []
        
        # First layer: no normalization
        layers.extend([
            nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        
        # Middle layers: with normalization
        mult = 1
        for n in range(1, n_layers):
            mult_prev = mult
            mult = min(2 ** n, 8)
            layers.extend([
                nn.Conv3d(ndf * mult_prev, ndf * mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                self._get_norm_layer(ndf * mult, norm_layer),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        
        # Last layer: stride 1
        mult_prev = mult
        mult = min(2 ** n_layers, 8)
        layers.extend([
            nn.Conv3d(ndf * mult_prev, ndf * mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
            self._get_norm_layer(ndf * mult, norm_layer),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        
        # Output layer
        layers.append(nn.Conv3d(ndf * mult, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def _get_norm_layer(self, num_features: int, norm_type: str) -> nn.Module:
        """Get normalization layer by type."""
        if norm_type == "instance":
            return nn.InstanceNorm3d(num_features, affine=False)
        elif norm_type == "batch":
            return nn.BatchNorm3d(num_features)
        else:
            raise ValueError(f"Unknown norm layer: {norm_type}")
    
    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, None]]:
        """Forward pass returning discriminator output.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        
        Returns:
            List of (feature_map, None) tuples for multi-scale discrimination
        """
        output = self.model(x)
        return [(output, None)]
```

**Step 4: Update __init__.py**

```python
# src/maskgit3d/models/discriminator/__init__.py
from .patch_discriminator import PatchDiscriminator3D

__all__ = ["PatchDiscriminator3D"]
```

**Step 5: Run test to verify it passes**

Run: `poetry run pytest tests/unit/models/discriminator/test_patch_discriminator.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/maskgit3d/models/discriminator/ tests/unit/models/discriminator/
git commit -m "feat(v2): create PatchDiscriminator3D for GAN training"
```

---

### Task 7: Create GAN Loss

**Files:**
- Create: `src/maskgit3d/losses/gan_loss.py`
- Test: `tests/unit/losses/test_gan_loss.py`

**Step 1: Write the failing test**

```python
# tests/unit/losses/test_gan_loss.py
import pytest
import torch
from src.maskgit3d.losses.gan_loss import GANLoss


def test_gan_loss_lsgan():
    """Test LSGAN loss."""
    loss_fn = GANLoss(gan_mode="lsgan")
    
    # Real prediction (should be close to 1)
    real_pred = torch.randn(2, 1, 4, 4, 4)
    loss_real = loss_fn(real_pred, target_is_real=True)
    
    # Fake prediction (should be close to 0)
    fake_pred = torch.randn(2, 1, 4, 4, 4)
    loss_fake = loss_fn(fake_pred, target_is_real=False)
    
    assert loss_real.dim() == 0, "Loss should be scalar"
    assert loss_fake.dim() == 0, "Loss should be scalar"


def test_gan_loss_discriminator():
    """Test discriminator loss calculation."""
    loss_fn = GANLoss(gan_mode="lsgan")
    
    real_pred = torch.ones(2, 1, 4, 4, 4) * 0.9  # Close to 1
    fake_pred = torch.ones(2, 1, 4, 4, 4) * 0.1  # Close to 0
    
    loss_d = loss_fn.discriminator_loss(real_pred, fake_pred)
    
    assert loss_d.dim() == 0, "Loss should be scalar"


def test_gan_loss_generator():
    """Test generator loss calculation."""
    loss_fn = GANLoss(gan_mode="lsgan")
    
    fake_pred = torch.ones(2, 1, 4, 4, 4) * 0.1  # Should be 1
    loss_g = loss_fn.generator_loss(fake_pred)
    
    assert loss_g.dim() == 0, "Loss should be scalar"
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/losses/test_gan_loss.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create GANLoss**

```python
# src/maskgit3d/losses/gan_loss.py
from typing import Optional

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """GAN loss supporting multiple GAN modes.
    
    Args:
        gan_mode: Type of GAN loss - "lsgan", "vanilla", or "hinge"
        target_real_label: Label value for real images (default: 1.0)
        target_fake_label: Label value for fake images (default: 0.0)
    """
    
    def __init__(
        self,
        gan_mode: str = "lsgan",
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ):
        super().__init__()
        
        self.gan_mode = gan_mode
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "hinge":
            self.loss = None  # Hinge uses custom logic
        else:
            raise NotImplementedError(f"GAN mode {gan_mode} not implemented")
    
    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Create label tensors with same shape as prediction."""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Calculate GAN loss.
        
        Args:
            prediction: Discriminator output
            target_is_real: True if target is real, False if fake
        
        Returns:
            Scalar loss value
        """
        if self.gan_mode == "hinge":
            if target_is_real:
                # Hinge loss for real: max(0, 1 - D(x))
                loss = -torch.mean(torch.nn.functional.relu(1.0 - prediction))
            else:
                # Hinge loss for fake: max(0, 1 + D(G(z)))
                loss = -torch.mean(torch.nn.functional.relu(1.0 + prediction))
            return loss
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)
    
    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate total discriminator loss.
        
        Args:
            real_pred: Discriminator output for real images
            fake_pred: Discriminator output for fake images
        
        Returns:
            Scalar discriminator loss
        """
        loss_real = self(real_pred, target_is_real=True)
        loss_fake = self(fake_pred, target_is_real=False)
        return (loss_real + loss_fake) * 0.5
    
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Calculate generator adversarial loss.
        
        Args:
            fake_pred: Discriminator output for generated images
        
        Returns:
            Scalar generator adversarial loss
        """
        return self(fake_pred, target_is_real=True)
```

**Step 4: Update __init__.py**

```python
# src/maskgit3d/losses/__init__.py
from .gan_loss import GANLoss

__all__ = ["GANLoss"]
```

**Step 5: Run test to verify it passes**

Run: `poetry run pytest tests/unit/losses/test_gan_loss.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/maskgit3d/losses/ tests/unit/losses/
git commit -m "feat(v2): create GANLoss supporting LSGAN, vanilla, and hinge"
```

---

### Task 8: Create VQ Loss

**Files:**
- Create: `src/maskgit3d/losses/vq_loss.py`
- Test: `tests/unit/losses/test_vq_loss.py`

**Step 1: Write the failing test**

```python
# tests/unit/losses/test_vq_loss.py
import pytest
import torch
from src.maskgit3d.losses.vq_loss import VQLoss


def test_vq_loss():
    """Test VQ loss calculation."""
    loss_fn = VQLoss(commitment_cost=0.25)
    
    # Create fake quantized values
    z = torch.randn(2, 64, 4, 4, 4, requires_grad=True)
    z_q = z + torch.randn_like(z) * 0.1  # Slightly different
    
    loss = loss_fn(z, z_q)
    
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/losses/test_vq_loss.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create VQLoss**

```python
# src/maskgit3d/losses/vq_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQLoss(nn.Module):
    """Vector Quantization loss.
    
    Combines codebook loss and commitment loss.
    
    Args:
        commitment_cost: Weight for commitment loss (default: 0.25)
    """
    
    def __init__(self, commitment_cost: float = 0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
    
    def forward(self, z: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        """Calculate VQ loss.
        
        Args:
            z: Encoder output (continuous)
            z_q: Quantized output
        
        Returns:
            Scalar VQ loss
        """
        # Codebook loss: ||sg[z] - e||^2
        codebook_loss = F.mse_loss(z_q, z.detach())
        
        # Commitment loss: ||z - sg[e]||^2
        commitment_loss = F.mse_loss(z, z_q.detach())
        
        return codebook_loss + self.commitment_cost * commitment_loss
```

**Step 4: Update __init__.py**

```python
# src/maskgit3d/losses/__init__.py
from .gan_loss import GANLoss
from .vq_loss import VQLoss

__all__ = ["GANLoss", "VQLoss"]
```

**Step 5: Run test to verify it passes**

Run: `poetry run pytest tests/unit/losses/test_vq_loss.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/maskgit3d/losses/ tests/unit/losses/
git commit -m "feat(v2): create VQLoss for codebook and commitment losses"
```

---

## Phase 3: Tasks Layer (LightningModule)

### Task 9: Create BaseTask

**Files:**
- Create: `src/maskgit3d/tasks/base_task.py`
- Test: `tests/unit/tasks/test_base_task.py`

**Step 1: Write the failing test**

```python
# tests/unit/tasks/test_base_task.py
import pytest
from pathlib import Path
import torch
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.maskgit3d.tasks.base_task import BaseTask


class DummyTask(BaseTask):
    """Simple task for testing BaseTask."""
    
    def __init__(self, model_cfg, optimizer_cfg, scheduler_cfg=None):
        super().__init__(optimizer_cfg, scheduler_cfg)
        self.model = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self(x), y)


def test_base_task_configure_optimizers():
    """Test optimizer configuration from Hydra config."""
    optimizer_cfg = DictConfig({
        "_target_": "torch.optim.Adam",
        "lr": 1e-3,
    })
    
    task = DummyTask(
        model_cfg=None,
        optimizer_cfg=optimizer_cfg,
    )
    
    optimizers = task.configure_optimizers()
    
    assert isinstance(optimizers, torch.optim.Adam)


def test_base_task_save_hyperparameters():
    """Test that hyperparameters are saved."""
    optimizer_cfg = DictConfig({"_target_": "torch.optim.Adam", "lr": 1e-3})
    scheduler_cfg = DictConfig({
        "_target_": "torch.optim.lr_scheduler.StepLR",
        "step_size": 10,
    })
    
    task = DummyTask(
        model_cfg=None,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
    )
    
    assert hasattr(task, "hparams")
    assert "optimizer_cfg" in task.hparams
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/tasks/test_base_task.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create BaseTask**

```python
# src/maskgit3d/tasks/base_task.py
from typing import Any, Dict, Optional

import torch
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from hydra.utils import instantiate


class BaseTask(LightningModule):
    """Base class for all training tasks.
    
    Provides common functionality:
    - Hyperparameter saving for checkpoint restoration
    - Optimizer/scheduler configuration from Hydra configs
    
    Args:
        optimizer_cfg: Hydra config for optimizer instantiation
        scheduler_cfg: Hydra config for scheduler instantiation (optional)
    """
    
    def __init__(
        self,
        optimizer_cfg: DictConfig,
        scheduler_cfg: Optional[DictConfig] = None,
    ):
        super().__init__()
        # Save hyperparameters to checkpoint for load_from_checkpoint
        self.save_hyperparameters(logger=False)
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
    
    def configure_optimizers(self) -> Any:
        """Configure optimizer and optional scheduler from Hydra configs."""
        optimizer = instantiate(self.optimizer_cfg, params=self.parameters())
        
        if self.scheduler_cfg is None:
            return optimizer
        
        scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/tasks/test_base_task.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/maskgit3d/tasks/base_task.py tests/unit/tasks/test_base_task.py
git commit -m "feat(v2): create BaseTask with optimizer/scheduler configuration"
```

---

### Task 10: Create VQVAETask (Manual Optimization)

**Files:**
- Create: `src/maskgit3d/tasks/vqvae_task.py`
- Test: `tests/unit/tasks/test_vqvae_task.py`

**Step 1: Write the failing test**

```python
# tests/unit/tasks/test_vqvae_task.py
import pytest
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.maskgit3d.tasks.vqvae_task import VQVAETask


def test_vqvae_task_manual_optimization():
    """Test that VQVAETask uses manual optimization."""
    model_cfg = DictConfig({
        "_target_": "src.maskgit3d.models.vqvae.VQVAE",
        "spatial_dims": 3,
        "in_channels": 1,
        "embedding_dim": 32,
        "num_embeddings": 128,
    })
    
    discriminator_cfg = DictConfig({
        "_target_": "src.maskgit3d.models.discriminator.PatchDiscriminator3D",
        "in_channels": 1,
        "ndf": 16,
        "n_layers": 2,
    })
    
    optimizer_cfg = DictConfig({
        "_target_": "torch.optim.Adam",
        "lr": 1e-4,
    })
    
    task = VQVAETask(
        model_cfg=model_cfg,
        discriminator_cfg=discriminator_cfg,
        optimizer_cfg=optimizer_cfg,
    )
    
    assert task.automatic_optimization is False, "VQVAETask should use manual optimization"


def test_vqvae_task_training_step():
    """Test VQVAETask training step runs without error."""
    model_cfg = DictConfig({
        "_target_": "src.maskgit3d.models.vqvae.VQVAE",
        "spatial_dims": 3,
        "in_channels": 1,
        "embedding_dim": 32,
        "num_embeddings": 128,
    })
    
    discriminator_cfg = DictConfig({
        "_target_": "src.maskgit3d.models.discriminator.PatchDiscriminator3D",
        "in_channels": 1,
        "ndf": 16,
        "n_layers": 2,
    })
    
    optimizer_cfg = DictConfig({
        "_target_": "torch.optim.Adam",
        "lr": 1e-4,
    })
    
    task = VQVAETask(
        model_cfg=model_cfg,
        discriminator_cfg=discriminator_cfg,
        optimizer_cfg=optimizer_cfg,
    )
    
    # Create fake batch
    batch = (torch.randn(2, 1, 16, 16, 16), torch.zeros(2))
    
    # Configure optimizers (needed for training_step)
    optimizers = task.configure_optimizers()
    task.trainer = type('obj', (object,), {
        'optimizers': lambda: [optimizers[0], optimizers[1]],
        'strategy': type('obj', (object,), {'root_device': torch.device('cpu')})(),
    })()
    
    # Run training step
    result = task.training_step(batch, 0)
    
    # Should return None (manual optimization)
    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/tasks/test_vqvae_task.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create VQVAETask**

```python
# src/maskgit3d/tasks/vqvae_task.py
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from hydra.utils import instantiate

from .base_task import BaseTask
from ..losses import GANLoss


class VQVAETask(BaseTask):
    """VQVAE training task with GAN-based adversarial training.
    
    Uses manual optimization for GAN training with dual optimizers:
    - Optimizer G: VQVAE parameters (Generator)
    - Optimizer D: Discriminator parameters
    
    Args:
        model_cfg: Hydra config for VQVAE model
        discriminator_cfg: Hydra config for Discriminator
        optimizer_cfg: Hydra config for optimizers
        scheduler_cfg: Hydra config for schedulers (optional)
        gan_mode: GAN loss type - "lsgan", "vanilla", or "hinge" (default: "lsgan")
        lambda_recon: Weight for reconstruction loss (default: 1.0)
        lambda_adv: Weight for adversarial loss (default: 0.1)
    """
    
    def __init__(
        self,
        model_cfg: DictConfig,
        discriminator_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        scheduler_cfg: Optional[DictConfig] = None,
        gan_mode: str = "lsgan",
        lambda_recon: float = 1.0,
        lambda_adv: float = 0.1,
    ):
        super().__init__(optimizer_cfg, scheduler_cfg)
        
        # Manual optimization for GAN training
        self.automatic_optimization = False
        
        # Models
        self.vqvae = instantiate(model_cfg)
        self.discriminator = instantiate(discriminator_cfg)
        
        # Losses
        self.gan_loss = GANLoss(gan_mode=gan_mode)
        
        # Loss weights
        self.lambda_recon = lambda_recon
        self.lambda_adv = lambda_adv
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through VQVAE."""
        return self.vqvae(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> None:
        """Training step with manual GAN optimization."""
        x, _ = batch
        
        # Get optimizers
        opt_g, opt_d = self.optimizers()
        
        # =====================
        # Train Discriminator
        # =====================
        x_recon, vq_loss, _ = self.vqvae(x)
        
        # Discriminator predictions
        real_pred = self.discriminator(x)[0][0]
        fake_pred = self.discriminator(x_recon.detach())[0][0]
        
        # Discriminator loss
        loss_d = self.gan_loss.discriminator_loss(real_pred, fake_pred)
        
        # Update discriminator
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()
        
        # =====================
        # Train Generator (VQVAE)
        # =====================
        x_recon, vq_loss, _ = self.vqvae(x)
        fake_pred = self.discriminator(x_recon)[0][0]
        
        # Generator losses
        loss_adv = self.gan_loss.generator_loss(fake_pred)
        loss_recon = F.l1_loss(x_recon, x)
        loss_g = self.lambda_adv * loss_adv + self.lambda_recon * loss_recon + vq_loss
        
        # Update generator
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()
        
        # Logging
        self.log_dict({
            "train/loss_d": loss_d,
            "train/loss_g": loss_g,
            "train/loss_recon": loss_recon,
            "train/loss_adv": loss_adv,
            "train/vq_loss": vq_loss,
        }, prog_bar=True)
    
    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Validation step."""
        x, _ = batch
        x_recon, vq_loss, _ = self.vqvae(x)
        
        loss_recon = F.l1_loss(x_recon, x)
        
        self.log_dict({
            "val/loss_recon": loss_recon,
            "val/vq_loss": vq_loss,
        }, prog_bar=True)
    
    def configure_optimizers(self) -> List[Any]:
        """Configure dual optimizers for GAN training."""
        opt_g = instantiate(self.optimizer_cfg, params=self.vqvae.parameters())
        opt_d = instantiate(self.optimizer_cfg, params=self.discriminator.parameters())
        return [opt_g, opt_d]
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/tasks/test_vqvae_task.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/maskgit3d/tasks/vqvae_task.py tests/unit/tasks/test_vqvae_task.py
git commit -m "feat(v2): create VQVAETask with manual GAN optimization"
```

---

## Phase 4: Data Layer (DataModule)

### Task 11: Create MedMNIST3DDataModule

**Files:**
- Read: `src/maskgit3d/infrastructure/data/medmnist3d_provider.py`
- Create: `src/maskgit3d/data/medmnist3d.py`
- Test: `tests/unit/data/test_medmnist3d.py`

**Step 1: Write the failing test**

```python
# tests/unit/data/test_medmnist3d.py
import pytest
import torch
from pathlib import Path
from src.maskgit3d.data.medmnist3d import MedMNIST3DDataModule


def test_medmnist3d_datamodule_setup():
    """Test MedMNIST3D DataModule setup."""
    datamodule = MedMNIST3DDataModule(
        data_dir="./data",
        batch_size=2,
        num_workers=0,
        image_size=28,
    )
    
    # Setup should create datasets
    datamodule.setup(stage="fit")
    
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None


def test_medmnist3d_datamodule_dataloaders():
    """Test MedMNIST3D DataModule dataloaders."""
    datamodule = MedMNIST3DDataModule(
        data_dir="./data",
        batch_size=2,
        num_workers=0,
        image_size=28,
    )
    
    datamodule.setup(stage="fit")
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    assert train_loader is not None
    assert val_loader is not None
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/data/test_medmnist3d.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create MedMNIST3DDataModule**

```python
# src/maskgit3d/data/medmnist3d.py
from typing import Optional, Sequence, Tuple

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import medmnist


class MedMNIST3DDataModule(LightningDataModule):
    """LightningDataModule for MedMNIST-3D dataset.
    
    Args:
        data_dir: Root directory for data storage
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        image_size: Target image size (D, H, W)
        pin_memory: Whether to pin memory for dataloaders
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 8,
        image_size: int = 28,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.pin_memory = pin_memory
        
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for train/val/test."""
        # Transforms
        train_transform = T.Compose([
            T.Lambda(lambda x: torch.from_numpy(x).float()),
            T.Resize(self.image_size),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        
        val_transform = T.Compose([
            T.Lambda(lambda x: torch.from_numpy(x).float()),
            T.Resize(self.image_size),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        
        # Download and create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = medmnist.OrganMNIST3D(
                root=self.data_dir,
                split="train",
                download=True,
                transform=train_transform,
            )
            self.val_dataset = medmnist.OrganMNIST3D(
                root=self.data_dir,
                split="val",
                download=True,
                transform=val_transform,
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = medmnist.OrganMNIST3D(
                root=self.data_dir,
                split="test",
                download=True,
                transform=val_transform,
            )
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/data/test_medmnist3d.py -v`
Expected: PASS (may need mock for actual download)

**Step 5: Commit**

```bash
git add src/maskgit3d/data/medmnist3d.py tests/unit/data/test_medmnist3d.py
git commit -m "feat(v2): create MedMNIST3DDataModule"
```

---

## Phase 5: Configuration Files

### Task 12: Create Hydra Configuration Files

**Files:**
- Create: `configs/train.yaml`
- Create: `configs/eval.yaml`
- Create: `configs/model/vqvae.yaml`
- Create: `configs/task/vqvae.yaml`
- Create: `configs/data/medmnist3d.yaml`
- Create: `configs/optimizer/adam.yaml`
- Create: `configs/scheduler/cosine.yaml`
- Create: `configs/trainer/default.yaml`
- Create: `configs/callbacks/default.yaml`
- Create: `configs/logger/tensorboard.yaml`

**Step 1: Create main train config**

```yaml
# configs/train.yaml
defaults:
  - _self_
  - model: vqvae
  - task: vqvae
  - data: medmnist3d
  - optimizer: adam
  - scheduler: cosine
  - trainer: default
  - callbacks: default
  - logger: tensorboard

seed: 42
ckpt_path: null
```

**Step 2: Create model configs**

```yaml
# configs/model/vqvae.yaml
_target_: src.maskgit3d.models.vqvae.VQVAE
spatial_dims: 3
in_channels: 1
embedding_dim: 256
num_embeddings: 8192
commitment_cost: 0.25
encoder_channels:
  - 32
  - 64
  - 128
  - 256
decoder_channels:
  - 256
  - 128
  - 64
  - 32
```

**Step 3: Create task configs**

```yaml
# configs/task/vqvae.yaml
_target_: src.maskgit3d.tasks.vqvae_task.VQVAETask
model_cfg: ${model}
discriminator_cfg:
  _target_: src.maskgit3d.models.discriminator.PatchDiscriminator3D
  in_channels: 1
  ndf: 64
  n_layers: 3
optimizer_cfg: ${optimizer}
scheduler_cfg: ${scheduler}
gan_mode: lsgan
lambda_recon: 1.0
lambda_adv: 0.1
```

**Step 4: Create data config**

```yaml
# configs/data/medmnist3d.yaml
_target_: src.maskgit3d.data.medmnist3d.MedMNIST3DDataModule
data_dir: ${oc.env:DATA_DIR,./data}
batch_size: 32
num_workers: 8
image_size: 28
pin_memory: true
```

**Step 5: Create optimizer config**

```yaml
# configs/optimizer/adam.yaml
_target_: torch.optim.Adam
lr: 1e-4
betas:
  - 0.9
  - 0.999
weight_decay: 0.0
```

**Step 6: Create scheduler config**

```yaml
# configs/scheduler/cosine.yaml
_target_: torch.optim.lr_scheduler.CosineAnnealingLR
T_max: 100
eta_min: 1e-6
```

**Step 7: Create trainer config**

```yaml
# configs/trainer/default.yaml
_target_: lightning.pytorch.Trainer
max_epochs: 100
accelerator: auto
devices: 1
precision: 32
log_every_n_steps: 20
gradient_clip_val: 1.0
accumulate_grad_batches: 1
```

**Step 8: Create callbacks config**

```yaml
# configs/callbacks/default.yaml
_target_: lightning.pytorch.callbacks.Callback
```

**Step 9: Create logger config**

```yaml
# configs/logger/tensorboard.yaml
_target_: lightning.pytorch.loggers.TensorBoardLogger
save_dir: ./logs
name: default
version: null
```

**Step 10: Create eval config**

```yaml
# configs/eval.yaml
defaults:
  - _self_
  - model: vqvae
  - task: vqvae
  - data: medmnist3d
  - trainer: default

ckpt_path: null
```

**Step 11: Commit**

```bash
git add configs/
git commit -m "feat(v2): create Hydra configuration files"
```

---

## Phase 6: Entry Points

### Task 13: Create train.py Entry Point

**Files:**
- Create: `src/maskgit3d/train.py`

**Step 1: Create train.py**

```python
# src/maskgit3d/train.py
"""Training entry point using Hydra configuration."""
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import lightning.pytorch as L


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg: Hydra configuration dictionary
    """
    # Set seed for reproducibility
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)
    
    # Instantiate components
    datamodule = instantiate(cfg.data)
    task = instantiate(cfg.task)
    callbacks = instantiate(cfg.callbacks)
    logger = instantiate(cfg.logger)
    
    # Instantiate trainer
    trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )
    
    # Train
    trainer.fit(
        task,
        datamodule=datamodule,
        ckpt_path=cfg.get("ckpt_path"),
    )


if __name__ == "__main__":
    main()
```

**Step 2: Test train.py can be imported**

Run: `python -c "from src.maskgit3d.train import main; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add src/maskgit3d/train.py
git commit -m "feat(v2): create train.py entry point"
```

---

### Task 14: Create eval.py Entry Point

**Files:**
- Create: `src/maskgit3d/eval.py`

**Step 1: Create eval.py**

```python
# src/maskgit3d/eval.py
"""Evaluation entry point using Hydra configuration."""
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import lightning.pytorch as L


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """Main evaluation function.
    
    Args:
        cfg: Hydra configuration dictionary
    """
    # Instantiate components
    datamodule = instantiate(cfg.data)
    task = instantiate(cfg.task)
    
    # Load checkpoint if specified
    if cfg.get("ckpt_path") is not None:
        task = task.load_from_checkpoint(cfg.ckpt_path)
    
    # Instantiate trainer
    trainer = instantiate(cfg.trainer)
    
    # Validate
    trainer.validate(task, datamodule=datamodule)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add src/maskgit3d/eval.py
git commit -m "feat(v2): create eval.py entry point"
```

---

## Phase 7: Metrics & Callbacks

### Task 15: Create Image Metrics

**Files:**
- Create: `src/maskgit3d/metrics/image_metrics.py`
- Test: `tests/unit/metrics/test_image_metrics.py`

**Step 1: Write the failing test**

```python
# tests/unit/metrics/test_image_metrics.py
import pytest
import torch
from src.maskgit3d.metrics.image_metrics import ImageMetrics


def test_image_metrics_ssim():
    """Test SSIM metric calculation."""
    metrics = ImageMetrics(spatial_dims=3)
    
    # Same images should have SSIM close to 1
    x = torch.randn(2, 1, 16, 16, 16)
    results = metrics.compute(x, x)
    
    assert "ssim" in results
    assert results["ssim"].item() > 0.99  # Should be very close to 1


def test_image_metrics_psnr():
    """Test PSNR metric calculation."""
    metrics = ImageMetrics(spatial_dims=3)
    
    # Same images should have very high PSNR
    x = torch.randn(2, 1, 16, 16, 16)
    results = metrics.compute(x, x)
    
    assert "psnr" in results
    assert results["psnr"].item() > 30  # Should be high for identical images
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/metrics/test_image_metrics.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create ImageMetrics**

```python
# src/maskgit3d/metrics/image_metrics.py
from typing import Dict

import torch
from monai.metrics import SSIMMetric, PSNRMetric


class ImageMetrics:
    """Image quality metrics using MONAI.
    
    Args:
        spatial_dims: Number of spatial dimensions (2 or 3)
    """
    
    def __init__(self, spatial_dims: int = 3):
        self.ssim_metric = SSIMMetric(spatial_dims=spatial_dims, data_range=1.0)
        self.psnr_metric = PSNRMetric(spatial_dims=spatial_dims, data_range=1.0)
    
    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute image quality metrics.
        
        Args:
            pred: Predicted images (B, C, D, H, W)
            target: Target images (B, C, D, H, W)
        
        Returns:
            Dictionary with SSIM and PSNR values
        """
        return {
            "ssim": self.ssim_metric(pred, target).mean(),
            "psnr": self.psnr_metric(pred, target).mean(),
        }
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/metrics/test_image_metrics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/maskgit3d/metrics/image_metrics.py tests/unit/metrics/test_image_metrics.py
git commit -m "feat(v2): create ImageMetrics using MONAI SSIM and PSNR"
```

---

## Summary

This implementation plan covers:

1. **Phase 1**: Foundation (Directory structure, VQVAE models)
2. **Phase 2**: Discriminator & Losses
3. **Phase 3**: Tasks Layer (BaseTask, VQVAETask with manual optimization)
4. **Phase 4**: Data Layer (MedMNIST3DDataModule)
5. **Phase 5**: Configuration Files (Hydra configs)
6. **Phase 6**: Entry Points (train.py, eval.py)
7. **Phase 7**: Metrics & Callbacks

**Total Tasks**: 15
**Estimated Time**: 2-3 hours for experienced engineer following TDD

**Next Steps after Phase 7**:
- Phase 8: MaskGIT Transformer model
- Phase 9: MaskGITTask
- Phase 10: Integration tests
- Phase 11: Remove v1 code
- Phase 12: Update documentation