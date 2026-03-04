"""Shared pytest fixtures for maskgit-3d tests."""


import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def synthetic_3d_tensor():
    """Returns a small 3D tensor (B, C, D, H, W) for fast testing."""
    return torch.randn(2, 1, 16, 16, 16)


@pytest.fixture(scope="session")
def synthetic_3d_batch():
    """Returns a tuple of (image, label) tensors."""
    image = torch.randn(2, 1, 16, 16, 16)
    label = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()
    return image, label


@pytest.fixture(scope="session")
def synthetic_3d_volume():
    """Returns a single 3D volume (C, D, H, W) for model testing."""
    return torch.randn(1, 32, 32, 32)


@pytest.fixture
def temp_nifti_dir(tmp_path_factory):
    """Creates a temporary directory with synthetic NIfTI files."""
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("nibabel not installed")

    d = tmp_path_factory.mktemp("nifti_data")

    # Create dummy image
    data = np.random.rand(16, 16, 16).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, str(d / "image_001.nii.gz"))

    # Create dummy label
    label_data = np.random.randint(0, 2, (16, 16, 16)).astype(np.float32)
    label_img = nib.Nifti1Image(label_data, np.eye(4))
    nib.save(label_img, str(d / "label_001.nii.gz"))

    return d


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Returns a temporary directory for checkpoint testing."""
    return tmp_path


@pytest.fixture
def device():
    """Returns the appropriate device for testing (cpu)."""
    return torch.device("cpu")



@pytest.fixture
def small_vqgan_config():
    """Returns a minimal VQGAN configuration for fast testing."""
    return {
        "in_channels": 1,
        "out_channels": 1,
        "hidden_channels": 32,
        "channel_multipliers": (1, 2),
        "num_res_blocks": 1,
        "resolution": 16,
        "attn_resolutions": (),
        "dropout": 0.0,
        "codebook_size": 64,
        "embed_dim": 16,
    }


@pytest.fixture
def small_maskgit_config():
    """Returns a minimal MaskGIT configuration for fast testing."""
    return {
        "vocab_size": 128,
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 2,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "max_seq_len": 256,
    }
