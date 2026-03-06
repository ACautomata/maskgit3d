"""Integration test for crop_size and roi_size configuration."""

import pytest
from omegaconf import OmegaConf

from maskgit3d.infrastructure.data.medmnist_provider import MedMnist3DDataProvider


@pytest.mark.integration
def test_crop_roi_config_integration():
    """Test that crop_size and roi_size flow from config to provider."""
    # Simulate config from YAML
    cfg = OmegaConf.create(
        {
            "dataset": {
                "type": "medmnist3d",
                "dataset_name": "organ",
                "crop_size": [48, 48, 48],
                "roi_size": [64, 64, 64],
                "batch_size": 2,
                "num_workers": 0,
                "data_dir": "./data",
            }
        }
    )

    # Create provider
    provider = MedMnist3DDataProvider(
        dataset_type=cfg.dataset.dataset_name,
        crop_size=tuple(cfg.dataset.crop_size),
        roi_size=tuple(cfg.dataset.roi_size),
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        data_root=cfg.dataset.data_dir,
        download=False,
    )

    # Verify crop_size and roi_size are set correctly
    assert provider.crop_size == (48, 48, 48), f"Expected (48, 48, 48), got {provider.crop_size}"
    assert provider.roi_size == (64, 64, 64), f"Expected (64, 64, 64), got {provider.roi_size}"


@pytest.mark.integration
def test_default_crop_roi_sizes():
    """Test that default crop_size and roi_size work when not specified."""
    # Simulate config without explicit crop_size/roi_size
    cfg = OmegaConf.create(
        {
            "dataset": {
                "type": "medmnist3d",
                "dataset_name": "organ",
                "spatial_size": [64, 64, 64],
                "batch_size": 2,
                "num_workers": 0,
                "data_dir": "./data",
            }
        }
    )

    # Create provider without explicit crop_size/roi_size
    provider = MedMnist3DDataProvider(
        dataset_type=cfg.dataset.dataset_name,
        spatial_size=tuple(cfg.dataset.spatial_size),
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        data_root=cfg.dataset.data_dir,
        download=False,
    )

    # Verify defaults (should equal spatial_size)
    assert provider.crop_size == (64, 64, 64), (
        f"Expected default (64, 64, 64), got {provider.crop_size}"
    )
    assert provider.roi_size == (64, 64, 64), (
        f"Expected default (64, 64, 64), got {provider.roi_size}"
    )


@pytest.mark.integration
def test_crop_roi_with_different_values():
    """Test various crop_size and roi_size configurations."""
    test_cases = [
        # (crop_size, roi_size)
        ((32, 32, 32), (64, 64, 64)),
        ((48, 48, 48), (64, 64, 64)),
        ((56, 56, 56), (64, 64, 64)),
        ((64, 64, 64), (64, 64, 64)),
    ]

    for crop_size, roi_size in test_cases:
        provider = MedMnist3DDataProvider(
            dataset_type="organ",
            crop_size=crop_size,
            roi_size=roi_size,
            batch_size=1,
            num_workers=0,
            data_root="./data",
            download=False,
        )

        assert provider.crop_size == crop_size, (
            f"crop_size mismatch: expected {crop_size}, got {provider.crop_size}"
        )
        assert provider.roi_size == roi_size, (
            f"roi_size mismatch: expected {roi_size}, got {provider.roi_size}"
        )
