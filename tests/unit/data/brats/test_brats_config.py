"""Tests for BraTS2023 configuration."""

import pytest

from maskgit3d.data.brats.config import (
    MODALITY_ORDER,
    BraTS2023Config,
    BraTSSubDataset,
)


class TestBraTSSubDataset:
    """Test BraTS sub-dataset enum."""

    def test_enum_values(self) -> None:
        """Test that enum has correct values."""
        assert BraTSSubDataset.GLI.value == "gli"
        assert BraTSSubDataset.MEN.value == "men"
        assert BraTSSubDataset.MET.value == "met"

    def test_enum_membership(self) -> None:
        """Test that all expected members exist."""
        assert hasattr(BraTSSubDataset, "GLI")
        assert hasattr(BraTSSubDataset, "MEN")
        assert hasattr(BraTSSubDataset, "MET")


class TestModalityOrder:
    """Test modality order constant."""

    def test_modality_order_fixed(self) -> None:
        """Test that modality order is fixed and correct."""
        expected = ("t1n", "t1c", "t2w", "t2f")
        assert expected == MODALITY_ORDER
        assert len(MODALITY_ORDER) == 4

    def test_modality_order_immutable(self) -> None:
        """Test that modality order cannot be accidentally modified."""
        # Tuple is immutable, this test documents that behavior
        assert isinstance(MODALITY_ORDER, tuple)


class TestBraTS2023Config:
    """Test BraTS2023 configuration dataclass."""

    def test_default_config_creation(self) -> None:
        """Test creating config with defaults."""
        config = BraTS2023Config()
        assert config.data_dir == "/data/dataset/"
        assert config.crop_size == (128, 128, 128)
        assert config.batch_size == 2
        assert config.num_workers == 4
        assert config.pin_memory is True
        assert config.train_ratio == 0.8
        assert config.seed == 42
        assert config.subdatasets == [BraTSSubDataset.GLI, BraTSSubDataset.MEN, BraTSSubDataset.MET]
        assert config.drop_last_train is True

    def test_custom_config_creation(self) -> None:
        """Test creating config with custom values."""
        config = BraTS2023Config(
            data_dir="/custom/path",
            crop_size=(64, 64, 64),
            batch_size=4,
            num_workers=8,
            train_ratio=0.9,
            seed=123,
            subdatasets=[BraTSSubDataset.GLI],
        )
        assert config.data_dir == "/custom/path"
        assert config.crop_size == (64, 64, 64)
        assert config.batch_size == 4
        assert config.num_workers == 8
        assert config.train_ratio == 0.9
        assert config.seed == 123
        assert config.subdatasets == [BraTSSubDataset.GLI]

    def test_modality_order_property(self) -> None:
        """Test that config exposes modality order."""
        config = BraTS2023Config()
        assert config.modality_order == MODALITY_ORDER
        assert len(config.modality_order) == 4

    def test_num_modalities_property(self) -> None:
        """Test that config exposes number of modalities."""
        config = BraTS2023Config()
        assert config.num_modalities == 4

    def test_invalid_crop_size_not_multiple_of_16(self) -> None:
        """Test that crop size not divisible by 16 raises error."""
        with pytest.raises(ValueError, match="divisible by 16"):
            BraTS2023Config(crop_size=(100, 100, 100))

    def test_invalid_crop_size_wrong_dimensions(self) -> None:
        """Test that crop size with wrong dimensions raises error."""
        with pytest.raises(ValueError, match="3-element tuple"):
            BraTS2023Config(crop_size=(128, 128))

    def test_invalid_train_ratio_too_high(self) -> None:
        """Test that train_ratio > 1 raises error."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            BraTS2023Config(train_ratio=1.5)

    def test_invalid_train_ratio_too_low(self) -> None:
        """Test that train_ratio <= 0 raises error."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            BraTS2023Config(train_ratio=0.0)

        with pytest.raises(ValueError, match="between 0 and 1"):
            BraTS2023Config(train_ratio=-0.1)

    def test_stratification_enabled_by_default(self) -> None:
        """Test that stratification is enabled by default."""
        config = BraTS2023Config()
        assert config.stratify is True

    def test_subdatasets_all_by_default(self) -> None:
        """Test that all subdatasets are included by default."""
        config = BraTS2023Config()
        assert BraTSSubDataset.GLI in config.subdatasets
        assert BraTSSubDataset.MEN in config.subdatasets
        assert BraTSSubDataset.MET in config.subdatasets
