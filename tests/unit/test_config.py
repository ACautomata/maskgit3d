"""
Unit tests for config modules.

These tests verify the functionality of configuration and dependency injection.
"""

import inspect

import pytest

from maskgit3d.config.modules import (
    MaisiVQModule,
    _validate_float_param,
    _validate_param,
    create_maskgit_module,
    create_vqgan_module,
)
from maskgit3d.infrastructure.maskgit.maskgit_model import MaskGITModel
from maskgit3d.infrastructure.vqgan import MaisiVQModel3D


def test_maskgit_model_requires_explicit_vqgan_and_transformer_dependencies():
    """MaskGITModel requires explicit vqgan and transformer dependencies."""
    with pytest.raises(TypeError):
        MaskGITModel()


class TestParameterValidation:
    """Tests for parameter validation functions."""

    def test_validate_param_valid(self):
        """Test valid parameter passes."""
        _validate_param("test", 10, min_val=1, max_val=100)
        _validate_param("test", 1, min_val=1)
        _validate_param("test", 100, max_val=100)

    def test_validate_param_below_min(self):
        """Test parameter below minimum raises error."""
        with pytest.raises(ValueError, match="must be >= 5"):
            _validate_param("test", 3, min_val=5)

    def test_validate_param_above_max(self):
        """Test parameter above maximum raises error."""
        with pytest.raises(ValueError, match="must be <= 10"):
            _validate_param("test", 15, max_val=10)

    def test_validate_float_param_valid(self):
        """Test valid float parameter passes."""
        _validate_float_param("test", 0.5, min_val=0.0, max_val=1.0)
        _validate_float_param("test", 1e-5, min_val=1e-10)

    def test_validate_float_param_below_min(self):
        """Test float parameter below minimum raises error."""
        with pytest.raises(ValueError, match="must be >= 0.1"):
            _validate_float_param("test", 0.05, min_val=0.1)


class TestCreateVQGANModule:
    """Tests for create_vqgan_module."""

    def test_create_vqgan_module_valid(self):
        """Test creating valid VQGAN module."""
        module = create_vqgan_module(
            image_size=64,
            in_channels=1,
            n_embed=256,
            embed_dim=32,
            latent_channels=32,
            lr=1e-4,
            batch_size=2,
            num_train=100,
            num_val=20,
        )
        assert module is not None

    def test_create_vqgan_module_invalid_image_size(self):
        """Test invalid image_size raises error."""
        with pytest.raises(ValueError, match="image_size must be >= 8"):
            create_vqgan_module(image_size=4)

    def test_create_vqgan_module_invalid_lr(self):
        """Test invalid learning rate raises error."""
        with pytest.raises(ValueError, match="lr must be >= 1e-10"):
            create_vqgan_module(lr=0)

    def test_create_vqgan_module_invalid_batch_size(self):
        """Test invalid batch_size raises error."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            create_vqgan_module(batch_size=0)

    def test_create_vqgan_module_defaults(self):
        """Test creating module with defaults."""
        module = create_vqgan_module()
        assert module is not None


class TestCreateVQGANModuleInstantiation:
    """Regression tests for VQGAN module model instantiation."""

    def test_create_vqgan_module_can_instantiate_model(self):
        """Factory module should build an instantiable VQModel3D."""
        module = create_vqgan_module(
            image_size=32,
            in_channels=1,
            n_embed=128,
            embed_dim=32,
            latent_channels=64,
        )

        model = module.model_module.provide_vqgan_model()

        assert model is not None
        assert model.codebook_size == 128

    def test_create_vqgan_module_default_model_instantiation(self):
        """Default VQGAN module config should instantiate without constructor mismatch."""
        module = create_vqgan_module()

        model = module.model_module.provide_vqgan_model()

        assert model is not None


class TestCreateMaskGITModule:
    """Tests for create_maskgit_module."""

    def test_create_maskgit_module_valid(self):
        """Test creating valid MaskGIT module."""
        module = create_maskgit_module(
            image_size=64,
            in_channels=1,
            codebook_size=256,
            embed_dim=32,
            latent_channels=32,
            transformer_hidden=128,
            transformer_layers=4,
            transformer_heads=4,
            mask_ratio=0.5,
            lr=1e-4,
            batch_size=2,
            num_train=100,
            num_val=20,
        )
        assert module is not None

    def test_create_maskgit_module_invalid_image_size(self):
        """Test invalid image_size raises error."""
        with pytest.raises(ValueError, match="image_size must be >= 8"):
            create_maskgit_module(image_size=4)

    def test_create_maskgit_module_invalid_mask_ratio(self):
        """Test invalid mask_ratio raises error."""
        with pytest.raises(ValueError, match="mask_ratio must be >= 0.0"):
            create_maskgit_module(mask_ratio=-0.1)

    def test_create_maskgit_module_invalid_mask_ratio_high(self):
        """Test mask_ratio > 1.0 raises error."""
        with pytest.raises(ValueError, match="mask_ratio must be <= 1.0"):
            create_maskgit_module(mask_ratio=1.5)

    def test_create_maskgit_module_invalid_lr(self):
        """Test invalid learning rate raises error."""
        with pytest.raises(ValueError, match="lr must be >= 1e-10"):
            create_maskgit_module(lr=-1e-4)

    def test_create_maskgit_module_defaults(self):
        """Test creating module with defaults."""
        module = create_maskgit_module()
        assert module is not None


class TestMaskGITModuleInstantiation:
    """Regression tests for MaskGIT module runtime wiring."""

    def test_create_maskgit_module_can_instantiate_model(self):
        """Factory module should build an instantiable MaskGITModel."""
        module = create_maskgit_module(
            image_size=32,
            in_channels=1,
            codebook_size=128,
            embed_dim=32,
            latent_channels=64,
            transformer_hidden=128,
            transformer_layers=2,
            transformer_heads=4,
        )

        model = module.model_module.provide_maskgit_model()

        assert model is not None

    def test_maskgit_model_uses_quantize_attribute(self):
        """MaskGITModel should call VQModel3D quantize attribute."""
        source = inspect.getsource(MaskGITModel)

        assert "self.vqgan.quantize(" in source
        assert "self.vqgan.quantize.get_codebook_entry" in source
        assert "self.vqgan.quantizer" not in source


def test_maisi_vq_module_binds_maisi_class_name():
    """Guard: MaisiVQModule must bind MaisiVQModel3D, not a typo variant."""
    module = MaisiVQModule(model_config={"type": "maisi_vq", "params": {}})
    model = module.model_module.provide_maisi_vq_model()
    assert model.__class__ is MaisiVQModel3D


class TestOptimizerConfigDataclass:
    """Tests for OptimizerConfig dataclass."""

    def test_optimizer_config_dataclass_defaults_are_stable(self):
        """Test OptimizerConfig dataclass defaults are stable."""
        from maskgit3d.config.schemas import OptimizerConfig

        cfg = OptimizerConfig(type="adam", params={"lr": 1e-4})
        assert cfg.type == "adam"
        assert cfg.params["lr"] == 1e-4
