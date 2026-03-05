"""Tests for config/modules.py to improve coverage."""


import pytest

from maskgit3d.config.modules import (
    DataModule,
    InferenceModule,
    MaskGITModelModule,
    ModelModule,
    TrainingModule,
)


class TestDataModule:
    """Tests for DataModule."""

    def test_data_module_simple(self):
        """Test DataModule with simple dataset type."""
        config = {
            "type": "simple",
            "params": {
                "batch_size": 2,
                "spatial_size": (8, 8, 8),
                "num_workers": 0,
            },
        }

        module = DataModule(config)
        assert module is not None

    def test_data_module_medmnist(self):
        """Test DataModule with medmnist3d dataset type."""
        config = {
            "type": "medmnist3d",
            "params": {
                "batch_size": 2,
                "spatial_size": (8, 8, 8),
                "dataset_type": "organ",
                "input_size": 28,
                "data_root": "./data",
            },
        }

        module = DataModule(config)
        assert module is not None

    def test_data_module_brats(self):
        """Test DataModule with brats dataset type."""
        config = {
            "type": "brats",
            "params": {
                "batch_size": 1,
                "spatial_size": (8, 8, 8),
                "data_dir": "./data/brats",
                "modalities": ["t1", "t2"],
            },
        }

        module = DataModule(config)
        assert module is not None

    def test_data_module_unknown(self):
        """Test DataModule with unknown dataset type raises error."""
        from injector import Injector

        from maskgit3d.domain.interfaces import DataProvider

        config = {"type": "unknown", "params": {}}
        module = DataModule(config)
        with pytest.raises(ValueError, match="Unknown data type"):
            Injector([module]).get(DataProvider)


class TestModelModule:
    """Tests for ModelModule."""

    def test_model_module_vqgan(self):
        """Test ModelModule with vqgan model type."""
        config = {
            "type": "vqgan",
            "params": {
                "in_channels": 1,
                "codebook_size": 512,
                "embed_dim": 128,
                "latent_channels": 128,
                "num_channels": [32, 64],
                "num_res_blocks": [2, 2],
                "attention_levels": [False, False],
            },
        }

        module = ModelModule(config)
        assert module is not None

    def test_model_module_unknown(self):
        """Test ModelModule with unknown model type raises error."""
        from injector import Injector

        from maskgit3d.domain.interfaces import ModelInterface

        config = {"type": "unknown", "params": {}}
        module = ModelModule(config)
        with pytest.raises(ValueError, match="Unknown model type"):
            Injector([module]).get(ModelInterface)


class TestMaskGITModelModule:
    """Tests for MaskGITModelModule."""

    def test_maskgit_model_module_creation(self):
        """Test MaskGITModelModule creation."""
        model_config = {
            "type": "maskgit",
            "params": {
                "in_channels": 1,
                "codebook_size": 512,
                "embed_dim": 128,
                "latent_channels": 128,
                "resolution": 16,
            },
        }

        module = MaskGITModelModule(
            model_config=model_config,
            pretrained_vqvae_path=None,
            freeze_vqvae=True,
        )
        assert module is not None


class TestTrainingModule:
    """Tests for TrainingModule."""

    def test_training_module_maskgit(self):
        """Test TrainingModule with maskgit strategy."""
        training_config = {
            "type": "maskgit",
            "params": {"mask_ratio": 0.5, "reconstruction_weight": 1.0},
        }
        optimizer_config = {"type": "adam", "params": {"lr": 0.001}}

        module = TrainingModule(training_config, optimizer_config)
        assert module is not None

    def test_training_module_vqgan(self):
        """Test TrainingModule with vqgan strategy."""
        training_config = {
            "type": "vqgan",
            "params": {
                "codebook_weight": 1.0,
                "pixel_loss_weight": 1.0,
                "perceptual_weight": 0.0,
                "disc_weight": 0.1,
                "disc_start": 0,
            },
        }
        optimizer_config = {"type": "adam", "params": {"lr": 0.001}}

        module = TrainingModule(training_config, optimizer_config)
        assert module is not None

    def test_training_module_unknown(self):
        """Test TrainingModule with unknown strategy type raises error."""
        from injector import Injector

        from maskgit3d.domain.interfaces import TrainingStrategy

        training_config = {"type": "unknown", "params": {}}
        optimizer_config = {"type": "adam", "params": {"lr": 0.001}}
        module = TrainingModule(training_config, optimizer_config)
        with pytest.raises(ValueError, match="Unknown strategy type"):
            Injector([module]).get(TrainingStrategy)


class TestInferenceModule:
    """Tests for InferenceModule."""

    def test_inference_module_maskgit(self):
        """Test InferenceModule with maskgit type."""
        config = {"type": "maskgit", "params": {"mode": "generate", "num_iterations": 8}}

        module = InferenceModule(config)
        assert module is not None

    def test_inference_module_vqgan(self):
        """Test InferenceModule with vqgan type."""
        config = {"type": "vqgan", "params": {"mode": "reconstruct"}}

        module = InferenceModule(config)
        assert module is not None

    def test_inference_module_unknown(self):
        """Test InferenceModule with unknown type raises error."""
        from injector import Injector

        from maskgit3d.domain.interfaces import InferenceStrategy

        config = {"type": "unknown", "params": {}}
        module = InferenceModule(config)
        with pytest.raises(ValueError, match="Unknown inference type"):
            Injector([module]).get(InferenceStrategy)
