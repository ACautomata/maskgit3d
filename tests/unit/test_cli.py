"""Tests for CLI modules to improve coverage."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestCliMain:
    """Tests for CLI main module."""

    @pytest.fixture
    def mock_train_main(self):
        with patch("maskgit3d.cli.train.main") as mock:
            yield mock

    @pytest.fixture
    def mock_test_main(self):
        with patch("maskgit3d.cli.test.main") as mock:
            yield mock

    def test_main_prints_help_no_command(self, capsys):
        """Test main prints help when no command given."""
        from maskgit3d.cli.main import main

        with pytest.raises(SystemExit) as exc_info, patch.object(sys, "argv", ["maskgit3d"]):
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out or "MaskGIT" in captured.out

    def test_train_command(self, mock_train_main):
        """Test train command calls train_main."""
        from maskgit3d.cli.main import main

        with patch.object(sys, "argv", ["maskgit3d", "train", "model=maskgit"]):
            main()

        mock_train_main.assert_called_once()

    def test_test_command(self, mock_test_main):
        """Test test command calls test_main."""
        from maskgit3d.cli.main import main

        with patch.object(sys, "argv", ["maskgit3d", "test", "model=maskgit"]):
            main()

        mock_test_main.assert_called_once()

    def test_config_dir_override(self, mock_train_main):
        """Test --config-dir override sets environment variable."""
        from maskgit3d.cli.main import main

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(
                sys, "argv", ["maskgit3d", "--config-dir", "/custom/config", "train"]
            ):
                main()

            import os

            assert os.environ.get("HYDRA_CONFIG_PATH") == "/custom/config"

    def test_config_name_override(self, mock_train_main):
        """Test --config-name option is accepted."""
        from maskgit3d.cli.main import main

        with patch.object(sys, "argv", ["maskgit3d", "--config-name", "custom", "train"]):
            main()

        # Just verify it doesn't raise an error
        mock_train_main.assert_called_once()


class TestCliTrain:
    """Tests for CLI train module."""

    def test_extract_factory_params(self):
        """Test _extract_factory_params function."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _extract_factory_params

        cfg = OmegaConf.create(
            {
                "model": {
                    "image_size": 64,
                    "in_channels": 1,
                    "embed_dim": 256,
                    "latent_channels": 256,
                },
                "dataset": {"batch_size": 4},
                "training": {"optimizer": {"lr": 0.001}},
            }
        )

        params = _extract_factory_params(cfg)

        assert params["image_size"] == 64
        assert params["in_channels"] == 1
        assert params["embed_dim"] == 256
        assert params["latent_channels"] == 256
        assert params["lr"] == 0.001
        assert params["batch_size"] == 4

    def test_create_data_config_simple(self):
        """Test _create_data_config for simple dataset."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_data_config

        cfg = OmegaConf.create(
            {
                "dataset": {
                    "type": "simple",
                    "batch_size": 4,
                    "num_train": 100,
                    "num_val": 20,
                    "num_test": 20,
                },
                "model": {"image_size": 64, "in_channels": 1},
            }
        )

        data_config = _create_data_config(cfg)

        assert data_config["type"] == "simple"
        assert data_config["params"]["batch_size"] == 4
        assert data_config["params"]["num_train"] == 100

    def test_create_data_config_medmnist3d(self):
        """Test _create_data_config for medmnist3d dataset."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_data_config

        cfg = OmegaConf.create(
            {
                "dataset": {
                    "type": "medmnist3d",
                    "batch_size": 4,
                    "dataset_name": "organmnist3d",
                    "data_dir": "./data",
                },
                "model": {"image_size": 64, "in_channels": 1},
            }
        )

        data_config = _create_data_config(cfg)

        assert data_config["type"] == "medmnist3d"
        assert data_config["params"]["dataset_type"] == "organ"

    def test_create_data_config_brats(self):
        """Test _create_data_config for brats dataset."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_data_config

        cfg = OmegaConf.create(
            {
                "dataset": {
                    "type": "brats",
                    "batch_size": 4,
                    "data_dir": "./data/brats",
                    "train_ratio": 0.7,
                },
                "model": {"image_size": 64, "in_channels": 4},
            }
        )

        data_config = _create_data_config(cfg)

        assert data_config["type"] == "brats"
        assert data_config["params"]["train_ratio"] == 0.7

    def test_create_data_config_unknown_type(self):
        """Test _create_data_config raises error for unknown type."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_data_config

        cfg = OmegaConf.create(
            {
                "dataset": {"type": "unknown", "batch_size": 4},
                "model": {"image_size": 64, "in_channels": 1},
            }
        )

        with pytest.raises(ValueError, match="Unknown dataset type"):
            _create_data_config(cfg)

    def test_create_model_params_maskgit(self):
        """Test _create_model_params for maskgit model."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_model_params

        cfg = OmegaConf.create(
            {
                "model": {
                    "type": "maskgit",
                    "codebook_size": 1024,
                    "channel_multipliers": [1, 1, 2, 2, 4],
                    "transformer_hidden": 768,
                    "transformer_layers": 12,
                    "transformer_heads": 12,
                    "mask_schedule_type": "cosine",
                }
            }
        )
        base_params = {
            "image_size": 64,
            "in_channels": 1,
            "embed_dim": 256,
            "latent_channels": 256,
        }

        params = _create_model_params(cfg, "maskgit", base_params)

        assert params["codebook_size"] == 1024
        assert params["transformer_hidden"] == 768

    def test_create_model_params_vqgan(self):
        """Test _create_model_params for vqgan model."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_model_params

        cfg = OmegaConf.create(
            {
                "model": {
                    "type": "vqgan",
                    "codebook_size": 1024,
                    "channel_multipliers": [1, 2],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                }
            }
        )
        base_params = {
            "image_size": 64,
            "in_channels": 1,
            "embed_dim": 256,
            "latent_channels": 256,
        }

        params = _create_model_params(cfg, "vqgan", base_params)

        assert params["codebook_size"] == 1024
        assert "num_channels" in params
        assert "attention_levels" in params

    def test_create_model_params_unknown(self):
        """Test _create_model_params raises error for unknown type."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_model_params

        cfg = OmegaConf.create({"model": {"type": "unknown"}})
        base_params = {}

        with pytest.raises(ValueError, match="Unknown model type"):
            _create_model_params(cfg, "unknown", base_params)

    def test_create_training_config_vqgan(self):
        """Test _create_training_config for vqgan."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_training_config

        cfg = OmegaConf.create(
            {
                "model": {"type": "vqgan"},
                "training": {
                    "vqgan": {
                        "codebook_weight": 1.0,
                        "pixel_loss_weight": 1.0,
                        "perceptual_weight": 1.0,
                        "disc_weight": 0.1,
                        "disc_start": 500,
                    }
                },
            }
        )

        config = _create_training_config(cfg, "vqgan")

        assert config["type"] == "vqgan"
        assert config["params"]["codebook_weight"] == 1.0

    def test_create_training_config_maskgit(self):
        """Test _create_training_config for maskgit."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_training_config

        cfg = OmegaConf.create({"model": {"type": "maskgit", "mask_schedule_type": "cosine"}})

        config = _create_training_config(cfg, "maskgit")

        assert config["type"] == "maskgit"
        assert config["params"]["mask_schedule_type"] == "cosine"

    def test_create_optimizer_config(self):
        """Test _create_optimizer_config function."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_optimizer_config

        cfg = OmegaConf.create({"training": {"optimizer": {"type": "adam", "lr": 0.001}}})

        config = _create_optimizer_config(cfg)

        assert config["type"] == "adam"
        assert config["params"]["lr"] == 0.001

    def test_create_inference_config(self):
        """Test _create_inference_config function."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_inference_config

        cfg = OmegaConf.create({"model": {"type": "maskgit"}})

        config = _create_inference_config(cfg, "maskgit")

        assert config["type"] == "maskgit"
        assert config["params"]["mode"] == "reconstruct"


class TestCliTest:
    """Tests for CLI test module."""

    @patch("maskgit3d.cli.test.run_testing")
    def test_main(self, mock_run_testing):
        """Test main function in test module."""
        from maskgit3d.cli.test import main

        with patch.object(sys, "argv", ["test", "model=maskgit"]):
            main()

        mock_run_testing.assert_called_once()

    @patch("maskgit3d.cli.test.FabricTestPipeline")
    @patch("maskgit3d.cli.test.create_module_from_config")
    @patch("maskgit3d.cli.test.Injector")
    @patch("maskgit3d.cli.test.HydraConfig")
    def test_run_testing(
        self, mock_hydra_config, mock_injector_class, mock_create_module, mock_pipeline_class
    ):
        """Test run_testing function."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.test import run_testing

        # Setup mocks
        mock_module = MagicMock()
        mock_create_module.return_value = mock_module
        mock_injector = MagicMock()
        mock_injector_class.return_value = mock_injector
        mock_injector.get.return_value = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        # Mock HydraConfig
        mock_runtime = MagicMock()
        mock_runtime.output_dir = "/tmp/test_output"
        mock_hydra_config.get.return_value.runtime = mock_runtime

        cfg = OmegaConf.create(
            {
                "model": {"type": "maskgit"},
                "dataset": {"type": "simple"},
                "output": {"output_dir": "./outputs"},
                "checkpoint": {"load_from": "./checkpoint.ckpt"},
                "training": {"fabric": {}},
            }
        )

        run_testing(cfg)

        mock_create_module.assert_called_once_with(cfg)
        mock_pipeline_class.assert_called_once()
        mock_pipeline.run.assert_called_once()


class TestCreateDataConfigCropRoiSize:
    """Tests for crop_size and roi_size extraction in _create_data_config."""

    def test_create_data_config_extracts_crop_and_roi_from_medmnist3d(self):
        """Test that CLI extracts crop_size and roi_size from medmnist3d config."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_data_config

        cfg = OmegaConf.create(
            {
                "dataset": {
                    "type": "medmnist3d",
                    "crop_size": [32, 32, 32],
                    "roi_size": [64, 64, 64],
                    "batch_size": 8,
                    "dataset_name": "organmnist3d",
                    "data_dir": "./data",
                },
                "model": {"image_size": 64},
            }
        )

        data_config = _create_data_config(cfg)
        params = data_config["params"]

        assert params["crop_size"] == (32, 32, 32)
        assert params["roi_size"] == (64, 64, 64)

    def test_create_data_config_extracts_crop_and_roi_from_brats(self):
        """Test that CLI extracts crop_size and roi_size from brats config."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_data_config

        cfg = OmegaConf.create(
            {
                "dataset": {
                    "type": "brats",
                    "crop_size": [48, 48, 48],
                    "roi_size": [128, 128, 128],
                    "batch_size": 2,
                    "data_dir": "/data/brats",
                },
                "model": {"image_size": 64},
            }
        )

        data_config = _create_data_config(cfg)
        params = data_config["params"]

        assert params["crop_size"] == (48, 48, 48)
        assert params["roi_size"] == (128, 128, 128)

    def test_create_data_config_fallback_to_image_size(self):
        """Test that crop_size and roi_size fallback to image_size when not specified."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_data_config

        cfg = OmegaConf.create(
            {
                "dataset": {
                    "type": "simple",
                    "batch_size": 4,
                },
                "model": {"image_size": 64},
            }
        )

        data_config = _create_data_config(cfg)
        params = data_config["params"]

        assert params["crop_size"] == (64, 64, 64)
        assert params["roi_size"] == (64, 64, 64)

    def test_create_data_config_custom_fallback_image_size(self):
        """Test that crop_size and roi_size use custom image_size fallback."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_data_config

        cfg = OmegaConf.create(
            {
                "dataset": {
                    "type": "simple",
                    "batch_size": 4,
                },
                "model": {"image_size": 128},
            }
        )

        data_config = _create_data_config(cfg)
        params = data_config["params"]

        assert params["crop_size"] == (128, 128, 128)
        assert params["roi_size"] == (128, 128, 128)

    def test_create_data_config_tuple_values_pass_through(self):
        """Test that tuple values in config are passed through correctly."""
        from omegaconf import OmegaConf

        from maskgit3d.cli.train import _create_data_config

        cfg = OmegaConf.create(
            {
                "dataset": {
                    "type": "simple",
                    "crop_size": (32, 32, 32),
                    "roi_size": (64, 64, 64),
                    "batch_size": 4,
                },
                "model": {"image_size": 64},
            }
        )

        data_config = _create_data_config(cfg)
        params = data_config["params"]

        assert params["crop_size"] == (32, 32, 32)
        assert params["roi_size"] == (64, 64, 64)
