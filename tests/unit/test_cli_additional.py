"""Additional tests for CLI modules to improve coverage."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf


class TestCliTrainAdditional:
    """Additional tests for CLI train module."""

    def test_extract_factory_params_defaults(self):
        """Test _extract_factory_params with defaults."""
        from maskgit3d.cli.train import _extract_factory_params

        cfg = OmegaConf.create(
            {
                "model": {},
                "dataset": {"batch_size": 4},
                "training": {"optimizer": {"lr": 0.001}},
            }
        )

        params = _extract_factory_params(cfg)

        assert params["image_size"] == 64  # default
        assert params["in_channels"] == 1  # default
        assert params["embed_dim"] == 256  # default
        assert params["latent_channels"] == 256  # default
        assert params["lr"] == 0.001
        assert params["batch_size"] == 4

    def test_create_data_config_medmnist3d_organ(self):
        """Test _create_data_config for medmnist3d with organmnist3d."""
        from maskgit3d.cli.train import _create_data_config

        cfg = OmegaConf.create(
            {
                "dataset": {
                    "type": "medmnist3d",
                    "batch_size": 4,
                    "dataset_name": "organmnist3d",
                    "data_dir": "./data",
                    "download": True,
                },
                "model": {"image_size": 64, "in_channels": 1},
            }
        )

        data_config = _create_data_config(cfg)

        assert data_config["type"] == "medmnist3d"
        assert data_config["params"]["dataset_type"] == "organ"

    def test_create_data_config_medmnist3d_nodule(self):
        """Test _create_data_config for medmnist3d with nodulemnist3d."""
        from maskgit3d.cli.train import _create_data_config

        cfg = OmegaConf.create(
            {
                "dataset": {
                    "type": "medmnist3d",
                    "batch_size": 4,
                    "dataset_name": "nodulemnist3d",
                },
                "model": {"image_size": 64, "in_channels": 1},
            }
        )

        data_config = _create_data_config(cfg)

        assert data_config["type"] == "medmnist3d"
        assert data_config["params"]["dataset_type"] == "nodule"

    def test_create_data_config_organ_direct(self):
        """Test _create_data_config with organ type directly."""
        from maskgit3d.cli.train import _create_data_config

        cfg = OmegaConf.create(
            {
                "dataset": {"type": "organ", "batch_size": 4},
                "model": {"image_size": 64, "in_channels": 1},
            }
        )

        data_config = _create_data_config(cfg)

        # Should return medmnist3d with organ type
        assert data_config["type"] == "medmnist3d"
        assert data_config["params"]["dataset_type"] == "organ"

    def test_create_model_params_vqgan3d(self):
        """Test _create_model_params for vqgan3d."""
        from maskgit3d.cli.train import _create_model_params

        cfg = OmegaConf.create(
            {
                "model": {
                    "type": "vqgan3d",
                    "codebook_size": 1024,
                    "channel_multipliers": [1, 2, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [16, 8],
                }
            }
        )
        base_params = {
            "image_size": 64,
            "in_channels": 1,
            "embed_dim": 256,
            "latent_channels": 256,
        }

        params = _create_model_params(cfg, "vqgan3d", base_params)

        assert params["codebook_size"] == 1024
        assert "num_channels" in params
        assert "attention_levels" in params

    def test_create_model_params_maisi_vq(self):
        """Test _create_model_params for maisi_vq."""
        from maskgit3d.cli.train import _create_model_params

        cfg = OmegaConf.create(
            {"model": {"type": "maisi_vq", "codebook_size": 1024, "latent_channels": 4}}
        )
        base_params = {
            "image_size": 64,
            "in_channels": 1,
            "embed_dim": 256,
            "latent_channels": 256,
        }

        with patch("maskgit3d.infrastructure.vqgan.get_maisi_vq_config") as mock_get_config:
            mock_get_config.return_value = {"mock": "config"}
            params = _create_model_params(cfg, "maisi_vq", base_params)

        assert params == {"mock": "config"}
        mock_get_config.assert_called_once()

    def test_create_training_config_default_optimizer(self):
        """Test _create_optimizer_config with defaults."""
        from maskgit3d.cli.train import _create_optimizer_config

        cfg = OmegaConf.create({"training": {"optimizer": {}}})

        config = _create_optimizer_config(cfg)

        assert config["type"] == "adam"  # default
        assert config["params"]["lr"] == 1e-4  # default


class TestCliTestAdditional:
    """Additional tests for CLI test module."""

    def test_run_testing_without_metrics(self):
        """Test run_testing without metrics."""
        from maskgit3d.cli.test import run_testing

        cfg = OmegaConf.create(
            {
                "model": {"type": "maskgit"},
                "dataset": {"type": "simple"},
                "output": {"output_dir": "./outputs"},
                "checkpoint": {},
                "training": {"fabric": {}},
            }
        )

        with patch("maskgit3d.cli.test.create_module_from_config") as mock_create:
            mock_module = MagicMock()
            mock_create.return_value = mock_module

            with patch("maskgit3d.cli.test.Injector") as mock_injector_class:
                mock_injector = MagicMock()
                mock_injector_class.return_value = mock_injector
                mock_injector.get.return_value = MagicMock()

                with patch("maskgit3d.cli.test.FabricTestPipeline") as mock_pipeline:
                    run_testing(cfg)

                    mock_pipeline.assert_called_once()

    def test_run_testing_with_metrics(self):
        """Test run_testing with metrics."""
        from maskgit3d.cli.test import run_testing

        cfg = OmegaConf.create(
            {
                "model": {"type": "maskgit"},
                "dataset": {"type": "simple"},
                "output": {"output_dir": "./outputs"},
                "checkpoint": {"load_from": "./checkpoint.ckpt"},
                "training": {"fabric": {}},
            }
        )

        with patch("maskgit3d.cli.test.create_module_from_config") as mock_create:
            mock_module = MagicMock()
            mock_create.return_value = mock_module

            with patch("maskgit3d.cli.test.Injector") as mock_injector_class:
                mock_injector = MagicMock()
                mock_injector_class.return_value = mock_injector
                mock_injector.get.return_value = MagicMock()

                with patch("maskgit3d.cli.test.FabricTestPipeline") as mock_pipeline:
                    run_testing(cfg)

                    mock_pipeline.assert_called_once()


class TestCliMainAdditional:
    """Additional tests for CLI main module."""

    def test_main_no_args(self):
        """Test main with no arguments."""
        from maskgit3d.cli.main import main

        with pytest.raises(SystemExit) as exc_info, patch.object(sys, "argv", ["maskgit3d"]):
            main()

        assert exc_info.value.code == 1

    def test_train_command_with_overrides(self):
        """Test train command with config overrides."""
        from maskgit3d.cli.main import main

        with (
            patch("maskgit3d.cli.train.main") as mock_train_main,
            patch.object(
                sys, "argv", ["maskgit3d", "train", "model=maskgit", "training.num_epochs=10"]
            ),
        ):
            main()

            mock_train_main.assert_called_once()

    def test_config_dir_env(self):
        """Test that --config-dir sets environment variable."""
        from maskgit3d.cli.main import main

        with (
            patch("maskgit3d.cli.train.main"),
            patch.dict("os.environ", {}, clear=True),
            patch.object(sys, "argv", ["maskgit3d", "--config-dir", "/tmp/config", "train"]),
        ):
            main()

            import os

            assert os.environ.get("HYDRA_CONFIG_PATH") == "/tmp/config"
