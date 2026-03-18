from importlib import import_module
from unittest.mock import Mock

from omegaconf import OmegaConf

from src.maskgit3d.tasks.vqvae_task import VQVAETask


def test_build_training_task_returns_vqvae_task(monkeypatch) -> None:
    composition_module = import_module("src.maskgit3d.runtime.composition")
    cfg = OmegaConf.create(
        {
            "task": {
                "_target_": "maskgit3d.tasks.vqvae_task.VQVAETask",
                "model_config": {"_target_": "tests.ModelConfig"},
                "optimizer_config": {"_target_": "tests.OptimizerConfig"},
                "disc_optimizer_config": {"_target_": "tests.DiscOptimizerConfig"},
                "data_config": {"crop_size": [32, 32, 32]},
                "lr_g": 1e-4,
                "lr_d": 1e-4,
            },
            "model": {"_target_": "tests.ModelConfig"},
            "optimizer": {"_target_": "tests.OptimizerConfig"},
            "data": {"crop_size": [32, 32, 32]},
        }
    )

    fake_task = Mock(spec=VQVAETask)

    def fake_instantiate(config, **kwargs):
        target = config.get("_target_")
        if target == "tests.ModelConfig":
            return config
        if target == "maskgit3d.tasks.vqvae_task.VQVAETask":
            assert kwargs["model_config"].get("_target_") == "tests.ModelConfig"
            assert kwargs["optimizer_config"].get("_target_") == "tests.OptimizerConfig"
            assert kwargs["disc_optimizer_config"].get("_target_") == "tests.OptimizerConfig"
            assert list(kwargs["data_config"].crop_size) == [32, 32, 32]
            assert kwargs["_recursive_"] is False
            return fake_task
        return config

    monkeypatch.setattr(composition_module, "instantiate", fake_instantiate)

    assert composition_module.build_training_task(cfg) is fake_task
