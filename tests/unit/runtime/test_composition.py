from importlib import import_module
from unittest.mock import Mock

from omegaconf import OmegaConf

from src.maskgit3d.tasks.maskgit_task import MaskGITTask
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
    fake_vqvae = Mock()

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
    monkeypatch.setattr(composition_module, "create_vqvae_model", Mock(return_value=fake_vqvae))

    task = composition_module.build_training_task(cfg)

    assert task is fake_task
    composition_module.create_vqvae_model.assert_called_once_with(cfg.model)
    assert fake_task.vqvae is fake_vqvae


def test_build_training_task_replaces_maskgit_models_with_runtime_factories(monkeypatch) -> None:
    composition_module = import_module("src.maskgit3d.runtime.composition")
    cfg = OmegaConf.create(
        {
            "task": {
                "_target_": "maskgit3d.tasks.maskgit_task.MaskGITTask",
                "model_config": {"_target_": "tests.MaskGITModelConfig"},
                "optimizer_config": {"_target_": "tests.OptimizerConfig"},
                "vqvae_ckpt_path": "/tmp/vqvae.ckpt",
            },
            "model": {"_target_": "tests.MaskGITModelConfig"},
            "optimizer": {"_target_": "tests.OptimizerConfig"},
        }
    )

    fake_task = Mock(spec=MaskGITTask)
    fake_vqvae = object()
    fake_maskgit = object()

    def fake_instantiate(config, **kwargs):
        target = config.get("_target_")
        if target == "maskgit3d.tasks.maskgit_task.MaskGITTask":
            assert kwargs["model_config"].get("_target_") == "tests.MaskGITModelConfig"
            assert kwargs["optimizer_config"].get("_target_") == "tests.OptimizerConfig"
            assert kwargs["_recursive_"] is False
            return fake_task
        return config

    loader_mock = Mock(return_value=fake_vqvae)
    maskgit_factory_mock = Mock(return_value=fake_maskgit)

    monkeypatch.setattr(composition_module, "instantiate", fake_instantiate)
    monkeypatch.setattr(composition_module, "load_vqvae_from_checkpoint", loader_mock)
    monkeypatch.setattr(composition_module, "create_maskgit_model", maskgit_factory_mock)

    task = composition_module.build_training_task(cfg)

    assert task is fake_task
    loader_mock.assert_called_once_with("/tmp/vqvae.ckpt")
    maskgit_factory_mock.assert_called_once_with(cfg.model, fake_vqvae)
    assert fake_task.vqvae is fake_vqvae
    assert fake_task.maskgit is fake_maskgit
