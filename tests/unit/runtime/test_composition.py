from importlib import import_module
from unittest.mock import Mock

from omegaconf import OmegaConf

from src.maskgit3d.tasks.incontext_task import InContextMaskGITTask
from src.maskgit3d.tasks.maskgit_task import MaskGITTask
from src.maskgit3d.tasks.vqvae_task import VQVAETask


def test_build_training_task_returns_task_via_instantiate(monkeypatch) -> None:
    """Test that build_training_task dispatches to build_vqvae_task for VQVAE tasks."""
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
            "model": {
                "_target_": "tests.ModelConfig",
                "num_channels": [64, 128, 256, 256],
                "num_res_blocks": [2, 2, 2, 1],
                "attention_levels": [False, False, False, True],
            },
            "optimizer": {"_target_": "tests.OptimizerConfig"},
            "data": {"crop_size": [32, 32, 32]},
        }
    )

    fake_task = Mock(spec=VQVAETask)
    fake_task.hparams = {}
    fake_vqvae = Mock()
    fake_vqvae.encoder = Mock()
    fake_vqvae.quant_conv = Mock()
    fake_vqvae.post_quant_conv = Mock()
    fake_vqvae.decoder = Mock()
    fake_vqvae.quantizer = Mock()
    fake_vqvae.quantizer_type = "vq"

    def fake_instantiate(config, **kwargs):
        target = config.get("_target_")
        if target == "maskgit3d.tasks.vqvae_task.VQVAETask":
            assert kwargs["model_config"] is None
            assert kwargs["optimizer_config"].get("_target_") == "tests.OptimizerConfig"
            assert kwargs["disc_optimizer_config"].get("_target_") == "tests.OptimizerConfig"
            assert list(kwargs["data_config"].crop_size) == [32, 32, 32]
            assert kwargs["_recursive_"] is False
            return fake_task
        return config

    monkeypatch.setattr(composition_module, "instantiate", fake_instantiate)
    monkeypatch.setattr(composition_module, "create_vqvae_model", lambda cfg: fake_vqvae)
    monkeypatch.setattr(composition_module, "VQPerceptualLoss", Mock)
    monkeypatch.setattr(composition_module, "GANTrainingStrategy", Mock)
    monkeypatch.setattr(composition_module, "VQVAEReconstructor", Mock)
    monkeypatch.setattr(composition_module, "VQVAETrainingSteps", Mock)

    task = composition_module.build_training_task(cfg)

    assert task is fake_task


def test_build_training_task_returns_maskgit_task_via_instantiate(monkeypatch) -> None:
    """Test that build_training_task dispatches to build_maskgit_task for MaskGIT tasks."""
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
    fake_task.hparams = {}
    fake_task.vqvae = Mock()

    def fake_instantiate(config, **kwargs):
        target = config.get("_target_")
        if target == "maskgit3d.tasks.maskgit_task.MaskGITTask":
            # model_config is set to None by build_maskgit_task before instantiation
            assert kwargs["model_config"] is None
            assert kwargs["optimizer_config"].get("_target_") == "tests.OptimizerConfig"
            # MaskGIT tasks should NOT have disc_optimizer_config or data_config
            assert "disc_optimizer_config" not in kwargs
            assert "data_config" not in kwargs
            assert kwargs["_recursive_"] is False
            return fake_task
        return config

    monkeypatch.setattr(composition_module, "instantiate", fake_instantiate)
    # Mock checkpoint loading to avoid file not found error
    fake_vqvae = Mock()
    monkeypatch.setattr(
        composition_module, "load_vqvae_from_checkpoint", lambda ckpt_path: fake_vqvae
    )
    monkeypatch.setattr(composition_module, "create_maskgit_model", lambda cfg, vqvae: Mock())
    monkeypatch.setattr(composition_module, "MaskGITTrainingSteps", Mock)

    task = composition_module.build_training_task(cfg)

    assert task is fake_task


def test_build_incontext_task_via_instantiate(monkeypatch) -> None:
    """Test that build_incontext_task correctly builds InContextMaskGITTask."""
    composition_module = import_module("src.maskgit3d.runtime.composition")
    cfg = OmegaConf.create(
        {
            "task": {
                "_target_": "maskgit3d.tasks.incontext_task.InContextMaskGITTask",
                "model_config": {"_target_": "tests.InContextModelConfig"},
                "optimizer_config": {"_target_": "tests.OptimizerConfig"},
                "vqvae_ckpt_path": "/tmp/vqvae.ckpt",
                "num_modalities": 4,
                "hidden_size": 768,
                "num_layers": 12,
                "num_heads": 12,
                "mlp_ratio": 4.0,
                "dropout": 0.1,
                "gamma_type": "cosine",
                "lr": 2e-4,
                "weight_decay": 0.05,
                "warmup_steps": 1000,
            },
            "optimizer": {"_target_": "tests.OptimizerConfig"},
        }
    )

    fake_task = Mock(spec=InContextMaskGITTask)
    fake_task.hparams = {}
    fake_task.vqvae = Mock()

    def fake_instantiate(config, **kwargs):
        target = config.get("_target_")
        if target == "maskgit3d.tasks.incontext_task.InContextMaskGITTask":
            assert kwargs["model_config"] is None
            assert kwargs["optimizer_config"].get("_target_") == "tests.OptimizerConfig"
            assert "disc_optimizer_config" not in kwargs
            assert "data_config" not in kwargs
            assert kwargs["_recursive_"] is False
            assert kwargs["vqvae_ckpt_path"] == "/tmp/vqvae.ckpt"
            return fake_task
        return config

    monkeypatch.setattr(composition_module, "instantiate", fake_instantiate)
    fake_vqvae = Mock()
    fake_vqvae.encoder = Mock()
    fake_vqvae.encoder.encoder = Mock()
    fake_vqvae.encoder.encoder.num_channels = [64, 128, 256]
    fake_vqvae.quantizer = Mock()
    fake_vqvae.quantizer.num_embeddings = 8192
    monkeypatch.setattr(
        composition_module, "load_vqvae_from_checkpoint", lambda ckpt_path: fake_vqvae
    )
    monkeypatch.setattr(composition_module, "InContextMaskGIT", Mock())
    monkeypatch.setattr(composition_module, "InContextTrainingSteps", Mock)

    task = composition_module.build_incontext_task(cfg)

    assert task is fake_task
    assert task.vqvae is fake_vqvae
    assert task.incontext_model is not None
    assert task.training_steps is not None
    assert task.optimizer_factory is not None
