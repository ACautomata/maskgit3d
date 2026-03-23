"""Integration tests for callback configuration and trainer integration."""

from unittest.mock import MagicMock

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from maskgit3d import train as train_module
from maskgit3d.runtime.callback_selection import select_callback_config


class TestCallbackIntegration:
    """Test suite for callback integration with Hydra and Trainer."""

    def test_default_callbacks_instantiation(self):
        """Test that default callbacks can be instantiated from config."""
        from pathlib import Path

        config_dir = str(Path(__file__).parent.parent.parent / "src/maskgit3d/conf")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="train", overrides=["callbacks=default"])

            # Verify callbacks config exists
            assert cfg.callbacks is not None
            assert "best_checkpoint" in cfg.callbacks
            assert "early_stopping" in cfg.callbacks
            assert "lr_monitor" in cfg.callbacks
            assert "nan_detection" in cfg.callbacks
            assert "train_loss" in cfg.callbacks
            assert "masked_cross_entropy" in cfg.callbacks
            assert "best_checkpoint_maskgit" in cfg.callbacks
            assert "early_stopping_maskgit" in cfg.callbacks

    def test_maskgit_callbacks_instantiation(self):
        from pathlib import Path

        config_dir = str(Path(__file__).parent.parent.parent / "src/maskgit3d/conf")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=["task=maskgit", "model=maskgit", "callbacks=default"],
            )

            selected = select_callback_config(
                cfg.callbacks,
                type("MaskGITTask", (), {})(),
                stage="train",
            )

            assert selected is not None
            assert "masked_cross_entropy" in selected
            assert "mask_accuracy" in selected
            assert "train_loss" not in selected
            assert selected.best_checkpoint_maskgit.monitor == "val_loss"
            assert selected.early_stopping_maskgit.monitor == "val_loss"

    def test_callbacks_convert_to_list(self):
        """Test that callbacks dict converts to list for Trainer."""
        from pathlib import Path

        config_dir = str(Path(__file__).parent.parent.parent / "src/maskgit3d/conf")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="train", overrides=["callbacks=default"])

            selected = select_callback_config(
                cfg.callbacks,
                type("VQVAETask", (), {})(),
                stage="train",
            )
            callbacks_dict = instantiate(selected)
            callbacks_list = list(callbacks_dict.values())

            assert len(callbacks_list) == len(callbacks_dict)
            assert all(hasattr(cb, "on_train_batch_end") for cb in callbacks_list)

    def test_trainer_creation_with_callbacks(self):
        """Test that Trainer can be created with default callbacks."""
        from pathlib import Path

        config_dir = str(Path(__file__).parent.parent.parent / "src/maskgit3d/conf")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="train", overrides=["callbacks=default"])

            selected = select_callback_config(
                cfg.callbacks,
                type("VQVAETask", (), {})(),
                stage="train",
            )
            callbacks_dict = instantiate(selected)
            callbacks = list(callbacks_dict.values())

            # Should not raise
            trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=False)
            assert trainer is not None
            assert len(trainer.callbacks) >= len(callbacks)

    def test_vqvae_validation_metrics_are_logged_by_callback(self):
        from maskgit3d.callbacks.reconstruction_loss import ReconstructionLossCallback
        from maskgit3d.tasks.vqvae_task import VQVAETask

        task = VQVAETask(
            in_channels=1,
            out_channels=1,
            latent_channels=64,
            num_embeddings=100,
            embedding_dim=64,
            use_perceptual=False,
        )
        logged: dict[str, object] = {}

        def capture_log(name: str, value: object, **_: object) -> None:
            logged[name] = value

        task.log = capture_log  # type: ignore[method-assign]
        task.eval()
        callback = ReconstructionLossCallback()
        batch = (torch.randn(1, 1, 32, 32, 32), torch.zeros(1))

        outputs = task.validation_step(batch, 0)

        assert logged == {}
        callback.on_validation_batch_end(MagicMock(), task, outputs, batch, 0)
        assert "val_rec_loss" in logged

    def test_maskgit_validation_metrics_are_logged_by_callback(self):
        from maskgit3d.callbacks.mask_accuracy import MaskAccuracyCallback
        from maskgit3d.tasks.maskgit_task import MaskGITTask

        task = MaskGITTask(hidden_size=128, num_layers=2, num_heads=4, lr=1e-4)
        logged: dict[str, object] = {}

        def capture_log(name: str, value: object, **_: object) -> None:
            logged[name] = value

        task.log = capture_log  # type: ignore[method-assign]
        task.eval()
        callback = MaskAccuracyCallback()
        batch = torch.randn(1, 1, 16, 16, 16)

        with torch.no_grad():
            result = task.validation_step(batch, 0)

        assert isinstance(result, dict)
        assert "generated_images" in result
        assert logged == {}
        callback.on_validation_batch_end(MagicMock(), task, result, batch, 0)
        assert "val_mask_acc" in logged

    def test_checkpoint_monitor_metric_exists(self):
        """Test that checkpoint config monitors a metric that tasks log."""
        from pathlib import Path

        config_dir = str(Path(__file__).parent.parent.parent / "src/maskgit3d/conf")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="train", overrides=["callbacks=default"])

            assert cfg.callbacks.best_checkpoint.monitor == "val_rec_loss"
            assert cfg.callbacks.early_stopping.monitor == "val_rec_loss"
            assert cfg.callbacks.best_checkpoint_maskgit.monitor == "val_loss"
            assert cfg.callbacks.early_stopping_maskgit.monitor == "val_loss"

    def test_eval_default_callbacks_include_metrics(self):
        from pathlib import Path

        config_dir = str(Path(__file__).parent.parent.parent / "src/maskgit3d/conf")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="eval")

            assert cfg.callbacks is not None
            assert "sample_saving" in cfg.callbacks
            assert "reconstruction_loss" in cfg.callbacks
            assert "masked_cross_entropy" in cfg.callbacks

    def test_train_medmnist_uses_fid_for_vqvae_selection(self):
        from pathlib import Path

        config_dir = str(Path(__file__).parent.parent.parent / "src/maskgit3d/conf")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="train_medmnist")

            assert cfg.callbacks.best_checkpoint.monitor == "val_fid"
            assert cfg.callbacks.early_stopping.monitor == "val_fid"

    def test_train_main_with_default_callbacks(self, monkeypatch):
        """Test that train.main converts callbacks dict to list for Trainer."""
        cfg = OmegaConf.create(
            {
                "data": {"_target_": "tests.DummyDataModule"},
                "task": {"_target_": "tests.DummyTask"},
                "trainer": {"_target_": "tests.DummyTrainer"},
                "callbacks": {
                    "cb1": {"_target_": "tests.DummyCallback"},
                    "cb2": {"_target_": "tests.DummyCallback"},
                },
                "logger": None,
                "checkpoint_path": None,
            }
        )

        calls = []

        class DummyTrainer:
            def __init__(self, **kwargs):
                calls.append(("Trainer.__init__", kwargs))
                self.callbacks = kwargs.get("callbacks", [])

            def fit(self, task, datamodule, ckpt_path=None):
                calls.append(
                    (
                        "Trainer.fit",
                        {"task": task, "datamodule": datamodule, "ckpt_path": ckpt_path},
                    )
                )

        class DummyDataModule:
            pass

        class DummyTask:
            pass

        class DummyCallback:
            pass

        def fake_instantiate(config, **kwargs):
            target = config.get("_target_") if hasattr(config, "get") else None

            if target == "tests.DummyDataModule":
                return DummyDataModule()
            if target == "tests.DummyTask":
                return DummyTask()
            if target == "tests.DummyCallback":
                return DummyCallback()
            if target == "tests.DummyTrainer":
                assert "callbacks" in kwargs
                callbacks = kwargs["callbacks"]
                assert isinstance(callbacks, list), f"Expected list, got {type(callbacks)}"
                assert len(callbacks) == 2
                return DummyTrainer(**kwargs)
            if target is None and hasattr(config, "keys") and "cb1" in config:
                return config
            raise AssertionError(f"Unexpected target: {target}, config: {config}")

        def fake_build_training_task(cfg):
            return DummyTask()

        monkeypatch.setattr(train_module, "instantiate", fake_instantiate)
        monkeypatch.setattr(train_module, "_build_training_task", fake_build_training_task)
        monkeypatch.setattr(train_module, "to_absolute_path", lambda path: path)

        train_module.main.__wrapped__(cfg)

        init_call = [c for c in calls if c[0] == "Trainer.__init__"][0]
        assert isinstance(init_call[1]["callbacks"], list)
