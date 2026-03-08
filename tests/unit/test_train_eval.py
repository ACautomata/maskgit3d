from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from src.maskgit3d import eval as eval_module
from src.maskgit3d import train as train_module


def test_train_main_passes_resolved_checkpoint_path(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = OmegaConf.create(
        {
            "data": {"_target_": "tests.DataModule"},
            "task": {"_target_": "tests.Task"},
            "trainer": {"_target_": "tests.Trainer"},
            "callbacks": {"_target_": "tests.Callbacks"},
            "logger": {"_target_": "tests.Logger"},
            "checkpoint_path": "checkpoints/latest.ckpt",
        }
    )

    datamodule = Mock(name="datamodule")
    task = Mock(name="task")
    callbacks = Mock(name="callbacks")
    logger = Mock(name="logger")
    trainer = Mock(name="trainer")

    def fake_instantiate(config, **kwargs):
        target = config.get("_target_")
        if target == "tests.DataModule":
            return datamodule
        if target == "tests.Task":
            return task
        if target == "tests.Callbacks":
            return callbacks
        if target == "tests.Logger":
            return logger
        if target == "tests.Trainer":
            assert kwargs == {"callbacks": callbacks, "logger": logger}
            return trainer
        raise AssertionError(f"Unexpected target: {target}")

    monkeypatch.setattr(train_module, "instantiate", fake_instantiate)
    monkeypatch.setattr(train_module, "to_absolute_path", lambda path: f"/abs/{path}")

    train_module.main.__wrapped__(cfg)

    trainer.fit.assert_called_once_with(
        task,
        datamodule=datamodule,
        ckpt_path="/abs/checkpoints/latest.ckpt",
    )


def test_eval_main_requires_ckpt_path() -> None:
    cfg = OmegaConf.create(
        {
            "data": {"_target_": "tests.DataModule"},
            "task": {"_target_": "tests.Task"},
            "trainer": {"_target_": "tests.Trainer"},
            "ckpt_path": None,
            "mode": "validate",
        }
    )

    with pytest.raises(ValueError, match=r"cfg\.ckpt_path must be set"):
        eval_module.main.__wrapped__(cfg)


def test_eval_main_requires_task_target() -> None:
    cfg = OmegaConf.create(
        {
            "data": {"_target_": "tests.DataModule"},
            "task": {},
            "trainer": {"_target_": "tests.Trainer"},
            "ckpt_path": "checkpoints/best.ckpt",
            "mode": "validate",
        }
    )

    with pytest.raises(ValueError, match=r"cfg\.task\._target_ must be set"):
        eval_module.main.__wrapped__(cfg)


@pytest.mark.parametrize(
    ("mode", "trainer_method"),
    [("validate", "validate"), ("test", "test")],
)
def test_eval_main_loads_checkpoint_and_runs_requested_stage(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    trainer_method: str,
) -> None:
    cfg = OmegaConf.create(
        {
            "data": {"_target_": "tests.DataModule"},
            "task": {"_target_": "tests.Task"},
            "trainer": {"_target_": "tests.Trainer"},
            "ckpt_path": "checkpoints/best.ckpt",
            "mode": mode,
        }
    )

    datamodule = Mock(name="datamodule")
    trainer = Mock(name="trainer")
    loaded_task = Mock(name="loaded_task")

    class FakeTask:
        @classmethod
        def load_from_checkpoint(cls, path: str):
            assert path == "/abs/checkpoints/best.ckpt"
            return loaded_task

    def fake_instantiate(config, **kwargs):
        target = config.get("_target_")
        if target == "tests.DataModule":
            return datamodule
        if target == "tests.Trainer":
            assert kwargs == {}
            return trainer
        raise AssertionError(f"Unexpected target: {target}")

    monkeypatch.setattr(eval_module, "instantiate", fake_instantiate)
    monkeypatch.setattr(eval_module, "get_class", lambda target: FakeTask)
    monkeypatch.setattr(eval_module, "to_absolute_path", lambda path: f"/abs/{path}")

    eval_module.main.__wrapped__(cfg)

    getattr(trainer, trainer_method).assert_called_once_with(loaded_task, datamodule=datamodule)
