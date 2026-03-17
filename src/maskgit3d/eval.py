from typing import Any

import hydra
from hydra.utils import get_class, instantiate, to_absolute_path
from lightning.pytorch import LightningModule
from omegaconf import DictConfig, OmegaConf


def _resolve_required_ckpt_path(path: str | None) -> str:
    if path is None:
        raise ValueError("cfg.ckpt_path must be set for evaluation.")
    return to_absolute_path(path)


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg: DictConfig) -> None:
    ckpt_path = _resolve_required_ckpt_path(cfg.get("ckpt_path"))
    task_target = cfg.task.get("_target_")
    if task_target is None:
        raise ValueError("cfg.task._target_ must be set for evaluation.")

    task_class: type[LightningModule] = get_class(task_target)
    task: LightningModule = task_class.load_from_checkpoint(
        ckpt_path,
        weights_only=False,
    )

    datamodule = instantiate(cfg.data)

    callbacks: Any = instantiate(cfg.callbacks) if cfg.get("callbacks") is not None else None
    logger: Any = instantiate(cfg.logger) if cfg.get("logger") is not None else None

    trainer_kwargs: dict[str, Any] = {}
    if callbacks is not None:
        # Convert dict/DictConfig values to list for Lightning Trainer
        if isinstance(callbacks, dict) or OmegaConf.is_dict(callbacks):
            trainer_kwargs["callbacks"] = list(callbacks.values())
        else:
            trainer_kwargs["callbacks"] = callbacks
    if logger is not None:
        trainer_kwargs["logger"] = logger

    trainer = instantiate(cfg.trainer, **trainer_kwargs)
    if cfg.get("mode", "validate") == "test":
        trainer.test(task, datamodule=datamodule)
    else:
        trainer.validate(task, datamodule=datamodule)


if __name__ == "__main__":
    main()
