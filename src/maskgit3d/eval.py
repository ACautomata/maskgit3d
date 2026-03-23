from importlib import import_module
from typing import Any

import hydra
from hydra.utils import instantiate, to_absolute_path
from lightning import seed_everything
from lightning.pytorch import LightningModule
from omegaconf import DictConfig, OmegaConf

from maskgit3d.runtime.callback_selection import select_callback_config


def _resolve_required_ckpt_path(path: str | None) -> str:
    if path is None:
        raise ValueError("cfg.ckpt_path must be set for evaluation.")
    return to_absolute_path(path)


def _build_eval_task(cfg: DictConfig, ckpt_path: str) -> LightningModule:
    task: LightningModule = import_module("maskgit3d.runtime.composition").build_eval_task(
        cfg, ckpt_path
    )
    return task


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg: DictConfig) -> None:
    seed = cfg.get("seed")
    if seed is not None:
        seed_everything(int(seed), workers=True)

    ckpt_path = _resolve_required_ckpt_path(cfg.get("ckpt_path"))
    task: LightningModule = _build_eval_task(cfg, ckpt_path)

    datamodule = instantiate(cfg.data)

    callback_cfg = select_callback_config(cfg.get("callbacks"), task, stage="eval")
    callbacks: Any = instantiate(callback_cfg) if callback_cfg is not None else None
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
