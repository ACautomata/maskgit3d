from __future__ import annotations

from hydra.utils import get_class, instantiate
from lightning import LightningModule
from omegaconf import DictConfig

from .checkpoints import load_vqvae_from_checkpoint
from .model_factory import create_maskgit_model, create_vqvae_model
from ..tasks.maskgit_task import MaskGITTask
from ..tasks.vqvae_task import VQVAETask


def _get_task_target(cfg: DictConfig) -> str:
    task_target = cfg.task.get("_target_")
    if task_target is None:
        raise ValueError("cfg.task._target_ must be set.")
    return str(task_target)


def build_training_task(cfg: DictConfig) -> VQVAETask | MaskGITTask:
    _get_task_target(cfg)

    task_kwargs: dict[str, object] = {
        "_recursive_": False,
    }
    if cfg.get("model") is not None:
        task_kwargs["model_config"] = cfg.model
    if cfg.get("optimizer") is not None:
        task_kwargs["optimizer_config"] = cfg.optimizer
        task_kwargs["disc_optimizer_config"] = cfg.optimizer
    if cfg.get("data") is not None:
        task_kwargs["data_config"] = cfg.data

    task = instantiate(cfg.task, **task_kwargs)
    if not isinstance(task, VQVAETask | MaskGITTask):
        raise TypeError("build_training_task only supports VQVAETask and MaskGITTask.")

    return task


def build_eval_task(cfg: DictConfig, ckpt_path: str) -> LightningModule:
    task_target = _get_task_target(cfg)
    task_class: type[LightningModule] = get_class(task_target)
    return task_class.load_from_checkpoint(
        ckpt_path,
        weights_only=False,
    )
