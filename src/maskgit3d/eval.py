
import hydra
from hydra.utils import get_class, instantiate, to_absolute_path
from lightning.pytorch import LightningModule
from omegaconf import DictConfig


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
    task: LightningModule = task_class.load_from_checkpoint(ckpt_path)

    datamodule = instantiate(cfg.data)
    trainer = instantiate(cfg.trainer)
    if cfg.get("mode", "validate") == "test":
        trainer.test(task, datamodule=datamodule)
    else:
        trainer.validate(task, datamodule=datamodule)


if __name__ == "__main__":
    main()
