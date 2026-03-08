from typing import Any

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig


def _resolve_optional_path(path: str | None) -> str | None:
    if path is None:
        return None
    return to_absolute_path(path)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.data)
    task = instantiate(cfg.task)

    callbacks: Any = instantiate(cfg.callbacks) if cfg.get("callbacks") is not None else None
    logger: Any = instantiate(cfg.logger) if cfg.get("logger") is not None else None

    trainer_kwargs: dict[str, Any] = {}
    if callbacks is not None:
        trainer_kwargs["callbacks"] = callbacks
    if logger is not None:
        trainer_kwargs["logger"] = logger

    trainer = instantiate(cfg.trainer, **trainer_kwargs)
    trainer.fit(task, datamodule=datamodule, ckpt_path=_resolve_optional_path(cfg.get("ckpt_path")))


if __name__ == "__main__":
    main()
