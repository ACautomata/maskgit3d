from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
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
    trainer.fit(task, datamodule=datamodule)


if __name__ == "__main__":
    main()
