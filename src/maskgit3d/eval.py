import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.data)
    task = instantiate(cfg.task)
    task = task.load_from_checkpoint(cfg.ckpt_path)

    trainer = instantiate(cfg.trainer)
    if cfg.get("mode", "validate") == "test":
        trainer.test(task, datamodule=datamodule)
    else:
        trainer.validate(task, datamodule=datamodule)


if __name__ == "__main__":
    main()
