"""Testing CLI for maskgit3d with Hydra configuration."""

import logging
import os

import hydra
from injector import Injector
from omegaconf import DictConfig, OmegaConf

from maskgit3d.application.pipeline import TestPipeline
from maskgit3d.cli.train import create_module_from_config

logger = logging.getLogger(__name__)


@hydra.main(config_path="pkg://maskgit3d.conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run testing/inference with Hydra configuration."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Create DI module from config
    module = create_module_from_config(cfg)

    # Create injector
    injector = Injector([module])

    # Get test pipeline
    pipeline = injector.get(TestPipeline)

    # Setup output directory from config
    output_dir = cfg.output.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Run testing
    logger.info(
        "Starting testing: model=%s, dataset=%s",
        cfg.model.type,
        cfg.dataset.type,
    )

    pipeline.run(output_dir=output_dir)

    logger.info("Testing completed! Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
