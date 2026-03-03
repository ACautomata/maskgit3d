"""Testing CLI for maskgit3d with Hydra configuration."""

import logging
import os
import sys

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from injector import Injector
from omegaconf import DictConfig, OmegaConf

from maskgit3d.application.pipeline import TestPipeline
from maskgit3d.cli.train import create_module_from_config

logger = logging.getLogger(__name__)


def run_testing(cfg: DictConfig) -> None:
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    module = create_module_from_config(cfg)
    injector = Injector([module])

    from maskgit3d.domain.interfaces import (
        ModelInterface,
        DataProvider,
        InferenceStrategy,
        Metrics,
    )

    model = injector.get(ModelInterface)
    data_provider = injector.get(DataProvider)
    inference_strategy = injector.get(InferenceStrategy)

    try:
        metrics = injector.get(Metrics)
    except Exception:
        metrics = None

    output_dir = cfg.output.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    pipeline = TestPipeline(
        model=model,
        data_provider=data_provider,
        inference_strategy=inference_strategy,
        metrics=metrics,
        output_dir=output_dir,
    )

    logger.info(
        "Starting testing: model=%s, dataset=%s",
        cfg.model.type,
        cfg.dataset.type,
    )

    checkpoint_path = cfg.checkpoint.get("load_from")
    save_predictions = cfg.output.get("save_predictions", False)

    results = pipeline.run(
        checkpoint_path=checkpoint_path,
        save_predictions=save_predictions,
    )

    logger.info("Testing completed! Results saved to %s", output_dir)


def main():
    overrides = []
    for arg in sys.argv[1:]:
        if "=" in arg:
            overrides.append(arg)

    GlobalHydra.instance().clear()

    from pathlib import Path
    import maskgit3d

    package_dir = Path(maskgit3d.__file__).parent
    conf_dir = str(package_dir / "conf")

    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
        run_testing(cfg)


if __name__ == "__main__":
    main()
