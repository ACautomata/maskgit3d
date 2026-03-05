"""Testing CLI for maskgit3d with Hydra configuration."""

import logging
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from injector import Injector
from omegaconf import DictConfig, OmegaConf

from maskgit3d.application.pipeline import FabricTestPipeline
from maskgit3d.cli.train import create_module_from_config

logger = logging.getLogger(__name__)


def run_testing(cfg: DictConfig) -> None:
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    module = create_module_from_config(cfg)
    injector = Injector([module])

    from maskgit3d.domain.interfaces import (
        DataProvider,
        InferenceStrategy,
        Metrics,
        ModelInterface,
    )

    model = injector.get(ModelInterface)
    data_provider = injector.get(DataProvider)
    inference_strategy = injector.get(InferenceStrategy)

    try:
        metrics = injector.get(Metrics)
    except Exception:
        metrics = None

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output directory: %s", output_dir)

    checkpoint_path = cfg.checkpoint.get("load_from")

    fabric_cfg = cfg.get("training", {}).get("fabric", {})
    accelerator = fabric_cfg.get("accelerator", "auto")
    devices = fabric_cfg.get("devices", "auto")
    strategy = fabric_cfg.get("strategy", "auto")
    precision = fabric_cfg.get("precision", "32-true")

    pipeline = FabricTestPipeline(
        model=model,
        data_provider=data_provider,
        inference_strategy=inference_strategy,
        metrics=metrics,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        checkpoint_path=checkpoint_path,
        output_dir=str(output_dir),
    )

    logger.info(
        "Starting testing: model=%s, dataset=%s",
        cfg.model.type,
        cfg.dataset.type,
    )

    save_predictions = cfg.output.get("save_predictions", False)
    export_nifti = cfg.output.get("export_nifti", False)
    enable_tensorboard = cfg.output.get("enable_tensorboard", False)
    tensorboard_dir = cfg.output.get("tensorboard_dir", None)

    pipeline.run(
        save_predictions=save_predictions,
        export_nifti=export_nifti,
        enable_tensorboard=enable_tensorboard,
        tensorboard_dir=tensorboard_dir,
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
