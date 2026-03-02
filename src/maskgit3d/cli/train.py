"""Training CLI for maskgit3d with Hydra configuration."""

import logging
import os

# Disable Hydra shell completion to avoid Python 3.14 compatibility issues
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
from injector import Injector
from omegaconf import DictConfig, OmegaConf

from maskgit3d.application.pipeline import FabricTrainingPipeline, TrainingPipeline
from maskgit3d.config.modules import (
    create_maskgit_module,
    create_vqgan_module,
)

logger = logging.getLogger(__name__)


def _extract_factory_params(cfg: DictConfig) -> dict:
    """Extract parameters for factory functions from Hydra config.

    Maps Hydra config structure to the keyword arguments expected by
    create_maskgit_module / create_vqgan_module / create_maisi_vq_module.
    Only passes parameters that the factory functions actually accept.
    """
    model = cfg.model
    dataset = cfg.dataset
    training = cfg.training

    # Common parameters shared by all factory functions
    params = {
        "image_size": model.get("image_size", 64),
        "in_channels": model.get("in_channels", 1),
        "embed_dim": model.get("embed_dim", 256),
        "latent_channels": model.get("latent_channels", 256),
        "lr": training.optimizer.lr,
        "batch_size": dataset.batch_size,
    }

    return params


def create_module_from_config(cfg: DictConfig):
    """Create DI module from Hydra configuration.

    Routes to the appropriate factory function based on model type,
    passing only the parameters each factory accepts.
    """
    model_type = cfg.model.type

    base_params = _extract_factory_params(cfg)

    if model_type == "maskgit":
        maskgit_params = {
            **base_params,
            "codebook_size": cfg.model.get("codebook_size", 1024),
            "transformer_hidden": cfg.model.get("transformer_hidden", 768),
            "transformer_layers": cfg.model.get("transformer_layers", 12),
            "transformer_heads": cfg.model.get("transformer_heads", 12),
            "mask_ratio": cfg.model.get("mask_ratio", 0.5),
            "num_train": cfg.dataset.get("num_train", 1000),
            "num_val": cfg.dataset.get("num_val", 100),
        }
        return create_maskgit_module(**maskgit_params)

    elif model_type in ("vqgan", "vqgan3d"):
        vqgan_params = {
            **base_params,
            "n_embed": cfg.model.get("codebook_size", 1024),
            "num_train": cfg.dataset.get("num_train", 1000),
            "num_val": cfg.dataset.get("num_val", 100),
        }
        return create_vqgan_module(**vqgan_params)

    elif model_type == "maisi_vq":
        from maskgit3d.config.modules import create_maisi_vq_module

        maisi_params = {
            **base_params,
            "codebook_size": cfg.model.get("codebook_size", 1024),
        }
        return create_maisi_vq_module(**maisi_params)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


@hydra.main(config_path="pkg://maskgit3d.conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run training with Hydra configuration."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Create DI module from config
    module = create_module_from_config(cfg)

    # Create injector
    injector = Injector([module])

    # Determine pipeline type based on Fabric config
    use_fabric = cfg.training.fabric.get("enabled", False)
    pipeline_class = FabricTrainingPipeline if use_fabric else TrainingPipeline

    pipeline = injector.get(pipeline_class)

    # Run training
    num_epochs = cfg.training.num_epochs
    logger.info(
        "Starting training: model=%s, dataset=%s, epochs=%d",
        cfg.model.type,
        cfg.dataset.type,
        num_epochs,
    )

    pipeline.run(num_epochs=num_epochs)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
