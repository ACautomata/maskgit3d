"""Training CLI for maskgit3d with Hydra configuration."""

import logging

import hydra
from injector import Injector
from omegaconf import DictConfig, OmegaConf

from maskgit3d.application.pipeline import FabricTrainingPipeline, TrainingPipeline

logger = logging.getLogger(__name__)


def _extract_factory_params(cfg: DictConfig) -> dict:
    model = cfg.model
    dataset = cfg.dataset
    training = cfg.training
    return {
        "image_size": model.get("image_size", 64),
        "in_channels": model.get("in_channels", 1),
        "embed_dim": model.get("embed_dim", 256),
        "latent_channels": model.get("latent_channels", 256),
        "lr": training.optimizer.lr,
        "batch_size": dataset.batch_size,
    }


def _create_data_config(cfg: DictConfig) -> dict:
    dataset = cfg.dataset
    model = cfg.model
    return {
        "type": dataset.get("type", "simple"),
        "params": {
            "num_train": dataset.get("num_train", 100),
            "num_val": dataset.get("num_val", 20),
            "num_test": dataset.get("num_test", 20),
            "batch_size": dataset.get("batch_size", 4),
            "in_channels": model.get("in_channels", 1),
            "out_channels": model.get("in_channels", 1),
            "spatial_size": (
                model.get("image_size", 64),
                model.get("image_size", 64),
                model.get("image_size", 64),
            ),
            "num_workers": dataset.get("num_workers", 0),
        },
    }


def create_module_from_config(cfg: DictConfig):
    from maskgit3d.config.modules import (
        DataModule,
        InferenceModule,
        MaskGITModelModule,
        ModelModule,
        TrainingModule,
    )
    from maskgit3d.domain.interfaces import MaskGITModelInterface, ModelInterface

    model_type = cfg.model.type
    base_params = _extract_factory_params(cfg)
    data_config = _create_data_config(cfg)
    training_config = _create_training_config(cfg, model_type)
    optimizer_config = _create_optimizer_config(cfg)
    inference_config = _create_inference_config(cfg, model_type)

    class CompositeModule:
        def __init__(self):
            self.data_module = DataModule(data_config)
            self.training_module = TrainingModule(training_config, optimizer_config)
            self.inference_module = InferenceModule(inference_config)
            if model_type == "maskgit":
                model_params = _create_model_params(cfg, model_type, base_params)
                pretrained_path = cfg.model.get("pretrained_vqgan_path", None)
                freeze_vqgan = cfg.model.get("freeze_vqgan", True)
                self.model_module = MaskGITModelModule(
                    {"type": model_type, "params": model_params},
                    pretrained_vqgan_path=pretrained_path,
                    freeze_vqgan=freeze_vqgan,
                )
                self.maskgit_model = self.model_module.provide_maskgit_model()
            else:
                model_config = {
                    "type": model_type,
                    "params": _create_model_params(cfg, model_type, base_params),
                }
                self.model_module = ModelModule(model_config)

        def configure(self, binder):
            if model_type == "maskgit":
                binder.bind(MaskGITModelInterface, to=lambda: self.maskgit_model)
                binder.bind(ModelInterface, to=lambda: self.maskgit_model)
            else:
                binder.install(self.model_module)
            binder.install(self.data_module)
            binder.install(self.training_module)
            binder.install(self.inference_module)

        def __call__(self, binder):
            self.configure(binder)

    return CompositeModule()


def _create_model_params(cfg: DictConfig, model_type: str, base_params: dict) -> dict:
    if model_type == "maskgit":
        return {
            "in_channels": base_params["in_channels"],
            "codebook_size": cfg.model.get("codebook_size", 1024),
            "embed_dim": base_params["embed_dim"],
            "latent_channels": base_params["latent_channels"],
            "resolution": base_params["image_size"],
            "channel_multipliers": tuple(cfg.model.get("channel_multipliers", [1, 1, 2, 2, 4])),
            "transformer_hidden": cfg.model.get("transformer_hidden", 768),
            "transformer_layers": cfg.model.get("transformer_layers", 12),
            "transformer_heads": cfg.model.get("transformer_heads", 12),
            "mask_ratio": cfg.model.get("mask_ratio", 0.5),
        }
    elif model_type in ("vqgan", "vqgan3d"):
        return {
            "in_channels": base_params["in_channels"],
            "codebook_size": cfg.model.get("codebook_size", 1024),
            "embed_dim": base_params["embed_dim"],
            "latent_channels": base_params["latent_channels"],
            "resolution": base_params["image_size"],
            "channel_multipliers": tuple(cfg.model.get("channel_multipliers", [1, 2])),
            "num_res_blocks": cfg.model.get("num_res_blocks", 2),
            "attn_resolutions": tuple(cfg.model.get("attn_resolutions", [])),
            "dropout": cfg.model.get("dropout", 0.0),
        }
    elif model_type == "maisi_vq":
        from maskgit3d.infrastructure.vqgan import get_maisi_vq_config

        return get_maisi_vq_config(
            image_size=base_params["image_size"],
            in_channels=base_params["in_channels"],
            codebook_size=cfg.model.get("codebook_size", 1024),
            embed_dim=base_params["embed_dim"],
            latent_channels=cfg.model.get("latent_channels", 4),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _create_training_config(cfg: DictConfig, model_type: str) -> dict:
    strategy_type = "maskgit" if model_type == "maskgit" else "vqgan"
    return {
        "type": strategy_type,
        "params": {
            "codebook_weight": 1.0,
            "pixel_loss_weight": 1.0,
        }
        if strategy_type == "vqgan"
        else {
            "mask_ratio": cfg.model.get("mask_ratio", 0.5),
            "reconstruction_weight": 1.0,
        },
    }


def _create_optimizer_config(cfg: DictConfig) -> dict:
    return {
        "type": cfg.training.optimizer.get("type", "adam"),
        "params": {"lr": cfg.training.optimizer.get("lr", 1e-4)},
    }


def _create_inference_config(cfg: DictConfig, model_type: str) -> dict:
    inference_type = "maskgit" if model_type == "maskgit" else "vqgan"
    return {
        "type": inference_type,
        "params": {"mode": "reconstruct"},
    }


@hydra.main(config_path="pkg://maskgit3d.conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    module = create_module_from_config(cfg)
    injector = Injector([module])
    use_fabric = cfg.training.fabric.get("enabled", False)
    pipeline_class = FabricTrainingPipeline if use_fabric else TrainingPipeline
    from maskgit3d.domain.interfaces import (
        DataProvider,
        ModelInterface,
        OptimizerFactory,
        TrainingStrategy,
    )

    model = injector.get(ModelInterface)
    data_provider = injector.get(DataProvider)
    training_strategy = injector.get(TrainingStrategy)
    optimizer_factory = injector.get(OptimizerFactory)
    pipeline = pipeline_class(
        model=model,
        data_provider=data_provider,
        training_strategy=training_strategy,
        optimizer_factory=optimizer_factory,
    )
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
