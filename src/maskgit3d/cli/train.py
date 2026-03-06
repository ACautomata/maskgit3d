"""Training CLI for maskgit3d with Hydra configuration."""

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from injector import Injector
from omegaconf import DictConfig, OmegaConf

from maskgit3d.application.pipeline import FabricTrainingPipeline

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
    """Create data provider configuration based on dataset type.

    Different data providers accept different parameters:
    - SimpleDataProvider: num_train, num_val, num_test
    - MedMnist3DDataProvider: dataset_type, input_size, data_root, download
    - BraTSDataProvider: data_dir, modalities, train_ratio, val_ratio, test_ratio, etc.
    """
    dataset = cfg.dataset
    model = cfg.model
    dataset_type = dataset.get("type", "simple")

    # Helper to convert list to tuple
    def to_tuple(val):
        return tuple(val) if isinstance(val, list) else val

    common_params = {
        "batch_size": dataset.get("batch_size", 4),
        "num_workers": dataset.get("num_workers", 0),
        # Use crop_size/roi_size from dataset config, fallback to image_size
        "crop_size": to_tuple(dataset.get("crop_size", (model.get("image_size", 64),) * 3)),
        "roi_size": to_tuple(dataset.get("roi_size", (model.get("image_size", 64),) * 3)),
        # Keep spatial_size for backward compatibility
        "spatial_size": (
            model.get("image_size", 64),
            model.get("image_size", 64),
            model.get("image_size", 64),
        ),
    }

    if dataset_type == "simple":
        params = {
            **common_params,
            "num_train": dataset.get("num_train", 100),
            "num_val": dataset.get("num_val", 20),
            "num_test": dataset.get("num_test", 20),
            "in_channels": model.get("in_channels", 1),
            "out_channels": model.get("in_channels", 1),
        }
    elif dataset_type == "medmnist3d":
        dataset_name = dataset.get("dataset_name", "organmnist3d")
        medmnist_type = dataset_name.replace("mnist3d", "")
        params = {
            **common_params,
            "dataset_type": medmnist_type,
            "input_size": dataset.get("input_size", 28),
            "data_root": dataset.get("data_dir", dataset.get("data_root", "./data")),
            "download": dataset.get("download", True),
            "in_channels": model.get("in_channels", 1),
            "pin_memory": dataset.get("pin_memory", True),
            "drop_last_train": dataset.get("drop_last_train", True),
        }
    elif dataset_type in ("organ", "nodule", "adrenal", "vessel", "fracture", "synapse"):
        params = {
            **common_params,
            "dataset_type": dataset_type,
            "input_size": dataset.get("input_size", 28),
            "data_root": dataset.get("data_dir", dataset.get("data_root", "./data")),
            "download": dataset.get("download", True),
            "in_channels": model.get("in_channels", 1),
            "pin_memory": dataset.get("pin_memory", True),
            "drop_last_train": dataset.get("drop_last_train", True),
        }
        return {"type": "medmnist3d", "params": params}
    elif dataset_type == "brats":
        params = {
            **common_params,
            "data_dir": dataset.get("data_dir", None),
            "modalities": dataset.get("modalities", None),
            "train_ratio": dataset.get("train_ratio", 0.7),
            "val_ratio": dataset.get("val_ratio", 0.15),
            "test_ratio": dataset.get("test_ratio", 0.15),
            "random_seed": dataset.get("random_seed", 42),
            "normalize_mode": dataset.get("normalize_mode", "zscore"),
            "version": dataset.get("version", "2023"),
            "task": dataset.get("task", "reconstruction"),
            "tumor_types": dataset.get("tumor_types", None),
            "data_dirs": dataset.get("data_dirs", None),
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return {"type": dataset_type, "params": params}


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
                freeze_vqvae = cfg.model.get("freeze_vqgan", True)
                self.model_module = MaskGITModelModule(
                    {"type": model_type, "params": model_params},
                    pretrained_vqvae_path=pretrained_path,
                    freeze_vqvae=freeze_vqvae,
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
        }
    elif model_type in ("vqgan", "vqgan3d"):
        channel_multipliers = list(cfg.model.get("channel_multipliers", [1, 2]))
        base_channel = 64
        num_channels = tuple(base_channel * m for m in channel_multipliers)
        num_levels = len(channel_multipliers)

        raw_res_blocks = cfg.model.get("num_res_blocks", 2)
        if isinstance(raw_res_blocks, int):
            num_res_blocks = tuple([raw_res_blocks] * num_levels)
        else:
            num_res_blocks = tuple(raw_res_blocks)

        # attn_resolutions → attention_levels: level i has resolution image_size // 2^i
        image_size = base_params["image_size"]
        attn_resolutions = list(cfg.model.get("attn_resolutions", []))
        attention_levels = tuple(
            (image_size // (2**i)) in attn_resolutions for i in range(num_levels)
        )

        return {
            "in_channels": base_params["in_channels"],
            "codebook_size": cfg.model.get("codebook_size", 1024),
            "embed_dim": base_params["embed_dim"],
            "latent_channels": base_params["latent_channels"],
            "num_channels": num_channels,
            "num_res_blocks": num_res_blocks,
            "attention_levels": attention_levels,
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
    if strategy_type == "vqgan":
        vqgan_cfg = cfg.training.get("vqgan", {})
        params = {
            "codebook_weight": vqgan_cfg.get("codebook_weight", 1.0),
            "pixel_loss_weight": vqgan_cfg.get("pixel_loss_weight", 1.0),
            "perceptual_weight": vqgan_cfg.get("perceptual_weight", 1.0),
            "disc_weight": vqgan_cfg.get("disc_weight", 0.1),
            "disc_start": vqgan_cfg.get("disc_start", 500),
        }
    else:
        params = {
            "mask_schedule_type": cfg.model.get("mask_schedule_type", "cosine"),
            "reconstruction_weight": 1.0,
        }
    return {
        "type": strategy_type,
        "params": params,
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

    # Get Hydra's runtime output directory
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    checkpoint_dir = output_dir / "checkpoints"

    logger.info("Output directory: %s", output_dir)
    logger.info("Checkpoint directory: %s", checkpoint_dir)

    module = create_module_from_config(cfg)
    injector = Injector([module])
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

    # Use Fabric configuration from config
    fabric_cfg = cfg.get("training", {}).get("fabric", {})
    accelerator = fabric_cfg.get("accelerator", "auto")
    devices = fabric_cfg.get("devices", "auto")
    strategy = fabric_cfg.get("strategy", "auto")
    precision = fabric_cfg.get("precision", "32-true")

    pipeline = FabricTrainingPipeline(
        model=model,
        data_provider=data_provider,
        training_strategy=training_strategy,
        optimizer_factory=optimizer_factory,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        checkpoint_dir=str(checkpoint_dir),
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
