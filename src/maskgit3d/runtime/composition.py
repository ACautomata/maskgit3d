from __future__ import annotations

from typing import Any, cast

from hydra.utils import get_class, instantiate
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

from ..inference import VQVAEReconstructor
from ..losses.vq_perceptual_loss import VQPerceptualLoss
from ..models.incontext.incontext_maskgit import InContextMaskGIT
from ..models.maskgit import MaskGIT
from ..models.vqvae.splitting import compute_downsampling_factor, resolve_num_splits
from ..tasks.gan_training_strategy import GANTrainingStrategy
from ..tasks.incontext_task import InContextMaskGITTask
from ..tasks.maskgit_task import MaskGITTask
from ..tasks.vqvae_task import VQVAETask
from ..training import MaskGITTrainingSteps, VQVAETrainingSteps
from ..training.incontext_steps import InContextTrainingSteps
from .checkpoints import load_vqvae_from_checkpoint
from .model_factory import create_maskgit_model, create_vqvae_model
from .optimizer_factory import TransformerOptimizerFactory


def _get_task_target(cfg: DictConfig) -> str:
    task_target = cfg.task.get("_target_")
    if task_target is None:
        raise ValueError("cfg.task._target_ must be set.")
    return str(task_target)


def _build_task_kwargs(cfg: DictConfig, is_vqvae: bool = True) -> dict[str, object]:
    task_kwargs: dict[str, object] = {
        "_recursive_": False,
    }
    if cfg.get("model") is not None:
        task_kwargs["model_config"] = cfg.model
    if cfg.get("optimizer") is not None:
        task_kwargs["optimizer_config"] = cfg.optimizer
    if is_vqvae:
        # VQVAE tasks need separate discriminator optimizer config
        if cfg.get("disc_optimizer") is not None:
            task_kwargs["disc_optimizer_config"] = cfg.disc_optimizer
        elif cfg.get("optimizer") is not None:
            task_kwargs["disc_optimizer_config"] = cfg.optimizer
    if cfg.get("data") is not None and is_vqvae:
        task_kwargs["data_config"] = cfg.data
    return task_kwargs


def _resolved_vqvae_model_config(cfg: DictConfig) -> DictConfig:
    model_config = cfg.get("model")
    if model_config is None:
        raise ValueError("cfg.model must be set for build_vqvae_task.")

    crop_size = (
        tuple(cfg.data.crop_size)
        if cfg.get("data") is not None and cfg.data.get("crop_size") is not None
        else None
    )
    sliding_window_cfg = cfg.task.get("sliding_window") or {}
    roi_size = (
        tuple(sliding_window_cfg["roi_size"])
        if sliding_window_cfg.get("roi_size") is not None
        else None
    )
    dim_split = int(cfg.task.get("dim_split", model_config.get("dim_split", 0)))
    requested_num_splits = cfg.task.get("num_splits")
    resolved_num_splits, _ = resolve_num_splits(
        crop_size=crop_size,
        roi_size=roi_size,
        num_channels=tuple(model_config.num_channels),
        dim_split=dim_split,
        requested_num_splits=requested_num_splits,
    )
    return cast(
        DictConfig,
        OmegaConf.merge(
            model_config,
            {"num_splits": resolved_num_splits, "dim_split": dim_split},
        ),
    )


def _to_dict_config(value: Any) -> dict[str, Any]:
    container = OmegaConf.to_container(value, resolve=True)
    if isinstance(container, dict):
        return {str(key): item for key, item in container.items()}
    return {}


def _is_task_instance(task: object, task_type: type[object]) -> bool:
    return isinstance(task, task_type) or task.__class__.__name__ == task_type.__name__


def build_vqvae_task(cfg: DictConfig) -> VQVAETask:
    task_kwargs = _build_task_kwargs(cfg, is_vqvae=True)
    task_kwargs["model_config"] = None
    task = instantiate(cfg.task, **task_kwargs)
    if not _is_task_instance(task, VQVAETask):
        raise TypeError("build_vqvae_task only supports VQVAETask.")

    model_config = _resolved_vqvae_model_config(cfg)
    vqvae = create_vqvae_model(model_config)
    loss_fn = VQPerceptualLoss(
        disc_in_channels=int(model_config.get("out_channels", 1)),
        disc_num_layers=3,
        disc_ndf=64,
        disc_norm="instance",
        disc_loss=str(cfg.task.get("disc_loss", "hinge")),
        lambda_l1=float(cfg.task.get("lambda_l1", 1.0)),
        lambda_vq=float(cfg.task.get("lambda_vq", 1.0)),
        lambda_perceptual=float(cfg.task.get("lambda_perceptual", 0.1)),
        discriminator_weight=float(cfg.task.get("lambda_gan", 0.1)),
        disc_start=int(cfg.task.get("disc_start", 0)),
        disc_factor=float(cfg.task.get("disc_factor", 1.0)),
        use_adaptive_weight=bool(cfg.task.get("use_adaptive_weight", True)),
        adaptive_weight_max=float(cfg.task.get("adaptive_weight_max", 100.0)),
        perceptual_network=str(cfg.task.get("perceptual_network", "alex")),
        use_perceptual=bool(cfg.task.get("use_perceptual", True)),
    )
    gan_strategy = GANTrainingStrategy(
        float(cfg.task.get("gradient_clip_val", 1.0)),
        bool(cfg.task.get("gradient_clip_enabled", True)),
    )
    sliding_window_cfg = _to_dict_config(cfg.task.get("sliding_window") or OmegaConf.create({}))
    downsampling_factor = compute_downsampling_factor(list(model_config.num_channels))
    reconstructor = VQVAEReconstructor(
        sliding_window=sliding_window_cfg,
        downsampling_factor=downsampling_factor,
    )
    training_steps = VQVAETrainingSteps(
        vqvae=vqvae,
        loss_fn=loss_fn,
        gan_strategy=gan_strategy,
        reconstructor=reconstructor,
    )

    task.vqvae = vqvae
    task.loss_fn = loss_fn
    task.gan_strategy = gan_strategy
    task.sliding_window_cfg = sliding_window_cfg
    task._downsampling_factor = downsampling_factor
    task.reconstructor = reconstructor
    task.training_steps = training_steps
    task.hparams["model_config"] = model_config
    vqvae.enable_gradient_checkpointing()
    return cast(VQVAETask, task)


def build_maskgit_task(cfg: DictConfig) -> MaskGITTask:
    task_kwargs = _build_task_kwargs(cfg, is_vqvae=False)
    task_kwargs["model_config"] = None

    # Validate vqvae_ckpt_path before instantiation to provide clear error message
    ckpt_path = cfg.task.get("vqvae_ckpt_path")
    if ckpt_path is None:
        raise ValueError(
            "task.vqvae_ckpt_path is required for MaskGIT training. "
            "Please provide a path to a trained VQVAE checkpoint."
        )
    task_kwargs["vqvae_ckpt_path"] = ckpt_path

    task = instantiate(cfg.task, **task_kwargs)
    if not _is_task_instance(task, MaskGITTask):
        raise TypeError("build_maskgit_task only supports MaskGITTask.")

    vqvae = load_vqvae_from_checkpoint(str(ckpt_path))
    vqvae.eval()
    vqvae.requires_grad_(False)

    model_config = cfg.get("model")
    if model_config is not None:
        maskgit = create_maskgit_model(model_config, vqvae)
    else:
        maskgit = MaskGIT(
            vqvae=vqvae,
            hidden_size=int(cfg.task.get("hidden_size", 768)),
            num_layers=int(cfg.task.get("num_layers", 12)),
            num_heads=int(cfg.task.get("num_heads", 12)),
            mlp_ratio=float(cfg.task.get("mlp_ratio", 4.0)),
            dropout=float(cfg.task.get("dropout", 0.1)),
            gamma_type=str(cfg.task.get("gamma_type", "cosine")),
            sliding_window=_to_dict_config(cfg.task.get("sliding_window") or OmegaConf.create({})),
        )
    training_steps = MaskGITTrainingSteps(maskgit=maskgit)

    optimizer_factory = TransformerOptimizerFactory(
        lr=float(cfg.task.get("lr", 1e-4)),
        weight_decay=float(cfg.task.get("weight_decay", 0.01)),
        warmup_steps=int(cfg.task.get("warmup_steps", 1000)),
        optimizer_config=cfg.get("optimizer"),
    )

    task.vqvae = vqvae
    task.maskgit = maskgit
    task.training_steps = training_steps
    task.optimizer_factory = optimizer_factory
    if model_config is not None:
        task.hparams["model_config"] = model_config
    return cast(MaskGITTask, task)


def build_incontext_task(cfg: DictConfig) -> InContextMaskGITTask:
    """Build InContextMaskGITTask from Hydra config.

    Args:
        cfg: Hydra config with task parameters

    Returns:
        InContextMaskGITTask with all components initialized
    """
    task_kwargs = _build_task_kwargs(cfg, is_vqvae=False)
    task_kwargs["model_config"] = None

    # Validate vqvae_ckpt_path
    ckpt_path = cfg.task.get("vqvae_ckpt_path")
    if ckpt_path is None:
        raise ValueError(
            "task.vqvae_ckpt_path is required for InContextMaskGIT training. "
            "Please provide a path to a trained VQVAE checkpoint."
        )
    task_kwargs["vqvae_ckpt_path"] = ckpt_path

    task = instantiate(cfg.task, **task_kwargs)
    if not _is_task_instance(task, InContextMaskGITTask):
        raise TypeError("build_incontext_task only supports InContextMaskGITTask.")

    vqvae = load_vqvae_from_checkpoint(str(ckpt_path))
    vqvae.eval()
    vqvae.requires_grad_(False)

    incontext_model = InContextMaskGIT(
        vqvae=vqvae,
        num_modalities=int(cfg.task.get("num_modalities", 4)),
        hidden_size=int(cfg.task.get("hidden_size", 768)),
        num_layers=int(cfg.task.get("num_layers", 12)),
        num_heads=int(cfg.task.get("num_heads", 12)),
        mlp_ratio=float(cfg.task.get("mlp_ratio", 4.0)),
        dropout=float(cfg.task.get("dropout", 0.1)),
        gamma_type=str(cfg.task.get("gamma_type", "cosine")),
        sliding_window_cfg=_to_dict_config(cfg.task.get("sliding_window") or OmegaConf.create({})),
    )

    training_steps = InContextTrainingSteps(model=incontext_model)

    optimizer_factory = TransformerOptimizerFactory(
        lr=float(cfg.task.get("lr", 2e-4)),
        weight_decay=float(cfg.task.get("weight_decay", 0.05)),
        warmup_steps=int(cfg.task.get("warmup_steps", 1000)),
        optimizer_config=cfg.get("optimizer"),
    )

    task.vqvae = vqvae
    task.incontext_model = incontext_model
    task.training_steps = training_steps
    task.optimizer_factory = optimizer_factory
    return cast(InContextMaskGITTask, task)


def build_training_task(cfg: DictConfig) -> VQVAETask | MaskGITTask | InContextMaskGITTask:
    """Build training task by dispatching to specialized builders.

    This function determines the task type from cfg.task._target_ and
    delegates to the appropriate specialized builder (build_vqvae_task
    or build_maskgit_task) which properly constructs all components.
    """
    task_target = _get_task_target(cfg)

    # Dispatch to specialized builders based on task type
    # Check "incontext" before "maskgit" since InContextMaskGITTask contains "maskgit"
    if "incontext" in task_target.lower():
        return build_incontext_task(cfg)
    if "vqvae" in task_target.lower():
        return build_vqvae_task(cfg)
    if "maskgit" in task_target.lower():
        return build_maskgit_task(cfg)

    raise TypeError(
        f"build_training_task only supports VQVAETask, MaskGITTask, and InContextMaskGITTask. Got: {task_target}"
    )


def build_eval_task(cfg: DictConfig, ckpt_path: str) -> LightningModule:
    task_target = _get_task_target(cfg)
    task_class: type[LightningModule] = get_class(task_target)
    return task_class.load_from_checkpoint(
        ckpt_path,
        weights_only=False,
    )
