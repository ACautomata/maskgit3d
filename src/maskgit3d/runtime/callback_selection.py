from typing import Any

from omegaconf import OmegaConf

_KNOWN_CALLBACK_KEYS = {
    "vqvae_metrics",
    "maskgit_metrics",
    "vqvae_training_losses",
    "reconstruction_loss",
    "fid_logging",
    "masked_cross_entropy",
    "mask_accuracy",
    "best_checkpoint",
    "last_checkpoint",
    "early_stopping",
    "best_checkpoint_maskgit",
    "last_checkpoint_maskgit",
    "early_stopping_maskgit",
    "sample_saving",
    "lr_monitor",
    "nan_detection",
    "training_time",
    "gradient_norm",
    "training_stability",
    "cuda_memory",
}

_TRAIN_EXCLUDES = {
    "vqvae": {
        "maskgit_metrics",
        "masked_cross_entropy",
        "mask_accuracy",
        "best_checkpoint_maskgit",
        "last_checkpoint_maskgit",
        "early_stopping_maskgit",
    },
    "maskgit": {
        "vqvae_metrics",
        "vqvae_training_losses",
        "reconstruction_loss",
        "fid_logging",
        "best_checkpoint",
        "last_checkpoint",
        "early_stopping",
    },
}

_EVAL_BASE_KEEP = {"sample_saving"}
_EVAL_STAGE_EXCLUDES = {
    "vqvae": {
        "maskgit_metrics",
        "masked_cross_entropy",
        "mask_accuracy",
    },
    "maskgit": {
        "vqvae_metrics",
        "vqvae_training_losses",
        "reconstruction_loss",
        "fid_logging",
    },
}


def _task_family(task: Any) -> str:
    task_name = task.__class__.__name__.lower()
    if "maskgit" in task_name:
        return "maskgit"
    return "vqvae"


def select_callback_config(callbacks_cfg: Any, task: Any, *, stage: str) -> Any:
    if callbacks_cfg is None:
        return None
    if not (isinstance(callbacks_cfg, dict) or OmegaConf.is_dict(callbacks_cfg)):
        return callbacks_cfg

    callback_keys = set(callbacks_cfg.keys())
    if not (callback_keys & _KNOWN_CALLBACK_KEYS):
        return callbacks_cfg

    family = _task_family(task)
    unknown_keys = callback_keys - _KNOWN_CALLBACK_KEYS

    if stage == "train":
        excluded = _TRAIN_EXCLUDES[family]
        selected_keys = (callback_keys - excluded) | unknown_keys
    else:
        keep = _EVAL_BASE_KEEP | {
            f"{family}_metrics",
            "reconstruction_loss",
            "fid_logging",
            "masked_cross_entropy",
            "mask_accuracy",
        }
        selected_keys = (callback_keys & keep) | unknown_keys

    selected = {key: callbacks_cfg[key] for key in callbacks_cfg.keys() if key in selected_keys}
    return OmegaConf.create(selected)
