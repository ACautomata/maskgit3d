from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from omegaconf import DictConfig, OmegaConf

SchemaT = TypeVar("SchemaT", bound="ConfigSchema")


def _tuple_or_none(values: Any) -> tuple[Any, ...] | None:
    if values is None:
        return None
    return tuple(values)


def _as_two_float_tuple(values: Any) -> tuple[float, float]:
    first, second = tuple(values)
    return (float(first), float(second))


def _as_three_int_tuple(values: Any) -> tuple[int, int, int]:
    first, second, third = tuple(values)
    return (int(first), int(second), int(third))


def _dictconfig_to_container(conf: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(conf, DictConfig):
        data = OmegaConf.to_container(conf, resolve=True)
        if not isinstance(data, dict):
            raise TypeError("Expected DictConfig to contain a mapping")
        return {str(key): value for key, value in data.items()}
    return {str(key): value for key, value in dict(conf).items()}


@dataclass
class ConfigSchema:
    def to_conf(self) -> DictConfig:
        return cast(DictConfig, OmegaConf.structured(self))

    @classmethod
    def from_conf(cls: type[SchemaT], conf: DictConfig | dict[str, Any]) -> SchemaT:
        raw = _dictconfig_to_container(conf)
        return cast(SchemaT, cls(**raw))


def _dictconfig_to_obj(self: DictConfig, schema_cls: type[SchemaT]) -> SchemaT:
    return schema_cls.from_conf(self)


if not hasattr(DictConfig, "to_obj"):
    DictConfig.to_obj = _dictconfig_to_obj  # type: ignore[attr-defined]


@dataclass
class VQVAEModelConfig(ConfigSchema):
    _target_: str = "maskgit3d.models.vqvae.VQVAE"
    in_channels: int = 1
    out_channels: int = 1
    latent_channels: int = 256
    num_embeddings: int = 8192
    embedding_dim: int = 256
    num_channels: tuple[int, ...] = (64, 128, 256, 256)
    num_res_blocks: tuple[int, ...] = (2, 2, 2, 1)
    attention_levels: tuple[bool, ...] = (False, False, False, True)
    commitment_cost: float = 0.5
    quantizer_type: str = "vq"
    fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5, 5)
    num_splits: int = 1
    dim_split: int = 0

    def __post_init__(self) -> None:
        self.num_channels = tuple(self.num_channels)
        self.num_res_blocks = tuple(self.num_res_blocks)
        self.attention_levels = tuple(self.attention_levels)
        self.fsq_levels = tuple(self.fsq_levels)


@dataclass
class MaskGITModelConfig(ConfigSchema):
    _target_: str = "maskgit3d.models.maskgit.MaskGIT"
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    gamma_type: str = "cosine"
    num_iterations: int = 12
    temperature: float = 1.0


@dataclass
class OptimizerConfig(ConfigSchema):
    _target_: str = "torch.optim.Adam"
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        self.betas = cast(tuple[float, float], _as_two_float_tuple(self.betas))


@dataclass
class SchedulerConfig(ConfigSchema):
    warmup_steps: int = 1000


@dataclass
class SlidingWindowConfig(ConfigSchema):
    enabled: bool = False
    roi_size: tuple[int, int, int] = (32, 32, 32)
    overlap: float = 0.25
    mode: str = "gaussian"
    sigma_scale: float = 0.125
    sw_batch_size: int = 1
    sw_device: str | None = None  # Device for window processing, None = input tensor's device
    device: str | None = (
        None  # Device for output aggregation, None = input tensor's device (safe for metrics)
    )

    def __post_init__(self) -> None:
        self.roi_size = cast(tuple[int, int, int], _as_three_int_tuple(self.roi_size))


@dataclass
class TaskConfig(ConfigSchema):
    _target_: str = "maskgit3d.tasks.vqvae_task.VQVAETask"
    model_config: Any = None
    optimizer_config: OptimizerConfig | None = None
    disc_optimizer_config: OptimizerConfig | None = None
    data_config: Any = None
    vqvae_ckpt_path: str | None = None
    lr_g: float = 5.0e-5
    lr_d: float = 1.0e-4
    lr: float = 2e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    lambda_l1: float = 1.0
    lambda_vq: float = 1.0
    lambda_gan: float = 0.5
    use_perceptual: bool = True
    lambda_perceptual: float = 0.1
    perceptual_network: str = "alex"
    disc_start: int = 1000
    disc_factor: float = 1.0
    use_adaptive_weight: bool = True
    adaptive_weight_max: float = 100.0
    disc_loss: str = "hinge"
    gradient_clip_enabled: bool = True
    gradient_clip_val: float = 1.0
    sliding_window: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    num_splits: int | None = None
    dim_split: int = 0
    in_channels: int = 1
    out_channels: int = 1
    latent_channels: int = 256
    num_embeddings: int = 8192
    embedding_dim: int = 256
    num_channels: tuple[int, ...] = (64, 128, 256)
    num_res_blocks: tuple[int, ...] = (2, 2, 2)
    attention_levels: tuple[bool, ...] = (False, False, False)
    commitment_cost: float = 0.25
    quantizer_type: str = "vq"
    fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5, 5)
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    gamma_type: str = "cosine"

    def __post_init__(self) -> None:
        self.num_channels = tuple(self.num_channels)
        self.num_res_blocks = tuple(self.num_res_blocks)
        self.attention_levels = tuple(self.attention_levels)
        self.fsq_levels = tuple(self.fsq_levels)
        if not isinstance(self.sliding_window, SlidingWindowConfig):
            self.sliding_window = SlidingWindowConfig(**self.sliding_window)
        if self.optimizer_config is not None and not isinstance(
            self.optimizer_config, OptimizerConfig
        ):
            self.optimizer_config = OptimizerConfig(**self.optimizer_config)
        if self.disc_optimizer_config is not None and not isinstance(
            self.disc_optimizer_config, OptimizerConfig
        ):
            self.disc_optimizer_config = OptimizerConfig(**self.disc_optimizer_config)
        if self.model_config is not None and not isinstance(
            self.model_config, VQVAEModelConfig | MaskGITModelConfig
        ):
            model_target = self.model_config.get("_target_")
            if model_target == VQVAEModelConfig._target_:
                self.model_config = VQVAEModelConfig(**self.model_config)
            elif model_target == MaskGITModelConfig._target_:
                self.model_config = MaskGITModelConfig(**self.model_config)

    @classmethod
    def from_conf(cls, conf: DictConfig | dict[str, Any]) -> TaskConfig:
        raw = _dictconfig_to_container(conf)
        return cls(**raw)


@dataclass
class CheckpointConfig(ConfigSchema):
    checkpoint_path: str | None = None
    ckpt_path: str | None = None


@dataclass
class TrainConfig(ConfigSchema):
    seed: int = 42
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    scheduler: SchedulerConfig | None = None
    callbacks: Any = None
    logger: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, CheckpointConfig):
            self.checkpoint = CheckpointConfig(**self.checkpoint)
        if not isinstance(self.task, TaskConfig):
            self.task = TaskConfig(**self.task)
        if self.scheduler is not None and not isinstance(self.scheduler, SchedulerConfig):
            self.scheduler = SchedulerConfig(**self.scheduler)

    @classmethod
    def from_conf(cls, conf: DictConfig | dict[str, Any]) -> TrainConfig:
        raw = _dictconfig_to_container(conf)
        return cls(**raw)
