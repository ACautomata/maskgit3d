from omegaconf import OmegaConf


def test_vqvae_model_config_creation() -> None:
    from maskgit3d.config.schemas import VQVAEModelConfig

    config = VQVAEModelConfig()

    conf = OmegaConf.structured(config)

    assert conf._target_ == "maskgit3d.models.vqvae.VQVAE"
    assert conf.in_channels == 1
    assert list(conf.num_channels) == [64, 128, 256, 256]


def test_vqvae_model_config_round_trip_serialization() -> None:
    from maskgit3d.config.schemas import VQVAEModelConfig

    config = VQVAEModelConfig(
        embedding_dim=128,
        num_channels=(32, 64, 128),
        attention_levels=(False, False, True),
        quantizer_type="fsq",
    )

    restored = config.to_conf().to_obj(VQVAEModelConfig)

    assert restored == config


def test_maskgit_model_config_round_trip_serialization() -> None:
    from maskgit3d.config.schemas import MaskGITModelConfig

    config = MaskGITModelConfig(hidden_size=512, num_layers=8, dropout=0.0)

    restored = config.to_conf().to_obj(MaskGITModelConfig)

    assert restored == config


def test_optimizer_scheduler_and_sliding_window_round_trip_serialization() -> None:
    from maskgit3d.config.schemas import OptimizerConfig, SchedulerConfig, SlidingWindowConfig

    optimizer = OptimizerConfig(_target_="torch.optim.AdamW", weight_decay=0.01)
    scheduler = SchedulerConfig(warmup_steps=250)
    sliding_window = SlidingWindowConfig(enabled=True, roi_size=(48, 48, 32), overlap=0.5)

    assert optimizer.to_conf().to_obj(OptimizerConfig) == optimizer
    assert scheduler.to_conf().to_obj(SchedulerConfig) == scheduler
    assert sliding_window.to_conf().to_obj(SlidingWindowConfig) == sliding_window


def test_task_and_train_config_round_trip_serialization() -> None:
    from maskgit3d.config.schemas import (
        CheckpointConfig,
        OptimizerConfig,
        SchedulerConfig,
        SlidingWindowConfig,
        TaskConfig,
        TrainConfig,
        VQVAEModelConfig,
    )

    task = TaskConfig(
        model_config=VQVAEModelConfig(latent_channels=128),
        optimizer_config=OptimizerConfig(),
        disc_optimizer_config=OptimizerConfig(_target_="torch.optim.AdamW", weight_decay=0.01),
        sliding_window=SlidingWindowConfig(enabled=True),
        num_splits=2,
    )
    train = TrainConfig(
        checkpoint=CheckpointConfig(checkpoint_path="checkpoints/last.ckpt"),
        task=task,
        scheduler=SchedulerConfig(warmup_steps=500),
    )

    restored_task = task.to_conf().to_obj(TaskConfig)
    restored_train = train.to_conf().to_obj(TrainConfig)

    assert restored_task.model_config == task.model_config
    assert restored_task.optimizer_config == task.optimizer_config
    assert restored_task.disc_optimizer_config == task.disc_optimizer_config
    assert restored_task.sliding_window == task.sliding_window
    assert restored_train == train
