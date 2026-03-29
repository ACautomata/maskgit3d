"""Characterization tests for task configuration contracts.

These tests verify that the current Hydra config groups and CLI overrides
work as expected, ensuring that future refactoring won't break existing UX.

Updated to reflect the cleaned config structure (no unused model_config, optimizer_config in task configs).
"""

from pathlib import Path

from hydra import compose, initialize_config_dir

CONFIG_DIR = str(Path(__file__).parent.parent.parent / "src/maskgit3d/conf")


class TestTaskConfigContracts:
    """Verify task config groups have required structure."""

    def test_vqvae_task_config_has_required_params(self) -> None:
        """VQVAE task should have all required training parameters."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["task=vqvae"])

        # VQVAE task has training parameters directly (not via model_config)
        assert hasattr(cfg.task, "lr_g")
        assert hasattr(cfg.task, "lr_d")
        assert hasattr(cfg.task, "lambda_l1")
        assert hasattr(cfg.task, "lambda_vq")
        assert hasattr(cfg.task, "lambda_gan")
        assert hasattr(cfg.task, "sliding_window")

    def test_vqvae_task_target_is_correct(self) -> None:
        """VQVAE task should target VQVAETask."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["task=vqvae"])

        assert cfg.task._target_ == "maskgit3d.tasks.vqvae_task.VQVAETask"

    def test_maskgit_task_config_has_required_params(self) -> None:
        """MaskGIT task should have all required training parameters."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["task=maskgit", "model=maskgit"])

        # MaskGIT task has training parameters directly
        assert hasattr(cfg.task, "lr")
        assert hasattr(cfg.task, "weight_decay")
        assert hasattr(cfg.task, "warmup_steps")
        assert hasattr(cfg.task, "vqvae_ckpt_path")

    def test_maskgit_task_target_is_correct(self) -> None:
        """MaskGIT task should target MaskGITTask."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["task=maskgit"])

        assert cfg.task._target_ == "maskgit3d.tasks.maskgit_task.MaskGITTask"


class TestHydraOverrideContracts:
    """Verify Hydra CLI overrides work as expected."""

    def test_task_vqvae_override_resolves_correctly(self) -> None:
        """task=vqvae override should resolve to VQVAETask."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["task=vqvae"])

        assert cfg.task._target_ == "maskgit3d.tasks.vqvae_task.VQVAETask"

    def test_task_maskgit_override_resolves_correctly(self) -> None:
        """task=maskgit override should resolve to MaskGITTask."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["task=maskgit"])

        assert cfg.task._target_ == "maskgit3d.tasks.maskgit_task.MaskGITTask"

    def test_data_medmnist3d_override_resolves_correctly(self) -> None:
        """data=medmnist3d override should resolve to MedMNIST3DDataModule."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["data=medmnist3d"])

        assert cfg.data._target_ == "maskgit3d.data.medmnist.datamodule.MedMNIST3DDataModule"

    def test_model_vqvae_override_resolves_correctly(self) -> None:
        """model=vqvae override should resolve to VQVAE config."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["model=vqvae"])

        # Model config is at cfg.model (not cfg.task.model_config)
        assert cfg.model._target_ == "maskgit3d.models.vqvae.VQVAE"

    def test_model_maskgit_override_resolves_correctly(self) -> None:
        """model=maskgit override should resolve to MaskGIT model config."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["model=maskgit"])

        assert cfg.model._target_ == "maskgit3d.models.maskgit.MaskGIT"

    def test_optimizer_adam_override_resolves_correctly(self) -> None:
        """optimizer=adam override should resolve to AdamW optimizer config (project uses AdamW by default)."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["optimizer=adam"])

        assert hasattr(cfg, "optimizer")
        assert cfg.optimizer._target_ == "torch.optim.AdamW"

    def test_optimizer_adamw_override_resolves_correctly(self) -> None:
        """optimizer=adamw override should resolve to AdamW optimizer config."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["optimizer=adamw"])

        assert hasattr(cfg, "optimizer")
        assert cfg.optimizer._target_ == "torch.optim.AdamW"


class TestCompositionRootContracts:
    """Verify train.py and eval.py remain thin entrypoints.

    These tests ensure that model/optimizer/scheduler instantiation
    happens INSIDE task constructors, not at the composition root.
    """

    def test_train_cfg_has_expected_top_level_keys(self) -> None:
        """Train config should have data, task, trainer at top level."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train")

        assert "data" in cfg
        assert "task" in cfg
        assert "trainer" in cfg
        assert "model" in cfg

    def test_train_cfg_has_callbacks(self) -> None:
        """Train config should have callbacks configured."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train")

        assert hasattr(cfg, "callbacks")
        assert cfg.callbacks is not None

    def test_eval_cfg_requires_ckpt_path(self) -> None:
        """Eval config should require ckpt_path."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="eval")

        # ckpt_path is in the defaults
        assert hasattr(cfg, "ckpt_path")


class TestExistingConfigFilesExist:
    """Sanity check that all referenced config files exist."""

    def test_model_vqvae_yaml_exists(self) -> None:
        assert (Path(CONFIG_DIR) / "model" / "vqvae.yaml").exists()

    def test_model_maskgit_yaml_exists(self) -> None:
        assert (Path(CONFIG_DIR) / "model" / "maskgit.yaml").exists()

    def test_optimizer_adam_yaml_exists(self) -> None:
        assert (Path(CONFIG_DIR) / "optimizer" / "adam.yaml").exists()

    def test_optimizer_adamw_yaml_exists(self) -> None:
        assert (Path(CONFIG_DIR) / "optimizer" / "adamw.yaml").exists()

    def test_scheduler_cosine_yaml_exists(self) -> None:
        assert (Path(CONFIG_DIR) / "scheduler" / "cosine_warmup.yaml").exists()

    def test_data_medmnist3d_yaml_exists(self) -> None:
        assert (Path(CONFIG_DIR) / "data" / "medmnist3d.yaml").exists()

    def test_callbacks_vqvae_yaml_exists(self) -> None:
        assert (Path(CONFIG_DIR) / "callbacks" / "vqvae.yaml").exists()

    def test_callbacks_maskgit_yaml_exists(self) -> None:
        assert (Path(CONFIG_DIR) / "callbacks" / "maskgit.yaml").exists()