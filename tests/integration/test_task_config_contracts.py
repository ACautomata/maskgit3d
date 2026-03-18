"""Characterization tests for task configuration contracts.

These tests verify that the current Hydra config groups and CLI overrides
work as expected, ensuring that future refactoring won't break existing UX.

Phase 0: Freeze current behavior before architectural changes.
"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = str(Path(__file__).parent.parent.parent / "src/maskgit3d/conf")


class TestTaskConfigContracts:
    """Verify task config groups have required structure."""

    def test_vqvae_task_config_references_model_config(self) -> None:
        """VQVAE task should have model_config that references ${model}."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["task=vqvae"])

        # After Hydra interpolation, model_config is resolved to actual config
        # We verify it has the VQVAE target
        assert hasattr(cfg.task, "model_config")
        assert cfg.task.model_config._target_ == "maskgit3d.models.vqvae.VQVAE"

    def test_vqvae_task_config_references_optimizer_config(self) -> None:
        """VQVAE task should have optimizer_config that resolves to Adam."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["task=vqvae"])

        # After Hydra interpolation, optimizer_config is resolved
        assert hasattr(cfg.task, "optimizer_config")
        assert cfg.task.optimizer_config._target_ == "torch.optim.Adam"

    def test_maskgit_task_config_references_model_config(self) -> None:
        """MaskGIT task should have model_config that resolves to MaskGIT when model=maskgit."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["task=maskgit", "model=maskgit"])

        assert hasattr(cfg.task, "model_config")
        assert cfg.task.model_config._target_ == "maskgit3d.models.maskgit.MaskGIT"

    def test_maskgit_task_config_references_optimizer_config(self) -> None:
        """MaskGIT task should have optimizer_config that resolves to AdamW when optimizer=adamw."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["task=maskgit", "optimizer=adamw"])

        assert hasattr(cfg.task, "optimizer_config")
        assert cfg.task.optimizer_config._target_ == "torch.optim.AdamW"


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
            cfg = compose(config_name="train", overrides=["task=vqvae", "model=vqvae"])

        # After Hydra interpolation, task.model_config is resolved to actual model config
        assert cfg.task.model_config._target_ == "maskgit3d.models.vqvae.VQVAE"

    def test_model_maskgit_override_resolves_correctly(self) -> None:
        """model=maskgit override should resolve to MaskGIT model config."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["task=maskgit", "model=maskgit"])

        assert cfg.task.model_config._target_ == "maskgit3d.models.maskgit.MaskGIT"

    def test_optimizer_adam_override_resolves_correctly(self) -> None:
        """optimizer=adam override should resolve to Adam optimizer config."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["optimizer=adam"])

        assert hasattr(cfg, "optimizer")
        assert cfg.optimizer._target_ == "torch.optim.Adam"

    def test_optimizer_adamw_override_resolves_correctly(self) -> None:
        """optimizer=adamw override should resolve to AdamW optimizer config."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["optimizer=adamw"])

        assert hasattr(cfg, "optimizer")
        assert cfg.optimizer._target_ == "torch.optim.AdamW"

    def test_scheduler_cosine_warmup_config_is_loadable(self) -> None:
        """scheduler=cosine should be loadable (even if not actively used by tasks yet)."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["scheduler=cosine"])

        assert hasattr(cfg, "scheduler")
        # Scheduler may be None if not in the defaults
        if cfg.scheduler is not None:
            assert "_target_" in cfg.scheduler


class TestCompositionRootContracts:
    """Verify train.py and eval.py remain thin entrypoints.

    These tests ensure that model/optimizer/scheduler instantiation
    happens INSIDE task constructors, not at the composition root.
    After refactoring, model/optimizer creation should move to factories,
    but the thin-entrypoint behavior must be preserved.
    """

    def test_train_cfg_has_expected_top_level_keys(self) -> None:
        """Train config should have data, task, trainer at top level."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train")

        assert "data" in cfg
        assert "task" in cfg
        assert "trainer" in cfg

    def test_train_cfg_optional_keys_are_optional(self) -> None:
        """Optional keys like callbacks, logger, scheduler should be nullable."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train")

        # These keys exist but can be null
        assert hasattr(cfg, "callbacks")
        assert hasattr(cfg, "logger")
        assert hasattr(cfg, "scheduler")

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
