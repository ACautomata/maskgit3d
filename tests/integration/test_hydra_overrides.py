from pathlib import Path

from hydra import compose, initialize_config_dir

CONFIG_DIR = str(Path(__file__).parent.parent.parent / "src/maskgit3d/conf")


def test_train_allows_task_and_dataset_group_overrides() -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="train", overrides=["task=maskgit", "dataset=medmnist3d"])

    assert cfg.task._target_ == "maskgit3d.tasks.maskgit_task.MaskGITTask"
    assert cfg.data._target_ == "maskgit3d.data.medmnist.datamodule.MedMNIST3DDataModule"


def test_eval_allows_task_and_dataset_group_overrides() -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="eval", overrides=["task=vqvae", "dataset=medmnist3d"])

    assert cfg.task._target_ == "maskgit3d.tasks.vqvae_task.VQVAETask"
    assert cfg.data._target_ == "maskgit3d.data.medmnist.datamodule.MedMNIST3DDataModule"
