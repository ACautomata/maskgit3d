from unittest.mock import MagicMock

from omegaconf import OmegaConf

import maskgit3d.cli.train as train_cli


class _Binder:
    def __init__(self):
        self.bound = []
        self.installed = []

    def bind(self, interface, to):
        self.bound.append((interface, to))

    def install(self, module):
        self.installed.append(module)


def test_create_model_params_vqgan_with_tuple_num_res_blocks() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "type": "vqgan",
                "channel_multipliers": [1, 2, 4],
                "num_res_blocks": [1, 2, 3],
                "attn_resolutions": [32],
            }
        }
    )
    base_params = {
        "image_size": 64,
        "in_channels": 1,
        "embed_dim": 256,
        "latent_channels": 256,
    }

    result = train_cli._create_model_params(cfg, "vqgan", base_params)
    assert result["num_res_blocks"] == (1, 2, 3)
    assert result["attention_levels"] == (False, True, False)


def test_create_module_from_config_non_maskgit_configure_installs(monkeypatch) -> None:
    class DummyModule:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr("maskgit3d.config.modules.DataModule", DummyModule)
    monkeypatch.setattr("maskgit3d.config.modules.TrainingModule", DummyModule)
    monkeypatch.setattr("maskgit3d.config.modules.InferenceModule", DummyModule)
    monkeypatch.setattr("maskgit3d.config.modules.ModelModule", DummyModule)

    cfg = OmegaConf.create(
        {
            "model": {"type": "vqgan"},
            "dataset": {"type": "simple", "batch_size": 2},
            "training": {"optimizer": {"lr": 1e-4, "type": "adam"}},
        }
    )

    module = train_cli.create_module_from_config(cfg)
    binder = _Binder()
    module.configure(binder)

    assert len(binder.installed) == 4
    assert len(binder.bound) == 0


def test_create_module_from_config_maskgit_binds_interfaces(monkeypatch) -> None:
    class DummyModule:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

    class DummyMaskModule:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

        def provide_maskgit_model(self):
            return "maskgit-model"

    monkeypatch.setattr("maskgit3d.config.modules.DataModule", DummyModule)
    monkeypatch.setattr("maskgit3d.config.modules.TrainingModule", DummyModule)
    monkeypatch.setattr("maskgit3d.config.modules.InferenceModule", DummyModule)
    monkeypatch.setattr("maskgit3d.config.modules.MaskGITModelModule", DummyMaskModule)

    cfg = OmegaConf.create(
        {
            "model": {
                "type": "maskgit",
                "in_channels": 1,
                "codebook_size": 128,
                "embed_dim": 64,
                "latent_channels": 16,
                "transformer_hidden": 32,
                "transformer_layers": 2,
                "transformer_heads": 2,
            },
            "dataset": {"type": "simple", "batch_size": 2},
            "training": {"optimizer": {"lr": 1e-4, "type": "adam"}},
        }
    )

    module = train_cli.create_module_from_config(cfg)
    binder = _Binder()
    module.configure(binder)

    assert len(binder.bound) == 2
    assert len(binder.installed) == 3


def test_main_builds_pipeline_and_runs(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "model": {"type": "vqgan", "name": "vqgan"},
            "dataset": {"type": "simple", "name": "simple"},
            "training": {"num_epochs": 3, "optimizer": {"lr": 1e-4, "type": "adam"}, "fabric": {}},
        }
    )

    fake_module = object()
    monkeypatch.setattr(train_cli, "create_module_from_config", lambda _cfg: fake_module)

    fake_injector = MagicMock()
    fake_injector.get.return_value = object()
    monkeypatch.setattr(train_cli, "Injector", lambda modules: fake_injector)

    # Mock HydraConfig
    mock_hydra_config = MagicMock()
    mock_runtime = MagicMock()
    mock_runtime.output_dir = "/tmp/test_output"
    mock_hydra_config.get.return_value.runtime = mock_runtime
    monkeypatch.setattr(train_cli, "HydraConfig", mock_hydra_config)

    pipeline = MagicMock()
    monkeypatch.setattr(train_cli, "FabricTrainingPipeline", lambda **kwargs: pipeline)

    train_cli.main.__wrapped__(cfg)
    pipeline.run.assert_called_once_with(num_epochs=3)
