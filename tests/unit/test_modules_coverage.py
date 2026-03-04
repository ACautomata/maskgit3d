from typing import Any, cast

import pytest
from injector import Injector

import maskgit3d.config.modules as modules
from maskgit3d.domain.interfaces import (
    DataProvider,
    InferenceStrategy,
    Metrics,
    OptimizerFactory,
    TrainingStrategy,
)


def test_model_module_maskgit_branch(monkeypatch) -> None:
    class DummyMaskGITModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("maskgit3d.infrastructure.maskgit.MaskGITModel", DummyMaskGITModel)

    model = modules.ModelModule({"type": "maskgit", "params": {"x": 1}}).provide_model()
    assert isinstance(model, DummyMaskGITModel)
    assert model.kwargs["x"] == 1


def test_model_module_vqvae_and_unknown_branch(monkeypatch) -> None:
    class DummyVQVAE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(modules, "VQVAE", DummyVQVAE)

    vqvae = modules.ModelModule(
        {
            "type": "vqvae",
            "params": {
                "in_channels": 1,
                "codebook_size": 8,
                "embed_dim": 4,
                "latent_channels": 4,
                "num_channels": [8],
                "num_res_blocks": [1],
                "attention_levels": [False],
            },
        }
    ).provide_model()
    assert isinstance(vqvae, DummyVQVAE)

    with pytest.raises(ValueError, match="Unknown model type"):
        modules.ModelModule({"type": "bad", "params": {}}).provide_model()


def test_data_module_provider_paths_and_error(monkeypatch) -> None:
    class DummySimple:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyMed:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyBrats:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("maskgit3d.infrastructure.data.dataset.SimpleDataProvider", DummySimple)
    monkeypatch.setattr(
        "maskgit3d.infrastructure.data.medmnist_provider.MedMnist3DDataProvider", DummyMed
    )
    monkeypatch.setattr(
        "maskgit3d.infrastructure.data.brats_provider.BraTSDataProvider", DummyBrats
    )

    simple = modules.DataModule(
        {"type": "simple", "params": {"batch_size": 2}}
    ).provide_data_provider()
    med = modules.DataModule(
        {"type": "medmnist3d", "params": {"dataset_type": "nodule"}}
    ).provide_data_provider()
    brats = modules.DataModule(
        {"type": "brats", "params": {"batch_size": 1}}
    ).provide_data_provider()

    assert isinstance(simple, DummySimple)
    assert isinstance(med, DummyMed)
    assert isinstance(brats, DummyBrats)

    with pytest.raises(ValueError, match="Unknown data type"):
        Injector([modules.DataModule({"type": "nope", "params": {}})]).get(DataProvider)


def test_training_and_inference_strategy_provider_paths(monkeypatch) -> None:
    monkeypatch.setattr(modules, "MaskGITTrainingStrategy", lambda **kwargs: ("mg", kwargs))
    monkeypatch.setattr(modules, "VQGANTrainingStrategy", lambda **kwargs: ("vq", kwargs))
    monkeypatch.setattr(modules, "MaskGITInference", lambda **kwargs: ("mg_inf", kwargs))
    monkeypatch.setattr(modules, "VQGANInference", lambda **kwargs: ("vq_inf", kwargs))

    t_mod = modules.TrainingModule(
        training_config={"type": "maskgit", "params": {"a": 1}},
        optimizer_config={"type": "adam", "params": {"lr": 1e-3}},
    )
    assert cast(Any, Injector([t_mod]).get(TrainingStrategy))[0] == "mg"

    t_mod2 = modules.TrainingModule(
        training_config={"type": "vqgan", "params": {"b": 2}},
        optimizer_config={"type": "adam", "params": {"lr": 1e-3}},
    )
    assert cast(Any, Injector([t_mod2]).get(TrainingStrategy))[0] == "vq"

    with pytest.raises(ValueError, match="Unknown strategy type"):
        Injector(
            [
                modules.TrainingModule(
                    training_config={"type": "bad", "params": {}},
                    optimizer_config={"type": "adam", "params": {}},
                )
            ]
        ).get(TrainingStrategy)

    i_mod = modules.InferenceModule(inference_config={"type": "maskgit", "params": {"x": 1}})
    assert cast(Any, Injector([i_mod]).get(InferenceStrategy))[0] == "mg_inf"

    i_mod2 = modules.InferenceModule(inference_config={"type": "vqgan", "params": {"y": 2}})
    assert cast(Any, Injector([i_mod2]).get(InferenceStrategy))[0] == "vq_inf"

    with pytest.raises(ValueError, match="Unknown inference type"):
        Injector([modules.InferenceModule(inference_config={"type": "bad", "params": {}})]).get(
            InferenceStrategy
        )


def test_training_module_unknown_optimizer_raises() -> None:
    module = modules.TrainingModule(
        training_config={"type": "maskgit", "params": {}},
        optimizer_config={"type": "bad", "params": {}},
    )
    with pytest.raises(ValueError, match="Unknown optimizer type"):
        Injector([module]).get(OptimizerFactory)


def test_inference_module_metrics_none_and_unknown() -> None:
    assert modules.InferenceModule(metrics_config={}).provide_metrics() is None

    with pytest.raises(ValueError, match="Unknown metrics type"):
        modules.InferenceModule(metrics_config={"type": "bad"}).provide_metrics()


def test_inference_module_metrics_vqgan(monkeypatch) -> None:
    monkeypatch.setattr(modules, "VQGANMetrics", lambda **kwargs: {"ok": kwargs})
    value = modules.InferenceModule(
        metrics_config={"type": "vqgan", "params": {"p": 1}}
    ).provide_metrics()
    assert value == {"ok": {"p": 1}}


def test_system_module_device_provider(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    device = modules.SystemModule().provide_device()
    assert str(device) == "cpu"


def test_maskgit_model_module_load_pretrained_checkpoint_key_variants(monkeypatch) -> None:
    class DummyVQVAE:
        def __init__(self):
            self.loaded = []

        def load_state_dict(self, value):
            self.loaded.append(value)

    cases = [
        {"model_state_dict": {"a": 1}},
        {"vqvae": {"b": 2}},
        {"vqgan": {"c": 3}},
        {"state_dict": {"d": 4}},
        {"z": 9},
    ]

    for payload in cases:
        monkeypatch.setattr(
            "maskgit3d.infrastructure.checkpoints.load_checkpoint",
            lambda *args, payload=payload, **kwargs: payload,
        )
        m = modules.MaskGITModelModule(
            model_config={"type": "maskgit", "params": {}}, freeze_vqvae=False
        )
        vq = DummyVQVAE()
        m._load_pretrained_vqvae(cast(Any, vq), "unused.ckpt")
        assert len(vq.loaded) == 1


def test_maskgit_model_module_provider_happy_path_and_unknown(monkeypatch) -> None:
    class Param:
        def __init__(self):
            self.requires_grad = True

    class DummyVQVAE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._params = [Param(), Param()]

        def parameters(self):
            return self._params

    class DummyTransformer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyMaskGITModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(modules, "VQVAE", DummyVQVAE)
    monkeypatch.setattr(
        "maskgit3d.infrastructure.maskgit.transformer.MaskGITTransformer", DummyTransformer
    )
    monkeypatch.setattr("maskgit3d.infrastructure.maskgit.MaskGITModel", DummyMaskGITModel)

    called = {"loaded": 0}

    def fake_load(self, vqvae, checkpoint_path):
        _ = (vqvae, checkpoint_path)
        called["loaded"] += 1

    monkeypatch.setattr(modules.MaskGITModelModule, "_load_pretrained_vqvae", fake_load)

    m = modules.MaskGITModelModule(
        model_config={
            "type": "maskgit",
            "params": {
                "in_channels": 1,
                "codebook_size": 8,
                "embed_dim": 4,
                "latent_channels": 4,
                "transformer_hidden": 16,
                "transformer_layers": 2,
                "transformer_heads": 2,
                "mask_ratio": 0.7,
            },
        },
        pretrained_vqvae_path="fake.ckpt",
        freeze_vqvae=True,
    )
    model = m.provide_maskgit_model()
    assert isinstance(model, DummyMaskGITModel)
    assert called["loaded"] == 1
    assert model.kwargs["mask_ratio"] == 0.7

    with pytest.raises(ValueError, match="Unknown MaskGIT model type"):
        modules.MaskGITModelModule(
            model_config={"type": "bad", "params": {}}
        ).provide_maskgit_model()


def test_create_fabric_pipeline_forwards_arguments(monkeypatch) -> None:
    captured = {}

    class DummyPipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(modules, "FabricTrainingPipeline", DummyPipeline)

    result = modules.create_fabric_pipeline(
        model=cast(Any, "m"),
        data_provider=cast(Any, "d"),
        training_strategy=cast(Any, "t"),
        optimizer_factory=cast(Any, "o"),
        accelerator="cpu",
        devices=1,
        strategy="auto",
        precision="32-true",
        checkpoint_dir="/tmp/ckpt",
        log_interval=3,
    )
    assert isinstance(result, DummyPipeline)
    assert captured["checkpoint_dir"] == "/tmp/ckpt"


def test_vqvae_and_maskgit_module_configure_bindings(monkeypatch) -> None:
    class DummyBinder:
        def __init__(self):
            self.calls = []

        def bind(self, key, to):
            self.calls.append((key, to))

    class DummyVQVAE:
        def __init__(self, **kwargs):
            _ = kwargs

        def parameters(self):
            return []

    monkeypatch.setattr(modules, "VQVAE", DummyVQVAE)
    monkeypatch.setattr(modules, "VQGANTrainingStrategy", lambda **kwargs: kwargs)
    monkeypatch.setattr(modules, "VQGANInference", lambda **kwargs: kwargs)
    monkeypatch.setattr(modules, "VQGANMetrics", lambda **kwargs: kwargs)
    monkeypatch.setattr(modules, "VQGANOptimizerFactory", lambda **kwargs: kwargs)

    vq_module = modules.VQVAEModule(
        model_config={"params": {}},
        training_config={"params": {}},
        optimizer_config={"params": {}},
        inference_config={"params": {}},
        metrics_config={"type": "vqgan", "params": {}},
    )
    binder = DummyBinder()
    vq_module.configure(cast(Any, binder))
    assert len(binder.calls) >= 5

    class DummyMaskModelModule:
        def provide_maskgit_model(self):
            return "maskgit-model"

    mg_module = modules.MaskGITModule(
        model_config={"params": {}},
        training_config={"params": {}},
        optimizer_config={"params": {}},
        inference_config={"params": {}},
        metrics_config={"type": "maskgit", "params": {}},
    )
    monkeypatch.setattr(mg_module, "model_module", cast(Any, DummyMaskModelModule()))
    binder2 = DummyBinder()
    mg_module.configure(cast(Any, binder2))
    assert any(key is Metrics for key, _ in binder2.calls)


def test_create_factory_validation_errors() -> None:
    with pytest.raises(ValueError, match="image_size"):
        modules.create_vqvae_module(image_size=4)

    with pytest.raises(ValueError, match="mask_ratio"):
        modules.create_maskgit_module(mask_ratio=2.0)

    with pytest.raises(ValueError, match="must be <="):
        modules._validate_float_param("x", 2.0, max_val=1.0)


def test_create_vqvae_and_maskgit_module_success_configs(monkeypatch) -> None:
    monkeypatch.setattr(modules, "get_vqvae_config", lambda **kwargs: {"cfg": kwargs})

    vq_module = modules.create_vqvae_module(
        image_size=16,
        in_channels=1,
        codebook_size=8,
        embed_dim=4,
        latent_channels=4,
        num_channels=(8,),
        num_res_blocks=(1,),
        attention_levels=(False,),
        lr=1e-3,
        batch_size=2,
    )
    assert isinstance(vq_module, modules.VQVAEModule)
    assert vq_module.model_config["type"] == "vqvae"
    assert vq_module.optimizer_config["params"]["lr"] == 1e-3
    assert vq_module.metrics_config["type"] == "vqgan"

    mg_module = modules.create_maskgit_module(
        image_size=16,
        in_channels=1,
        codebook_size=8,
        embed_dim=4,
        latent_channels=4,
        transformer_hidden=16,
        transformer_layers=2,
        transformer_heads=2,
        mask_ratio=0.5,
        lr=1e-3,
        batch_size=2,
        num_train=10,
        num_val=2,
        pretrained_vqvae_path="stage1.ckpt",
        freeze_vqvae=False,
    )
    assert isinstance(mg_module, modules.MaskGITModule)
    assert mg_module.model_config["params"]["transformer_heads"] == 2
    assert mg_module.training_config["type"] == "maskgit"
    assert mg_module.metrics_config["type"] == "maskgit"
