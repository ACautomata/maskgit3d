import torch
from omegaconf import OmegaConf

from src.maskgit3d.runtime.optimizer_factory import (
    GANOptimizerFactory,
    TransformerOptimizerFactory,
    create_optimizer,
)


class DummyGANModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.quant_conv = torch.nn.Linear(4, 4)
        self.post_quant_conv = torch.nn.Linear(4, 4)
        self.decoder = torch.nn.Linear(4, 4)


class DummyDiscriminatorLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discriminator = torch.nn.Linear(4, 1)


class DummyTransformerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = torch.nn.Linear(4, 4)
        self.other = torch.nn.Linear(4, 4)


def test_create_optimizer_keeps_param_group_learning_rates() -> None:
    layer = torch.nn.Linear(4, 2)
    config = OmegaConf.create(
        {
            "_target_": "torch.optim.Adam",
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0,
        }
    )

    optimizer = create_optimizer(
        [
            {"params": layer.weight, "lr": 2e-4},
            {"params": layer.bias, "lr": 5e-4},
        ],
        config,
    )

    assert isinstance(optimizer, torch.optim.Adam)
    assert [group["lr"] for group in optimizer.param_groups] == [2e-4, 5e-4]


def test_gan_optimizer_factory_creates_generator_and_discriminator_optimizers() -> None:
    model = DummyGANModel()
    loss_fn = DummyDiscriminatorLoss()

    factory = GANOptimizerFactory()

    opt_g, opt_d = factory.create_optimizers(generator=model, discriminator=loss_fn.discriminator)

    assert isinstance(opt_g, torch.optim.Adam)
    assert isinstance(opt_d, torch.optim.Adam)


def test_gan_optimizer_factory_keeps_legacy_generator_param_groups() -> None:
    model = DummyGANModel()
    loss_fn = DummyDiscriminatorLoss()
    factory = GANOptimizerFactory(lr_g=2e-4, lr_d=5e-4)

    opt_g, _ = factory.create_optimizers(generator=model, discriminator=loss_fn.discriminator)

    expected_groups = [
        list(model.encoder.parameters()),
        list(model.quant_conv.parameters()),
        list(model.post_quant_conv.parameters()),
        list(model.decoder.parameters()),
    ]

    assert [group["lr"] for group in opt_g.param_groups] == [2e-4, 2e-4, 2e-4, 2e-4]
    assert [list(group["params"]) for group in opt_g.param_groups] == expected_groups


def test_transformer_optimizer_factory_creates_lightning_optimizer_dict() -> None:
    model = DummyTransformerModel()

    factory = TransformerOptimizerFactory(lr=3e-4, weight_decay=0.1, warmup_steps=4)

    result = factory.create_optimizer_and_scheduler(model)

    assert set(result) == {"optimizer", "lr_scheduler"}
    assert isinstance(result["optimizer"], torch.optim.AdamW)
    assert result["lr_scheduler"]["interval"] == "step"


def test_transformer_optimizer_factory_uses_transformer_params_and_warmup_scheduler() -> None:
    model = DummyTransformerModel()
    factory = TransformerOptimizerFactory(lr=3e-4, weight_decay=0.1, warmup_steps=4)

    result = factory.create_optimizer_and_scheduler(model)
    optimizer = result["optimizer"]
    scheduler = result["lr_scheduler"]["scheduler"]

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert list(optimizer.param_groups[0]["params"]) == list(model.transformer.parameters())
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert scheduler.base_lrs == [3e-4]
