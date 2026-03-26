import torch
from omegaconf import OmegaConf

from maskgit3d.runtime.optimizer_factory import (
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

    # Default optimizer changed to AdamW with betas=(0.9, 0.9)
    assert isinstance(opt_g, torch.optim.AdamW)
    assert isinstance(opt_d, torch.optim.AdamW)
    assert opt_g.defaults["betas"] == (0.9, 0.9)
    assert opt_d.defaults["betas"] == (0.9, 0.9)


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


def test_transformer_optimizer_factory_calculates_total_steps() -> None:
    model = DummyTransformerModel()
    factory = TransformerOptimizerFactory(
        lr=3e-4,
        weight_decay=0.1,
        warmup_steps=10,
        max_epochs=50,
        steps_per_epoch=100,
        min_lr_ratio=0.01,
    )

    result = factory.create_optimizer_and_scheduler(model)
    scheduler = result["lr_scheduler"]["scheduler"]

    # At step 0 (warmup start), lr should be 0
    assert scheduler.get_last_lr()[0] == 0.0

    # After warmup (step 10), lr should be at peak
    for _ in range(10):
        scheduler.step()
    assert abs(scheduler.get_last_lr()[0] - 3e-4) < 1e-10

    # At end of training (total_steps = 50 * 100 = 5000), lr should decay to min_lr_ratio
    for _ in range(4990):
        scheduler.step()
    expected_min_lr = 3e-4 * 0.01
    assert abs(scheduler.get_last_lr()[0] - expected_min_lr) < 1e-10


def test_gan_optimizer_factory_creates_schedulers() -> None:
    model = DummyGANModel()
    loss_fn = DummyDiscriminatorLoss()
    factory = GANOptimizerFactory(
        lr_g=1e-4,
        lr_d=2e-4,
        warmup_steps=5,
        max_epochs=10,
        steps_per_epoch=50,
        min_lr_ratio=0.1,
    )

    opt_g, opt_d = factory.create_optimizers(generator=model, discriminator=loss_fn.discriminator)
    sched_g, sched_d = factory.create_schedulers(opt_g, opt_d)

    assert isinstance(sched_g, torch.optim.lr_scheduler.LambdaLR)
    assert isinstance(sched_d, torch.optim.lr_scheduler.LambdaLR)
    assert sched_g.base_lrs == [1e-4, 1e-4, 1e-4, 1e-4]
    assert sched_d.base_lrs == [2e-4]


def test_gan_optimizer_factory_schedulers_warmup_and_decay() -> None:
    model = DummyGANModel()
    loss_fn = DummyDiscriminatorLoss()
    factory = GANOptimizerFactory(
        lr_g=1e-4,
        lr_d=2e-4,
        warmup_steps=2,
        max_epochs=2,
        steps_per_epoch=10,
        min_lr_ratio=0.5,
    )

    opt_g, opt_d = factory.create_optimizers(generator=model, discriminator=loss_fn.discriminator)
    sched_g, sched_d = factory.create_schedulers(opt_g, opt_d)

    # At warmup step 0, lr should be 0
    assert sched_g.get_last_lr()[0] == 0.0

    # After warmup (step 2), lr should be at peak
    sched_g.step()
    sched_g.step()
    assert abs(sched_g.get_last_lr()[0] - 1e-4) < 1e-10

    # At end (total_steps = 2 * 10 = 20), lr should decay to min_lr_ratio
    for _ in range(18):
        sched_g.step()
    expected_min_lr = 1e-4 * 0.5
    assert abs(sched_g.get_last_lr()[0] - expected_min_lr) < 1e-10


def test_gan_optimizer_factory_weight_decay_defaults() -> None:
    model = DummyGANModel()
    loss_fn = DummyDiscriminatorLoss()
    factory = GANOptimizerFactory()

    opt_g, opt_d = factory.create_optimizers(generator=model, discriminator=loss_fn.discriminator)

    assert opt_g.defaults["weight_decay"] == 1e-4
    assert opt_d.defaults["weight_decay"] == 0.0


def test_gan_optimizer_factory_custom_weight_decay() -> None:
    model = DummyGANModel()
    loss_fn = DummyDiscriminatorLoss()
    factory = GANOptimizerFactory(
        lr_g=1e-4,
        lr_d=1e-4,
        weight_decay_g=0.05,
        weight_decay_d=0.01,
    )

    opt_g, opt_d = factory.create_optimizers(generator=model, discriminator=loss_fn.discriminator)

    assert opt_g.defaults["weight_decay"] == 0.05
    assert opt_d.defaults["weight_decay"] == 0.01
