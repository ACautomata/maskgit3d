import torch

from src.maskgit3d.losses.gan_loss import GANLoss


def test_gan_loss_lsgan() -> None:
    loss_fn = GANLoss(gan_mode="lsgan")

    real_pred = torch.randn(2, 1, 4, 4, 4)
    loss_real = loss_fn(real_pred, target_is_real=True)

    fake_pred = torch.randn(2, 1, 4, 4, 4)
    loss_fake = loss_fn(fake_pred, target_is_real=False)

    assert loss_real.dim() == 0
    assert loss_fake.dim() == 0


def test_gan_loss_discriminator() -> None:
    loss_fn = GANLoss(gan_mode="lsgan")

    real_pred = torch.ones(2, 1, 4, 4, 4) * 0.9
    fake_pred = torch.ones(2, 1, 4, 4, 4) * 0.1

    loss_d = loss_fn.discriminator_loss(real_pred, fake_pred)

    assert loss_d.dim() == 0


def test_gan_loss_generator() -> None:
    loss_fn = GANLoss(gan_mode="lsgan")

    fake_pred = torch.ones(2, 1, 4, 4, 4) * 0.1
    loss_g = loss_fn.generator_loss(fake_pred)

    assert loss_g.dim() == 0
