"""Tests for VQPerceptualLoss."""

import pytest
import torch

from maskgit3d.losses.vq_perceptual_loss import (
    VQPerceptualLoss,
    adopt_weight,
    hinge_d_loss,
    lsgan_g_loss,
    vanilla_d_loss,
)


class TestAdoptWeight:
    def test_adopt_weight_before_threshold(self):
        assert adopt_weight(1.0, global_step=0, threshold=10) == 0.0
        assert adopt_weight(1.0, global_step=9, threshold=10) == 0.0

    def test_adopt_weight_at_threshold(self):
        assert adopt_weight(1.0, global_step=10, threshold=10) == 1.0

    def test_adopt_weight_after_threshold(self):
        assert adopt_weight(1.0, global_step=100, threshold=10) == 1.0

    def test_adopt_weight_custom_value(self):
        assert adopt_weight(1.0, global_step=0, threshold=10, value=0.5) == 0.5


class TestHingeDLoss:
    def test_hinge_d_loss_basic(self):
        logits_real = torch.tensor([1.0, 2.0, 0.5])
        logits_fake = torch.tensor([-1.0, -2.0, -0.5])
        loss = hinge_d_loss(logits_real, logits_fake)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_hinge_d_loss_real_greater_than_one(self):
        logits_real = torch.tensor([2.0, 3.0])
        logits_fake = torch.tensor([-2.0, -3.0])
        loss = hinge_d_loss(logits_real, logits_fake)
        assert loss.item() == 0.0

    def test_hinge_d_loss_real_less_than_one(self):
        logits_real = torch.tensor([0.0])
        logits_fake = torch.tensor([0.0])
        loss = hinge_d_loss(logits_real, logits_fake)
        assert loss.item() > 0


class TestVanillaDLoss:
    def test_vanilla_d_loss_basic(self):
        logits_real = torch.tensor([1.0, 2.0])
        logits_fake = torch.tensor([-1.0, -2.0])
        loss = vanilla_d_loss(logits_real, logits_fake)
        assert loss.item() > 0
        assert torch.isfinite(loss)


class TestLSGANGLoss:
    def test_lsgan_g_loss_basic(self):
        logits_fake = torch.tensor([1.0, 1.0])
        loss = lsgan_g_loss(logits_fake)
        assert loss.item() == 0.0

    def test_lsgan_g_loss_non_target(self):
        logits_fake = torch.tensor([0.0, 0.0])
        loss = lsgan_g_loss(logits_fake)
        assert loss.item() > 0


class TestVQPerceptualLoss:
    @pytest.fixture
    def loss_fn(self):
        return VQPerceptualLoss(
            disc_in_channels=1,
            disc_num_layers=2,
            disc_norm="batch",
            lambda_l1=1.0,
            lambda_vq=1.0,
            lambda_perceptual=0.1,
            discriminator_weight=0.1,
            disc_start=0,
            use_adaptive_weight=False,
            use_perceptual=False,
        )

    @pytest.fixture
    def loss_fn_adaptive(self):
        return VQPerceptualLoss(
            disc_in_channels=1,
            disc_num_layers=2,
            disc_norm="batch",
            lambda_l1=1.0,
            lambda_vq=1.0,
            lambda_perceptual=0.1,
            discriminator_weight=0.1,
            disc_start=0,
            use_adaptive_weight=True,
            use_perceptual=False,
        )

    @pytest.fixture
    def loss_fn_warmup(self):
        return VQPerceptualLoss(
            disc_in_channels=1,
            disc_num_layers=2,
            disc_norm="batch",
            disc_start=100,
            use_adaptive_weight=True,
            use_perceptual=False,
        )

    def test_init_default_params(self):
        loss_fn = VQPerceptualLoss()
        assert loss_fn.lambda_l1 == 1.0
        assert loss_fn.lambda_vq == 1.0
        assert loss_fn.disc_start == 0
        assert loss_fn.use_adaptive_weight is True

    def test_init_invalid_disc_loss(self):
        with pytest.raises(ValueError, match="Unsupported disc_loss"):
            VQPerceptualLoss(disc_loss="invalid")

    def test_forward_generator_update(self, loss_fn):
        batch_size = 2
        inputs = torch.randn(batch_size, 1, 16, 16, 16)
        reconstructions = torch.randn(batch_size, 1, 16, 16, 16)
        vq_loss = torch.tensor(0.5)

        loss, log = loss_fn(
            inputs=inputs,
            reconstructions=reconstructions,
            vq_loss=vq_loss,
            optimizer_idx=0,
            global_step=0,
            split="train",
        )

        assert torch.isfinite(loss)
        assert "train/total_loss" in log
        assert "train/nll_loss" in log
        assert "train/g_loss" in log
        assert "train/d_weight" in log

    def test_forward_discriminator_update(self, loss_fn):
        batch_size = 2
        inputs = torch.randn(batch_size, 1, 16, 16, 16)
        reconstructions = torch.randn(batch_size, 1, 16, 16, 16)
        vq_loss = torch.tensor(0.5)

        loss, log = loss_fn(
            inputs=inputs,
            reconstructions=reconstructions,
            vq_loss=vq_loss,
            optimizer_idx=1,
            global_step=0,
            split="train",
        )

        assert torch.isfinite(loss)
        assert "train/disc_loss" in log
        assert "train/logits_real" in log
        assert "train/logits_fake" in log

    def test_forward_invalid_optimizer_idx(self, loss_fn):
        inputs = torch.randn(1, 1, 16, 16, 16)
        reconstructions = torch.randn(1, 1, 16, 16, 16)
        vq_loss = torch.tensor(0.5)

        with pytest.raises(ValueError, match="Invalid optimizer_idx"):
            loss_fn(
                inputs=inputs,
                reconstructions=reconstructions,
                vq_loss=vq_loss,
                optimizer_idx=2,
                global_step=0,
                split="train",
            )

    def test_warmup_disc_factor_zero(self, loss_fn_warmup):
        batch_size = 2
        inputs = torch.randn(batch_size, 1, 16, 16, 16)
        reconstructions = torch.randn(batch_size, 1, 16, 16, 16)
        vq_loss = torch.tensor(0.5)

        loss, log = loss_fn_warmup(
            inputs=inputs,
            reconstructions=reconstructions,
            vq_loss=vq_loss,
            optimizer_idx=0,
            global_step=0,
            split="train",
        )

        assert log["train/disc_factor"].item() == 0.0

    def test_warmup_disc_factor_after_start(self, loss_fn_warmup):
        batch_size = 2
        inputs = torch.randn(batch_size, 1, 16, 16, 16)
        reconstructions = torch.randn(batch_size, 1, 16, 16, 16)
        vq_loss = torch.tensor(0.5)

        loss, log = loss_fn_warmup(
            inputs=inputs,
            reconstructions=reconstructions,
            vq_loss=vq_loss,
            optimizer_idx=0,
            global_step=100,
            split="train",
        )

        assert log["train/disc_factor"].item() == 1.0

    def test_adaptive_weight_with_last_layer(self, loss_fn_adaptive):
        batch_size = 2
        inputs = torch.randn(batch_size, 1, 16, 16, 16, requires_grad=False)
        reconstructions = torch.randn(batch_size, 1, 16, 16, 16, requires_grad=True)
        vq_loss = torch.tensor(0.5, requires_grad=True)
        last_layer = torch.nn.Parameter(torch.randn(1, 32, 3, 3, 3))

        loss, log = loss_fn_adaptive(
            inputs=inputs,
            reconstructions=reconstructions,
            vq_loss=vq_loss,
            optimizer_idx=0,
            global_step=0,
            last_layer=last_layer,
            split="train",
        )

        assert torch.isfinite(loss)
        assert log["train/d_weight"].item() >= 0

    def test_calculate_adaptive_weight_no_last_layer(self, loss_fn_adaptive):
        nll_loss = torch.tensor(1.0, requires_grad=True)
        g_loss = torch.tensor(0.5, requires_grad=True)

        weight = loss_fn_adaptive.calculate_adaptive_weight(nll_loss, g_loss, last_layer=None)

        assert weight.item() == pytest.approx(loss_fn_adaptive.discriminator_weight)

    def test_perceptual_loss_enabled(self):
        loss_fn = VQPerceptualLoss(
            disc_in_channels=1,
            use_perceptual=True,
            perceptual_network="alex",
        )
        assert loss_fn.perceptual_loss is not None

    def test_perceptual_loss_disabled(self):
        loss_fn = VQPerceptualLoss(
            disc_in_channels=1,
            use_perceptual=False,
        )
        assert loss_fn.perceptual_loss is None

    def test_hinge_vs_vanilla_disc_loss(self):
        batch_size = 2
        inputs = torch.randn(batch_size, 1, 16, 16, 16)
        reconstructions = torch.randn(batch_size, 1, 16, 16, 16)
        vq_loss = torch.tensor(0.5)

        loss_fn_hinge = VQPerceptualLoss(
            disc_in_channels=1,
            disc_num_layers=2,
            disc_norm="batch",
            disc_loss="hinge",
            use_perceptual=False,
        )
        loss_fn_vanilla = VQPerceptualLoss(
            disc_in_channels=1,
            disc_num_layers=2,
            disc_norm="batch",
            disc_loss="vanilla",
            use_perceptual=False,
        )

        _, log_hinge = loss_fn_hinge(
            inputs=inputs,
            reconstructions=reconstructions,
            vq_loss=vq_loss,
            optimizer_idx=1,
            global_step=0,
            split="train",
        )

        _, log_vanilla = loss_fn_vanilla(
            inputs=inputs,
            reconstructions=reconstructions,
            vq_loss=vq_loss,
            optimizer_idx=1,
            global_step=0,
            split="train",
        )

        assert "train/disc_loss" in log_hinge
        assert "train/disc_loss" in log_vanilla
