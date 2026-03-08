"""Tests for MaskGIT sampling logic."""

import pytest
import torch

from src.maskgit3d.models.maskgit import (
    MaskGITSampler,
    MaskGITTransformer,
    create_mask_schedule,
)


class TestMaskGITSampler:
    """Tests for MaskGITSampler."""

    def test_init(self):
        sampler = MaskGITSampler(num_iterations=12)
        assert sampler.num_iterations == 12
        assert sampler.temperature == 1.0
        assert sampler.mask_type == "random"

    def test_custom_params(self):
        sampler = MaskGITSampler(
            num_iterations=8,
            temperature=0.5,
            mask_type="confidence",
        )
        assert sampler.num_iterations == 8
        assert sampler.temperature == 0.5
        assert sampler.mask_type == "confidence"

    def test_sample_basic(self):
        model = MaskGITTransformer(
            vocab_size=1025,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
        model.eval()

        sampler = MaskGITSampler(num_iterations=4)
        shape = (1, 2, 2, 2)

        with torch.no_grad():
            tokens = sampler.sample(model, shape, torch.device("cpu"))

        assert tokens.shape == shape
        assert (tokens >= 0).all() and (tokens < 1024).all()

    def test_sample_random_mask(self):
        model = MaskGITTransformer(
            vocab_size=1025,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
        model.eval()

        sampler = MaskGITSampler(num_iterations=4, mask_type="random")
        shape = (2, 2, 2, 2)

        with torch.no_grad():
            tokens = sampler.sample(model, shape, torch.device("cpu"))

        assert tokens.shape == shape

    def test_sample_confidence_mask(self):
        model = MaskGITTransformer(
            vocab_size=1025,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
        model.eval()

        sampler = MaskGITSampler(num_iterations=4, mask_type="confidence")
        shape = (2, 2, 2, 2)

        with torch.no_grad():
            tokens = sampler.sample(model, shape, torch.device("cpu"))

        assert tokens.shape == shape


class TestCreateMaskSchedule:
    """Tests for create_mask_schedule function."""

    def test_cosine_schedule(self):
        schedule = create_mask_schedule(12, mode="cosine")
        assert schedule.shape == (12,)
        assert abs(schedule.sum().item() - 1.0) < 1e-5

    def test_linear_schedule(self):
        schedule = create_mask_schedule(10, mode="linear")
        assert schedule.shape == (10,)
        assert abs(schedule.sum().item() - 1.0) < 1e-5

    def test_sqrt_schedule(self):
        schedule = create_mask_schedule(10, mode="sqrt")
        assert schedule.shape == (10,)
        assert abs(schedule.sum().item() - 1.0) < 1e-5

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown schedule mode"):
            create_mask_schedule(10, mode="invalid")
