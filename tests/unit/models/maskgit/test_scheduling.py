"""Tests for MaskGIT training mask scheduler."""


import pytest
import torch

from maskgit3d.models.maskgit.scheduling import (
    TrainingMaskScheduler,
    mask_by_random_topk,
)


class TestTrainingMaskScheduler:
    """Tests for TrainingMaskScheduler."""

    def test_init_default(self):
        scheduler = TrainingMaskScheduler()
        assert scheduler.gamma_type == "cosine"

    def test_init_custom_gamma(self):
        scheduler = TrainingMaskScheduler(gamma_type="linear")
        assert scheduler.gamma_type == "linear"

    def test_invalid_gamma_type(self):
        with pytest.raises(ValueError, match="Unknown gamma type"):
            TrainingMaskScheduler(gamma_type="invalid")

    def test_sample_mask_ratio_cosine(self):
        scheduler = TrainingMaskScheduler(gamma_type="cosine")
        ratios = [scheduler.sample_mask_ratio() for _ in range(100)]
        assert all(0 <= r <= 1 for r in ratios)

    def test_sample_mask_ratio_linear(self):
        scheduler = TrainingMaskScheduler(gamma_type="linear")
        ratios = [scheduler.sample_mask_ratio() for _ in range(100)]
        assert all(0 <= r <= 1 for r in ratios)

    def test_compute_num_masked(self):
        scheduler = TrainingMaskScheduler()
        total_tokens = 100
        for _ in range(10):
            num_masked = scheduler.compute_num_masked(total_tokens)
            assert 1 <= num_masked < total_tokens

    def test_compute_num_masked_with_ratio(self):
        scheduler = TrainingMaskScheduler()
        total_tokens = 100
        num_masked = scheduler.compute_num_masked(total_tokens, mask_ratio=0.5)
        assert num_masked == 50

    def test_get_inference_schedule_cosine(self):
        scheduler = TrainingMaskScheduler()
        schedule = scheduler.get_inference_schedule(12, mode="cosine")
        assert schedule.shape == (12,)
        assert abs(schedule.sum().item() - 1.0) < 1e-5

    def test_get_inference_schedule_linear(self):
        scheduler = TrainingMaskScheduler()
        schedule = scheduler.get_inference_schedule(10, mode="linear")
        assert schedule.shape == (10,)
        assert abs(schedule.sum().item() - 1.0) < 1e-5

    def test_get_inference_schedule_sqrt(self):
        scheduler = TrainingMaskScheduler()
        schedule = scheduler.get_inference_schedule(10, mode="sqrt")
        assert schedule.shape == (10,)
        assert abs(schedule.sum().item() - 1.0) < 1e-5

    def test_invalid_inference_mode(self):
        scheduler = TrainingMaskScheduler()
        with pytest.raises(ValueError, match="Unknown inference schedule mode"):
            scheduler.get_inference_schedule(10, mode="invalid")


class TestMaskByRandomTopk:
    """Tests for mask_by_random_topk function."""

    def test_basic(self):
        batch_size = 2
        num_tokens = 100
        topk = 30

        confidence = torch.rand(batch_size, num_tokens)
        mask = mask_by_random_topk(topk, confidence)

        assert mask.shape == (batch_size, num_tokens)
        assert mask.sum().item() == batch_size * topk

    def test_temperature_effect(self):
        batch_size = 2
        num_tokens = 100
        topk = 30

        confidence = torch.rand(batch_size, num_tokens)

        mask_low_temp = mask_by_random_topk(topk, confidence, temperature=1.0)
        mask_high_temp = mask_by_random_topk(topk, confidence, temperature=10.0)

        assert not torch.equal(mask_low_temp, mask_high_temp)

    def test_extreme_confidence(self):
        batch_size = 2
        num_tokens = 100
        topk = 30

        confidence = torch.zeros(batch_size, num_tokens)
        confidence[:, :topk] = 1.0

        mask = mask_by_random_topk(topk, confidence, temperature=0.1)

        assert mask.sum().item() == batch_size * topk
