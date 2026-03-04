"""Unit tests for TrainingMaskScheduler."""

import math

import pytest
import torch

from maskgit3d.infrastructure.maskgit.scheduling import (
    TrainingMaskScheduler,
    mask_by_random_topk,
)


class TestTrainingMaskScheduler:
    """Tests for TrainingMaskScheduler class."""

    def test_init_default_gamma_type(self):
        scheduler = TrainingMaskScheduler()
        assert scheduler.gamma_type == "cosine"

    def test_init_custom_gamma_type(self):
        for gamma_type in ["cosine", "linear", "square", "cubic"]:
            scheduler = TrainingMaskScheduler(gamma_type=gamma_type)
            assert scheduler.gamma_type == gamma_type

    def test_init_invalid_gamma_type_raises(self):
        with pytest.raises(ValueError, match="Unknown gamma type"):
            TrainingMaskScheduler(gamma_type="invalid")

    def test_sample_mask_ratio_returns_valid_range(self):
        scheduler = TrainingMaskScheduler()
        for _ in range(100):
            ratio = scheduler.sample_mask_ratio()
            assert 0 <= ratio <= 1, f"Ratio {ratio} out of [0, 1] range"

    def test_sample_mask_ratio_cosine_distribution(self):
        """Test that cosine gamma produces expected distribution."""
        scheduler = TrainingMaskScheduler(gamma_type="cosine")
        ratios = [scheduler.sample_mask_ratio() for _ in range(1000)]
        mean_ratio = sum(ratios) / len(ratios)
        assert 0.5 < mean_ratio < 0.8, f"Cosine mean {mean_ratio} unexpected"

    def test_sample_mask_ratio_linear_distribution(self):
        """Test that linear gamma produces uniform distribution."""
        scheduler = TrainingMaskScheduler(gamma_type="linear")
        ratios = [scheduler.sample_mask_ratio() for _ in range(1000)]
        mean_ratio = sum(ratios) / len(ratios)
        assert 0.4 < mean_ratio < 0.6, f"Linear mean {mean_ratio} unexpected"

    def test_sample_mask_ratio_square_distribution(self):
        """Test that square gamma produces higher masking ratios."""
        scheduler = TrainingMaskScheduler(gamma_type="square")
        ratios = [scheduler.sample_mask_ratio() for _ in range(1000)]
        mean_ratio = sum(ratios) / len(ratios)
        assert 0.5 < mean_ratio < 0.85, f"Square mean {mean_ratio} unexpected"

    def test_sample_mask_ratio_cubic_distribution(self):
        """Test that cubic gamma produces even higher masking ratios."""
        scheduler = TrainingMaskScheduler(gamma_type="cubic")
        ratios = [scheduler.sample_mask_ratio() for _ in range(1000)]
        mean_ratio = sum(ratios) / len(ratios)
        assert 0.55 < mean_ratio < 0.9, f"Cubic mean {mean_ratio} unexpected"

    def test_compute_num_masked_returns_valid_count(self):
        scheduler = TrainingMaskScheduler()
        total_tokens = 100
        for _ in range(100):
            num_masked = scheduler.compute_num_masked(total_tokens)
            assert 1 <= num_masked < total_tokens

    def test_compute_num_masked_with_provided_ratio(self):
        scheduler = TrainingMaskScheduler()
        total_tokens = 100
        mask_ratio = 0.3
        num_masked = scheduler.compute_num_masked(total_tokens, mask_ratio)
        assert num_masked == 30

    def test_compute_num_masked_respects_minimum(self):
        scheduler = TrainingMaskScheduler()
        total_tokens = 10
        mask_ratio = 0.001
        num_masked = scheduler.compute_num_masked(total_tokens, mask_ratio)
        assert num_masked >= 1

    def test_compute_num_masked_respects_maximum(self):
        scheduler = TrainingMaskScheduler()
        total_tokens = 10
        mask_ratio = 0.999
        num_masked = scheduler.compute_num_masked(total_tokens, mask_ratio)
        assert num_masked < total_tokens

    def test_get_inference_schedule_cosine(self):
        scheduler = TrainingMaskScheduler()
        num_iterations = 12
        schedule = scheduler.get_inference_schedule(num_iterations, mode="cosine")
        assert len(schedule) == num_iterations
        assert abs(schedule.sum().item() - 1.0) < 1e-5

    def test_get_inference_schedule_linear(self):
        scheduler = TrainingMaskScheduler()
        num_iterations = 12
        schedule = scheduler.get_inference_schedule(num_iterations, mode="linear")
        assert len(schedule) == num_iterations
        expected = 1.0 / num_iterations
        for val in schedule:
            assert abs(val.item() - expected) < 1e-5

    def test_get_inference_schedule_sqrt(self):
        scheduler = TrainingMaskScheduler()
        num_iterations = 12
        schedule = scheduler.get_inference_schedule(num_iterations, mode="sqrt")
        assert len(schedule) == num_iterations
        assert abs(schedule.sum().item() - 1.0) < 1e-5

    def test_get_inference_schedule_invalid_mode_raises(self):
        scheduler = TrainingMaskScheduler()
        with pytest.raises(ValueError, match="Unknown inference schedule mode"):
            scheduler.get_inference_schedule(12, mode="invalid")

    def test_repr(self):
        scheduler = TrainingMaskScheduler(gamma_type="cosine")
        assert "cosine" in repr(scheduler)


class TestMaskByRandomTopk:
    """Tests for mask_by_random_topk function."""

    def test_returns_correct_shape(self):
        B, N = 2, 100
        confidence = torch.rand(B, N)
        topk = 30
        mask = mask_by_random_topk(topk, confidence)
        assert mask.shape == (B, N)
        assert mask.dtype == torch.bool

    def test_masks_correct_number_of_tokens(self):
        B, N = 2, 100
        confidence = torch.rand(B, N)
        topk = 30
        mask = mask_by_random_topk(topk, confidence, temperature=1.0)
        for i in range(B):
            assert mask[i].sum().item() == topk

    def test_higher_confidence_less_likely_masked(self):
        B, N = 1, 100
        confidence = torch.zeros(B, N)
        confidence[0, :50] = 1.0
        topk = 30
        mask = mask_by_random_topk(topk, confidence, temperature=0.001)
        masked_indices = mask[0].nonzero(as_tuple=True)[0]
        low_conf_masked = (masked_indices >= 50).sum().item()
        assert low_conf_masked > 20

    def test_temperature_affects_randomness(self):
        B, N = 1, 100
        confidence = torch.ones(B, N)
        topk = 50
        mask_low_temp = mask_by_random_topk(topk, confidence, temperature=0.01)
        mask_high_temp = mask_by_random_topk(topk, confidence, temperature=100.0)
        assert not torch.equal(mask_low_temp, mask_high_temp)
