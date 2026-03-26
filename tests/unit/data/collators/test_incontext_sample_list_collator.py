"""Tests for InContextSampleListCollator."""

import pytest
import torch

from maskgit3d.data.collators import InContextSampleListCollator
from maskgit3d.models.incontext.types import InContextSample


class TestInContextSampleListCollator:
    """Test suite for InContextSampleListCollator."""

    @pytest.fixture
    def collator(self) -> InContextSampleListCollator:
        """Create a default collator instance."""
        return InContextSampleListCollator()

    def test_single_sample(self, collator: InContextSampleListCollator) -> None:
        """Test collator with a single sample."""
        sample = InContextSample(
            context_images=[torch.rand(1, 8, 8, 8)],
            context_modality_ids=[0],
            target_image=torch.rand(1, 8, 8, 8),
            target_modality_id=1,
            mask_ratio=0.5,
        )

        result = collator([sample])

        assert "samples" in result
        assert len(result["samples"]) == 1
        assert result["samples"][0] is sample

    def test_multiple_samples(self, collator: InContextSampleListCollator) -> None:
        """Test collator with multiple samples."""
        samples = [
            InContextSample(
                context_images=[torch.rand(1, 8, 8, 8)],
                context_modality_ids=[0],
                target_image=torch.rand(1, 8, 8, 8),
                target_modality_id=1,
            ),
            InContextSample(
                context_images=[torch.rand(1, 16, 16, 16), torch.rand(1, 16, 16, 16)],
                context_modality_ids=[0, 1],
                target_image=torch.rand(1, 16, 16, 16),
                target_modality_id=2,
                mask_ratio=0.3,
            ),
        ]

        result = collator(samples)

        assert "samples" in result
        assert len(result["samples"]) == 2
        assert result["samples"][0] is samples[0]
        assert result["samples"][1] is samples[1]

    def test_output_format(self, collator: InContextSampleListCollator) -> None:
        """Test that output has correct format."""
        sample = InContextSample(
            context_images=[],
            context_modality_ids=[],
            target_image=torch.rand(1, 8, 8, 8),
            target_modality_id=0,
        )

        result = collator([sample])

        assert isinstance(result, dict)
        assert "samples" in result
        assert isinstance(result["samples"], list)

    def test_empty_batch(self, collator: InContextSampleListCollator) -> None:
        """Test collator with empty batch."""
        result = collator([])

        assert "samples" in result
        assert len(result["samples"]) == 0

    def test_samples_not_modified(self, collator: InContextSampleListCollator) -> None:
        """Test that original samples are returned (not copied)."""
        sample = InContextSample(
            context_images=[torch.rand(1, 8, 8, 8)],
            context_modality_ids=[0],
            target_image=torch.rand(1, 8, 8, 8),
            target_modality_id=1,
        )

        result = collator([sample])

        # Should be the exact same object, not a copy
        assert result["samples"][0] is sample


class TestInContextSampleListCollatorAny2One:
    """Test suite for any2one batching with InContextSampleListCollator."""

    @pytest.fixture
    def collator(self) -> InContextSampleListCollator:
        """Create a default collator instance."""
        return InContextSampleListCollator()

    def test_any2one_single_context(self, collator: InContextSampleListCollator) -> None:
        """Test collator with single context modality per sample."""
        samples = [
            InContextSample(
                context_images=[torch.rand(1, 8, 8, 8)],
                context_modality_ids=[0],
                target_image=torch.rand(1, 8, 8, 8),
                target_modality_id=1,
            ),
            InContextSample(
                context_images=[torch.rand(1, 8, 8, 8)],
                context_modality_ids=[1],
                target_image=torch.rand(1, 8, 8, 8),
                target_modality_id=2,
            ),
        ]

        result = collator(samples)

        assert "samples" in result
        assert len(result["samples"]) == 2

    def test_any2one_no_context(self, collator: InContextSampleListCollator) -> None:
        """Test collator with no context (empty context lists)."""
        samples = [
            InContextSample(
                context_images=[],
                context_modality_ids=[],
                target_image=torch.rand(1, 8, 8, 8),
                target_modality_id=0,
            ),
            InContextSample(
                context_images=[],
                context_modality_ids=[],
                target_image=torch.rand(1, 8, 8, 8),
                target_modality_id=1,
            ),
        ]

        result = collator(samples)

        assert "samples" in result
        assert len(result["samples"]) == 2

    def test_any2one_variable_context_counts(self, collator: InContextSampleListCollator) -> None:
        """Test collator with variable context counts (1, 2, 3 modalities)."""
        samples = [
            InContextSample(
                context_images=[torch.rand(1, 8, 8, 8)],
                context_modality_ids=[0],
                target_image=torch.rand(1, 8, 8, 8),
                target_modality_id=1,
            ),
            InContextSample(
                context_images=[torch.rand(1, 8, 8, 8), torch.rand(1, 8, 8, 8)],
                context_modality_ids=[0, 2],
                target_image=torch.rand(1, 8, 8, 8),
                target_modality_id=1,
            ),
            InContextSample(
                context_images=[
                    torch.rand(1, 8, 8, 8),
                    torch.rand(1, 8, 8, 8),
                    torch.rand(1, 8, 8, 8),
                ],
                context_modality_ids=[0, 1, 2],
                target_image=torch.rand(1, 8, 8, 8),
                target_modality_id=3,
            ),
        ]

        result = collator(samples)

        assert "samples" in result
        assert len(result["samples"]) == 3

    def test_any2one_different_modalities_per_sample(
        self, collator: InContextSampleListCollator
    ) -> None:
        """Test collator with different context modalities per sample."""
        samples = [
            InContextSample(
                context_images=[torch.rand(1, 8, 8, 8)],
                context_modality_ids=[0],
                target_image=torch.rand(1, 8, 8, 8),
                target_modality_id=2,
            ),
            InContextSample(
                context_images=[torch.rand(1, 8, 8, 8)],
                context_modality_ids=[1],
                target_image=torch.rand(1, 8, 8, 8),
                target_modality_id=3,
            ),
            InContextSample(
                context_images=[torch.rand(1, 8, 8, 8), torch.rand(1, 8, 8, 8)],
                context_modality_ids=[2, 3],
                target_image=torch.rand(1, 8, 8, 8),
                target_modality_id=0,
            ),
        ]

        result = collator(samples)

        assert "samples" in result
        assert len(result["samples"]) == 3
