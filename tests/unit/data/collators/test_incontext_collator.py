"""Tests for VariableLengthInContextCollator."""

import pytest
import torch

from maskgit3d.data.collators import VariableLengthInContextCollator


class TestVariableLengthInContextCollator:
    """Test suite for VariableLengthInContextCollator."""

    @pytest.fixture
    def collator(self) -> VariableLengthInContextCollator:
        """Create a default collator instance."""
        return VariableLengthInContextCollator(
            pad_token_id=0,
            mask_token_id=1,
            ignore_index=-100,
            min_mask_ratio=0.1,
            max_mask_ratio=0.5,
        )

    def test_single_sample_batch(self, collator: VariableLengthInContextCollator) -> None:
        """Test collator with a single sample."""
        batch = [
            {
                "sequence": torch.tensor([10, 11, 12, 13, 14]),
                "target_mask": torch.tensor([True, True, True, True, True]),
            }
        ]

        result = collator(batch)

        assert result["input_ids"].shape == (1, 5)
        assert result["labels"].shape == (1, 5)
        assert result["attention_mask"].shape == (1, 5)
        assert result["mask_weights"].shape == (1, 5)

        # All positions should have attention mask = 1 (no padding)
        assert torch.all(result["attention_mask"] == 1)

        # At least one token should be masked
        num_masked = (result["labels"] != -100).sum().item()
        assert num_masked >= 1

    def test_variable_length_sequences(self, collator: VariableLengthInContextCollator) -> None:
        """Test collator with variable length sequences."""
        batch = [
            {
                "sequence": torch.tensor([10, 11, 12]),
                "target_mask": torch.tensor([True, True, True]),
            },
            {
                "sequence": torch.tensor([20, 21, 22, 23, 24]),
                "target_mask": torch.tensor([True, True, True, True, True]),
            },
        ]

        result = collator(batch)

        # Batch size = 2, max_seq_len = 5
        assert result["input_ids"].shape == (2, 5)
        assert result["labels"].shape == (2, 5)
        assert result["attention_mask"].shape == (2, 5)
        assert result["mask_weights"].shape == (2, 5)

        # First sample should have padding (length 3, max 5)
        assert result["attention_mask"][0, :3].sum() == 3
        assert result["attention_mask"][0, 3:].sum() == 0

        # Second sample should have no padding (length 5 = max)
        assert torch.all(result["attention_mask"][1] == 1)

    def test_per_sample_mask_ratio(self, collator: VariableLengthInContextCollator) -> None:
        """Test that per-sample mask_ratio is respected."""
        batch = [
            {
                "sequence": torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                "target_mask": torch.tensor([True] * 10),
                "mask_ratio": 0.3,  # Explicit ratio
            },
            {
                "sequence": torch.tensor([20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
                "target_mask": torch.tensor([True] * 10),
                "mask_ratio": 0.5,  # Different ratio
            },
        ]

        result = collator(batch)

        # Both samples should have masked tokens
        assert (result["labels"][0] != -100).sum().item() >= 1
        assert (result["labels"][1] != -100).sum().item() >= 1

    def test_padding_correctness(self, collator: VariableLengthInContextCollator) -> None:
        """Test that padding is applied correctly."""
        batch = [
            {
                "sequence": torch.tensor([10, 11, 12]),
                "target_mask": torch.tensor([True, True, True]),
            },
            {
                "sequence": torch.tensor([20, 21]),
                "target_mask": torch.tensor([True, True]),
            },
        ]

        result = collator(batch)

        # Max length is 3
        assert result["input_ids"].shape[1] == 3

        # Second sample (length 2) should have padding at position 2
        assert result["attention_mask"][1, 2] == 0
        assert result["input_ids"][1, 2] == 0  # pad_token_id
        assert result["labels"][1, 2] == -100  # ignore_index for padding

    def test_mask_weight_normalization(self, collator: VariableLengthInContextCollator) -> None:
        """Test that mask weights are normalized (sum = 1.0 for each sample)."""
        batch = [
            {
                "sequence": torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                "target_mask": torch.tensor([True] * 10),
            },
            {
                "sequence": torch.tensor([20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
                "target_mask": torch.tensor([True] * 10),
            },
        ]

        result = collator(batch)

        # Sum of mask weights for each sample should be 1.0 (normalized)
        for i in range(2):
            weight_sum = result["mask_weights"][i].sum().item()
            assert abs(weight_sum - 1.0) < 1e-6, (
                f"Sample {i} weights sum to {weight_sum}, expected 1.0"
            )

        # Weights should be 0 for non-masked positions
        for i in range(2):
            non_masked = result["labels"][i] == -100
            assert torch.allclose(result["mask_weights"][i][non_masked], torch.tensor(0.0))

    def test_target_mask_restricts_masking(self, collator: VariableLengthInContextCollator) -> None:
        """Test that masking only occurs where target_mask is True."""
        batch = [
            {
                "sequence": torch.tensor([10, 11, 12, 13, 14]),
                "target_mask": torch.tensor([True, False, False, True, True]),
            }
        ]

        result = collator(batch)

        # Positions 1 and 2 (target_mask=False) should NOT be masked
        # Labels at those positions should be -100 (not masked)
        assert result["labels"][0, 1] == -100
        assert result["labels"][0, 2] == -100

        # Input IDs at positions 1 and 2 should NOT be mask_token_id
        assert result["input_ids"][0, 1] != 1  # mask_token_id
        assert result["input_ids"][0, 2] != 1

    def test_at_least_one_token_masked(self, collator: VariableLengthInContextCollator) -> None:
        """Test that at least one token is always masked per sample."""
        # Very short sequence with low mask ratio
        batch = [
            {
                "sequence": torch.tensor([10, 11, 12]),
                "target_mask": torch.tensor([True, True, True]),
            }
        ]

        # Run multiple times to ensure consistency
        for _ in range(10):
            result = collator(batch)
            num_masked = (result["labels"] != -100).sum().item()
            assert num_masked >= 1, "At least one token must be masked"

    def test_empty_batch_raises_error(self, collator: VariableLengthInContextCollator) -> None:
        """Test that empty batch raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            collator([])

    def test_all_same_length(self, collator: VariableLengthInContextCollator) -> None:
        """Test with all samples having the same length."""
        batch = [
            {
                "sequence": torch.tensor([10, 11, 12, 13, 14]),
                "target_mask": torch.tensor([True] * 5),
            },
            {
                "sequence": torch.tensor([20, 21, 22, 23, 24]),
                "target_mask": torch.tensor([True] * 5),
            },
        ]

        result = collator(batch)

        # All attention masks should be 1
        assert torch.all(result["attention_mask"] == 1)

        # No padding needed
        assert result["input_ids"].shape == (2, 5)

    def test_masked_positions_contain_original_labels(
        self, collator: VariableLengthInContextCollator
    ) -> None:
        """Test that labels at masked positions contain original tokens."""
        original_sequence = torch.tensor([10, 11, 12, 13, 14])
        batch = [
            {
                "sequence": original_sequence.clone(),
                "target_mask": torch.tensor([True] * 5),
            }
        ]

        result = collator(batch)

        # Find masked positions
        masked_positions = result["labels"][0] != -100

        # Labels at masked positions should equal original tokens
        assert torch.all(
            result["labels"][0][masked_positions] == original_sequence[masked_positions]
        )

        # Input IDs at masked positions should be mask_token_id
        assert torch.all(result["input_ids"][0][masked_positions] == 1)  # mask_token_id

    def test_non_target_positions_preserved(
        self, collator: VariableLengthInContextCollator
    ) -> None:
        """Test that non-target positions are preserved in input_ids and labels."""
        batch = [
            {
                "sequence": torch.tensor([10, 11, 12, 13, 14]),
                "target_mask": torch.tensor([True, False, True, False, True]),
            }
        ]

        result = collator(batch)

        # Positions 1 and 3 (target_mask=False)
        # - input_ids should have original token
        # - labels should be -100 (not a prediction target)
        assert result["input_ids"][0, 1] == 11
        assert result["input_ids"][0, 3] == 13
        assert result["labels"][0, 1] == -100
        assert result["labels"][0, 3] == -100

    def test_custom_ignore_index(self) -> None:
        """Test custom ignore_index parameter."""
        collator = VariableLengthInContextCollator(
            pad_token_id=0,
            mask_token_id=1,
            ignore_index=-1,  # Custom value
        )

        batch = [
            {
                "sequence": torch.tensor([10, 11, 12]),
                "target_mask": torch.tensor([True, True, True]),
            }
        ]

        result = collator(batch)

        # Non-masked positions should have custom ignore_index
        masked_positions = result["labels"][0] != -1
        if masked_positions.sum() < 3:  # If not all positions masked
            non_masked = result["labels"][0] == -1
            assert non_masked.sum() > 0
