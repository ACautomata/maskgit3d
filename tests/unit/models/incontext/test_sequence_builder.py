"""Unit tests for InContextSequenceBuilder."""

from __future__ import annotations

import pytest
import torch

from maskgit3d.models.incontext.sequence_builder import InContextSequenceBuilder


@pytest.fixture
def builder() -> InContextSequenceBuilder:
    """Create a default InContextSequenceBuilder for testing."""
    return InContextSequenceBuilder(
        num_modalities=4,
        latent_spatial_size=(2, 2, 2),
        vocab_size=256,
    )


@pytest.fixture
def builder_with_custom_tokens() -> InContextSequenceBuilder:
    """Create an InContextSequenceBuilder with custom special token IDs."""
    return InContextSequenceBuilder(
        num_modalities=4,
        latent_spatial_size=(2, 2, 2),
        vocab_size=256,
        cls_token_id=300,
        sep_token_id=301,
    )


class TestInContextSequenceBuilderInit:
    """Tests for InContextSequenceBuilder initialization."""

    def test_init_creates_default_special_tokens(self) -> None:
        """Test that default CLS and SEP tokens are created."""
        builder = InContextSequenceBuilder(
            num_modalities=4,
            latent_spatial_size=(2, 2, 2),
            vocab_size=256,
        )
        assert builder.cls_token_id == 256  # vocab_size
        assert builder.sep_token_id == 257  # vocab_size + 1

    def test_init_with_custom_special_tokens(self) -> None:
        """Test that custom CLS and SEP tokens are used."""
        builder = InContextSequenceBuilder(
            num_modalities=4,
            latent_spatial_size=(2, 2, 2),
            vocab_size=256,
            cls_token_id=300,
            sep_token_id=301,
        )
        assert builder.cls_token_id == 300
        assert builder.sep_token_id == 301

    def test_init_stores_parameters(self) -> None:
        """Test that initialization stores parameters correctly."""
        builder = InContextSequenceBuilder(
            num_modalities=8,
            latent_spatial_size=(4, 4, 4),
            vocab_size=1024,
        )
        assert builder.num_modalities == 8
        assert builder.latent_spatial_size == (4, 4, 4)
        assert builder.vocab_size == 1024


class TestSpecialTokenIDs:
    """Tests for special token ID allocation."""

    def test_cls_token_id(self, builder: InContextSequenceBuilder) -> None:
        """Test CLS token ID equals vocab_size."""
        assert builder.cls_token_id == 256

    def test_sep_token_id(self, builder: InContextSequenceBuilder) -> None:
        """Test SEP token ID equals vocab_size + 1."""
        assert builder.sep_token_id == 257

    def test_modality_label_ids(self, builder: InContextSequenceBuilder) -> None:
        """Test modality label IDs start at vocab_size + 2."""
        # Modality 0 -> vocab_size + 2
        # Modality 1 -> vocab_size + 3
        # etc.
        expected_modality_base = 256 + 2  # vocab_size + 2
        for mod_id in range(builder.num_modalities):
            expected_id = expected_modality_base + mod_id
            assert builder.get_modality_label_id(mod_id) == expected_id


class TestSequenceConstruction:
    """Tests for sequence building."""

    def test_build_with_single_context(self, builder: InContextSequenceBuilder) -> None:
        """Test sequence construction with a single context latent."""
        batch_size = 2
        context_latents = [torch.randint(0, 256, (batch_size, 2, 2, 2))]
        target_latent = torch.randint(0, 256, (batch_size, 2, 2, 2))
        context_modality_ids = [0]
        target_modality_id = 1

        sequence, target_mask, attention_mask = builder.build(
            context_latents=context_latents,
            target_latent=target_latent,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
        )

        # Expected sequence length:
        # 1 (CLS) + 1 (MOD_0) + 8 (latent_0) + 1 (MOD_TARGET) + 8 (target) + 1 (SEP) = 20
        expected_seq_len = 1 + 1 + 8 + 1 + 8 + 1
        assert sequence.shape == (batch_size, expected_seq_len)
        assert target_mask.shape == (batch_size, expected_seq_len)
        assert attention_mask.shape == (batch_size, expected_seq_len)

    def test_build_with_multiple_contexts(self, builder: InContextSequenceBuilder) -> None:
        """Test sequence construction with multiple context latents."""
        batch_size = 2
        context_latents = [
            torch.randint(0, 256, (batch_size, 2, 2, 2)),
            torch.randint(0, 256, (batch_size, 2, 2, 2)),
        ]
        target_latent = torch.randint(0, 256, (batch_size, 2, 2, 2))
        context_modality_ids = [0, 2]
        target_modality_id = 1

        sequence, target_mask, attention_mask = builder.build(
            context_latents=context_latents,
            target_latent=target_latent,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
        )

        # Expected sequence length:
        # 1 (CLS) + 1 (MOD_0) + 8 (latent_0) + 1 (MOD_1) + 8 (latent_1)
        # + 1 (MOD_TARGET) + 8 (target) + 1 (SEP) = 29
        expected_seq_len = 1 + (1 + 8) * 2 + 1 + 8 + 1
        assert sequence.shape == (batch_size, expected_seq_len)

    def test_build_flattens_latents_correctly(self, builder: InContextSequenceBuilder) -> None:
        """Test that 3D latents are flattened to 1D in row-major order."""
        batch_size = 1
        # Create deterministic latent values
        context_latents = [torch.arange(8).reshape(1, 2, 2, 2).expand(batch_size, -1, -1, -1)]
        target_latent = torch.arange(8, 16).reshape(1, 2, 2, 2).expand(batch_size, -1, -1, -1)
        context_modality_ids = [0]
        target_modality_id = 1

        sequence, _, _ = builder.build(
            context_latents=context_latents,
            target_latent=target_latent,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
        )

        # Sequence should be:
        # [CLS, MOD_0, 0, 1, 2, 3, 4, 5, 6, 7, MOD_TARGET, 8, 9, 10, 11, 12, 13, 14, 15, SEP]
        # Positions 2-9 should be context latent (0-7)
        # Positions 11-18 should be target latent (8-15)
        assert sequence[0, 0].item() == builder.cls_token_id
        assert sequence[0, 1].item() == builder.get_modality_label_id(0)
        # Context latent tokens (flattened)
        expected_context = torch.arange(8)
        assert torch.equal(sequence[0, 2:10], expected_context)
        # Target modality label
        assert sequence[0, 10].item() == builder.get_modality_label_id(1)
        # Target latent tokens (flattened)
        expected_target = torch.arange(8, 16)
        assert torch.equal(sequence[0, 11:19], expected_target)
        # SEP token
        assert sequence[0, 19].item() == builder.sep_token_id


class TestTargetMask:
    """Tests for target mask correctness."""

    def test_target_mask_single_context(self, builder: InContextSequenceBuilder) -> None:
        """Test target mask with single context."""
        batch_size = 2
        context_latents = [torch.randint(0, 256, (batch_size, 2, 2, 2))]
        target_latent = torch.randint(0, 256, (batch_size, 2, 2, 2))
        context_modality_ids = [0]
        target_modality_id = 1

        _, target_mask, _ = builder.build(
            context_latents=context_latents,
            target_latent=target_latent,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
        )

        # Target mask should be:
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        #  CLS MOD  --context--  MOD_T ----target----  SEP

        # CLS, MOD_0, context latent, MOD_TARGET should be 0
        assert target_mask[0, 0].item() == 0  # CLS
        assert target_mask[0, 1].item() == 0  # MOD_0
        assert target_mask[0, 2:10].sum().item() == 0  # context latent
        assert target_mask[0, 10].item() == 0  # MOD_TARGET

        # Target latent should be 1
        assert target_mask[0, 11:19].sum().item() == 8  # all 1s

        # SEP should be 0
        assert target_mask[0, 19].item() == 0  # SEP

    def test_target_mask_multiple_contexts(self, builder: InContextSequenceBuilder) -> None:
        """Test target mask with multiple contexts."""
        batch_size = 1
        context_latents = [
            torch.randint(0, 256, (batch_size, 2, 2, 2)),
            torch.randint(0, 256, (batch_size, 2, 2, 2)),
        ]
        target_latent = torch.randint(0, 256, (batch_size, 2, 2, 2))
        context_modality_ids = [0, 2]
        target_modality_id = 1

        _, target_mask, _ = builder.build(
            context_latents=context_latents,
            target_latent=target_latent,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
        )

        # Count total tokens and check only target positions are 1
        latent_size = 2 * 2 * 2  # 8

        # Should have exactly 8 ones (the target latent)
        assert target_mask[0].sum().item() == latent_size

        # All zeros except target positions
        # With 2 contexts:
        # 0: CLS, 1: MOD_0, 2-9: ctx_0, 10: MOD_1, 11-18: ctx_1, 19: MOD_T, 20-27: target, 28: SEP
        assert target_mask[0, :20].sum().item() == 0  # Before target (CLS, MODs, contexts)
        assert target_mask[0, 20:28].sum().item() == 8  # Target latent
        assert target_mask[0, 28:].sum().item() == 0  # After target (SEP)

    def test_target_mask_is_bool(self, builder: InContextSequenceBuilder) -> None:
        """Test that target mask is boolean tensor."""
        batch_size = 1
        context_latents = [torch.randint(0, 256, (batch_size, 2, 2, 2))]
        target_latent = torch.randint(0, 256, (batch_size, 2, 2, 2))

        _, target_mask, _ = builder.build(
            context_latents=context_latents,
            target_latent=target_latent,
            context_modality_ids=[0],
            target_modality_id=1,
        )

        assert target_mask.dtype == torch.bool


class TestAttentionMask:
    """Tests for attention mask."""

    def test_attention_mask_all_ones(self, builder: InContextSequenceBuilder) -> None:
        """Test that attention mask is all 1s (no padding)."""
        batch_size = 2
        context_latents = [torch.randint(0, 256, (batch_size, 2, 2, 2))]
        target_latent = torch.randint(0, 256, (batch_size, 2, 2, 2))

        _, _, attention_mask = builder.build(
            context_latents=context_latents,
            target_latent=target_latent,
            context_modality_ids=[0],
            target_modality_id=1,
        )

        assert attention_mask.dtype == torch.float
        assert (attention_mask == 1.0).all()


class TestDifferentSpatialSizes:
    """Tests for different latent spatial sizes."""

    def test_build_with_larger_spatial_size(self) -> None:
        """Test sequence construction with larger spatial size."""
        builder = InContextSequenceBuilder(
            num_modalities=4,
            latent_spatial_size=(4, 4, 4),
            vocab_size=256,
        )
        batch_size = 1
        context_latents = [torch.randint(0, 256, (batch_size, 4, 4, 4))]
        target_latent = torch.randint(0, 256, (batch_size, 4, 4, 4))

        sequence, target_mask, attention_mask = builder.build(
            context_latents=context_latents,
            target_latent=target_latent,
            context_modality_ids=[0],
            target_modality_id=1,
        )

        # Expected: 1 + 1 + 64 + 1 + 64 + 1 = 132
        expected_seq_len = 1 + 1 + 64 + 1 + 64 + 1
        assert sequence.shape == (batch_size, expected_seq_len)

        # Target mask should have 64 ones
        assert target_mask[0].sum().item() == 64

    def test_build_with_asymmetric_spatial_size(self) -> None:
        """Test sequence construction with asymmetric spatial size."""
        builder = InContextSequenceBuilder(
            num_modalities=4,
            latent_spatial_size=(2, 4, 4),
            vocab_size=256,
        )
        batch_size = 1
        context_latents = [torch.randint(0, 256, (batch_size, 2, 4, 4))]
        target_latent = torch.randint(0, 256, (batch_size, 2, 4, 4))

        sequence, target_mask, _ = builder.build(
            context_latents=context_latents,
            target_latent=target_latent,
            context_modality_ids=[0],
            target_modality_id=1,
        )

        # Expected: 1 + 1 + 32 + 1 + 32 + 1 = 68
        expected_seq_len = 1 + 1 + 32 + 1 + 32 + 1
        assert sequence.shape == (batch_size, expected_seq_len)


class TestModalityLabels:
    """Tests for modality label IDs in sequences."""

    def test_correct_modality_labels_in_sequence(self, builder: InContextSequenceBuilder) -> None:
        """Test that correct modality label IDs appear in sequence."""
        batch_size = 1
        context_latents = [
            torch.randint(0, 256, (batch_size, 2, 2, 2)),
            torch.randint(0, 256, (batch_size, 2, 2, 2)),
        ]
        target_latent = torch.randint(0, 256, (batch_size, 2, 2, 2))
        context_modality_ids = [0, 3]
        target_modality_id = 2

        sequence, _, _ = builder.build(
            context_latents=context_latents,
            target_latent=target_latent,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
        )

        # Check context modality labels
        # Position 1: MOD_0 (modality 0)
        assert sequence[0, 1].item() == builder.get_modality_label_id(0)
        # Position 10: MOD_1 (modality 3)
        assert sequence[0, 10].item() == builder.get_modality_label_id(3)
        # Position 19: MOD_TARGET (modality 2)
        assert sequence[0, 19].item() == builder.get_modality_label_id(2)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_build_with_no_context(self, builder: InContextSequenceBuilder) -> None:
        """Test sequence construction with no context latents."""
        batch_size = 1
        context_latents: list[torch.Tensor] = []
        target_latent = torch.randint(0, 256, (batch_size, 2, 2, 2))

        sequence, target_mask, _ = builder.build(
            context_latents=context_latents,
            target_latent=target_latent,
            context_modality_ids=[],
            target_modality_id=0,
        )

        # Expected: CLS + MOD_TARGET + target + SEP = 1 + 1 + 8 + 1 = 11
        expected_seq_len = 1 + 1 + 8 + 1
        assert sequence.shape == (batch_size, expected_seq_len)

    def test_build_batch_size_one(self, builder: InContextSequenceBuilder) -> None:
        """Test sequence construction with batch size 1."""
        batch_size = 1
        context_latents = [torch.randint(0, 256, (batch_size, 2, 2, 2))]
        target_latent = torch.randint(0, 256, (batch_size, 2, 2, 2))

        sequence, target_mask, attention_mask = builder.build(
            context_latents=context_latents,
            target_latent=target_latent,
            context_modality_ids=[0],
            target_modality_id=1,
        )

        assert sequence.shape[0] == 1
        assert target_mask.shape[0] == 1
        assert attention_mask.shape[0] == 1

    def test_build_larger_batch_size(self, builder: InContextSequenceBuilder) -> None:
        """Test sequence construction with larger batch size."""
        batch_size = 8
        context_latents = [torch.randint(0, 256, (batch_size, 2, 2, 2))]
        target_latent = torch.randint(0, 256, (batch_size, 2, 2, 2))

        sequence, target_mask, attention_mask = builder.build(
            context_latents=context_latents,
            target_latent=target_latent,
            context_modality_ids=[0],
            target_modality_id=1,
        )

        assert sequence.shape[0] == batch_size
        assert target_mask.shape[0] == batch_size
        assert attention_mask.shape[0] == batch_size
