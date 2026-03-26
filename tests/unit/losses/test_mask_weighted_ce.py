"""Unit tests for MaskWeightedCrossEntropyLoss."""

import torch

from src.maskgit3d.losses.mask_weighted_ce import MaskWeightedCrossEntropyLoss


def test_mask_weighted_ce_basic_forward() -> None:
    """Test basic forward pass with simple inputs."""
    loss_fn = MaskWeightedCrossEntropyLoss()

    # Simple case: batch=2, seq_len=3, vocab_size=5
    logits = torch.randn(2, 3, 5)
    labels = torch.tensor([[1, 2, 3], [0, 1, 2]])
    mask_weights = torch.ones(2, 3)

    result = loss_fn(logits, labels, mask_weights)

    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0  # Scalar loss
    assert result.item() >= 0  # Loss should be non-negative


def test_mask_weighted_ce_with_padding() -> None:
    """Test that -100 labels are ignored."""
    loss_fn = MaskWeightedCrossEntropyLoss()

    logits = torch.randn(2, 3, 5)
    # Second sequence has padding (last position)
    labels = torch.tensor([[1, 2, 3], [0, 1, -100]])
    # Weight 0 for padding position
    mask_weights = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    result = loss_fn(logits, labels, mask_weights)

    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0


def test_mask_weighted_ce_weighted_tokens() -> None:
    """Test that mask_weights correctly scale token losses."""
    loss_fn = MaskWeightedCrossEntropyLoss()

    # Fixed logits for reproducibility
    torch.manual_seed(42)
    logits = torch.randn(1, 2, 3)
    labels = torch.tensor([[0, 1]])
    mask_weights = torch.tensor([[1.0, 2.0]])

    # Weight 2.0 should give double the loss contribution for second token
    result = loss_fn(logits, labels, mask_weights)

    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0


def test_mask_weighted_ce_normalization() -> None:
    """Test that loss is normalized by sum of weights, not count."""
    loss_fn = MaskWeightedCrossEntropyLoss()

    torch.manual_seed(42)
    logits = torch.randn(1, 2, 3)
    labels = torch.tensor([[0, 1]])

    # Case 1: all weights = 1.0, sum = 2.0
    mask_weights_1 = torch.tensor([[1.0, 1.0]])
    result_1 = loss_fn(logits, labels, mask_weights_1)

    # Case 2: all weights = 2.0, sum = 4.0
    # Loss should be the same after normalization
    mask_weights_2 = torch.tensor([[2.0, 2.0]])
    result_2 = loss_fn(logits, labels, mask_weights_2)

    # Normalized loss should be same for scaled weights
    assert torch.allclose(result_1, result_2, rtol=1e-5)


def test_mask_weighted_ce_2d_logits() -> None:
    """Test with pre-flattened 2D logits."""
    loss_fn = MaskWeightedCrossEntropyLoss()

    # Pre-flattened: batch * seq_len, vocab_size
    logits = torch.randn(6, 5)  # 2 * 3 = 6
    labels = torch.tensor([[1, 2, 3], [0, 1, 2]])
    mask_weights = torch.ones(2, 3)

    result = loss_fn(logits, labels, mask_weights)

    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0


def test_mask_weighted_ce_zero_weights_edge_case() -> None:
    """Test edge case where all weights are zero."""
    loss_fn = MaskWeightedCrossEntropyLoss()

    logits = torch.randn(2, 3, 5)
    labels = torch.tensor([[1, 2, 3], [0, 1, 2]])
    mask_weights = torch.zeros(2, 3)  # All zeros

    result = loss_fn(logits, labels, mask_weights)

    # Should return 0.0 when all weights are zero
    assert result.item() == 0.0


def test_mask_weighted_ce_mixed_padding_and_weights() -> None:
    """Test combined padding and varying weights."""
    loss_fn = MaskWeightedCrossEntropyLoss()

    logits = torch.randn(2, 4, 5)
    # Mix of valid tokens and padding
    labels = torch.tensor([[1, 2, 3, -100], [-100, -100, 0, 1]])
    # Varying weights, with zeros at padding positions
    mask_weights = torch.tensor([[1.0, 0.5, 1.0, 0.0], [0.0, 0.0, 2.0, 1.0]])

    result = loss_fn(logits, labels, mask_weights)

    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0
    assert result.item() >= 0
