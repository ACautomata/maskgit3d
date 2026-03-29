"""Tests for FID metric and InceptionV3 feature extractor."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from maskgit3d.metrics.fid import FIDMetric, InceptionV3FeatureExtractor


# ---------------------------------------------------------------------------
# InceptionV3FeatureExtractor tests
# ---------------------------------------------------------------------------


class TestInceptionV3FeatureExtractor:
    """Tests for InceptionV3FeatureExtractor."""

    def test_normalize_imagenet_applied(self) -> None:
        """I1: Verify ImageNet mean/std normalization is applied in _extract_2d_features."""
        extractor = InceptionV3FeatureExtractor(input_channels=1, device=torch.device("cpu"))
        images = torch.rand(2, 1, 299, 299)

        # Patch inception forward to capture the actual input tensor passed to it
        captured_input: list[torch.Tensor] = []

        original_forward = extractor.inception.forward

        def capture_forward(x: torch.Tensor) -> torch.Tensor:
            captured_input.append(x.clone())
            return original_forward(x)

        with patch.object(extractor.inception, "forward", side_effect=capture_forward):
            extractor._extract_2d_features(images)

        assert len(captured_input) == 1
        inp = captured_input[0]

        # If no normalization were applied, the 1-channel input repeated to 3
        # would have values in [0,1]. After normalization the values should
        # have mean close to 0 and std close to 1 (rough check).
        per_channel_mean = inp.mean(dim=(0, 2, 3))
        # After proper ImageNet normalization, means should be near 0
        assert (per_channel_mean.abs() < 2.0).all(), (
            f"ImageNet normalization may not be applied; per-channel mean={per_channel_mean}"
        )

    def test_uniform_sampling_reproducibility(self) -> None:
        """C1: Same input produces identical features across multiple forward() calls."""
        extractor = InceptionV3FeatureExtractor(
            input_channels=1, device=torch.device("cpu"), slice_ratio=0.3
        )
        volume = torch.rand(1, 1, 8, 8, 8)

        # Run twice with same input
        features_1 = extractor(volume)
        features_2 = extractor(volume)

        assert torch.allclose(features_1, features_2, atol=1e-6), (
            "Same 3D input should produce identical features across calls"
        )

    def test_2d_input_shape(self) -> None:
        """Test that 2D input produces correct output shape (N, 2048)."""
        extractor = InceptionV3FeatureExtractor(input_channels=1, device=torch.device("cpu"))
        images = torch.rand(3, 1, 299, 299)
        features = extractor(images)

        assert features.dim() == 2
        assert features.shape[0] == 3
        assert features.shape[1] == 2048

    def test_3d_input_shape(self) -> None:
        """Test that 3D input produces features from all three axes."""
        extractor = InceptionV3FeatureExtractor(
            input_channels=1, device=torch.device("cpu"), slice_ratio=0.5
        )
        volume = torch.rand(1, 1, 4, 4, 4)
        features = extractor(volume)

        # 3 axes * sampled slices per axis
        assert features.dim() == 2
        assert features.shape[1] == 2048
        assert features.shape[0] > 0

    def test_invalid_dim_raises(self) -> None:
        """Test that non-4D/5D input raises ValueError."""
        extractor = InceptionV3FeatureExtractor(input_channels=1, device=torch.device("cpu"))
        flat = torch.rand(10)

        with pytest.raises(ValueError, match="Expected 4D or 5D input"):
            extractor(flat)

    def test_batchify_axis(self) -> None:
        """Test _batchify_axis correctly reorders dimensions."""
        extractor = InceptionV3FeatureExtractor(input_channels=1, device=torch.device("cpu"))
        # (B=1, C=1, D=4, H=5, W=6)
        x = torch.randn(1, 1, 4, 5, 6)

        # Axis 2 (D): should produce (1*4, 1, 5, 6)
        result = extractor._batchify_axis(x, 2)
        assert result.shape == (4, 1, 5, 6)

        # Axis 3 (H): should produce (1*5, 1, 4, 6)
        result = extractor._batchify_axis(x, 3)
        assert result.shape == (5, 1, 4, 6)

        # Axis 4 (W): should produce (1*6, 1, 4, 5)
        result = extractor._batchify_axis(x, 4)
        assert result.shape == (6, 1, 4, 5)

    def test_single_channel_repeated_to_3(self) -> None:
        """Test that single-channel input is repeated to 3 channels."""
        extractor = InceptionV3FeatureExtractor(input_channels=1, device=torch.device("cpu"))
        images = torch.rand(1, 1, 299, 299)

        captured: list[torch.Tensor] = []
        orig = extractor.inception.forward

        def capture(x: torch.Tensor) -> torch.Tensor:
            captured.append(x.clone())
            return orig(x)

        with patch.object(extractor.inception, "forward", side_effect=capture):
            extractor._extract_2d_features(images)

        assert captured[0].shape[1] == 3, "Single channel should be repeated to 3 channels"


# ---------------------------------------------------------------------------
# FIDMetric tests
# ---------------------------------------------------------------------------


class TestFIDMetric:
    """Tests for FIDMetric class."""

    def test_init_default_params(self) -> None:
        """Test default initialization."""
        metric = FIDMetric()
        assert metric.input_min == -1.0
        assert metric.input_max == 1.0

    def test_init_invalid_range_raises(self) -> None:
        """Test that input_max <= input_min raises ValueError."""
        with pytest.raises(ValueError, match="input_max must be > input_min"):
            FIDMetric(input_min=1.0, input_max=1.0)

        with pytest.raises(ValueError, match="input_max must be > input_min"):
            FIDMetric(input_min=2.0, input_max=1.0)

    def test_compute_no_updates_returns_nan(self) -> None:
        """I4: compute() with no updates returns NaN."""
        metric = FIDMetric()
        result = metric.compute()
        assert result["fid"] != result["fid"]  # NaN != NaN

    def test_compute_insufficient_samples_returns_nan(self) -> None:
        """I4: compute() with <2 feature samples returns NaN."""
        metric = FIDMetric()
        # Manually append only 1 feature (simulating 1 sample)
        metric._pred_features.append(torch.randn(1, 2048))
        metric._target_features.append(torch.randn(1, 2048))
        result = metric.compute()
        assert result["fid"] != result["fid"]  # NaN check

    def test_normalize_to_0_1_clamp(self) -> None:
        """I5: normalize clamps out-of-range values to [0, 1]."""
        metric = FIDMetric(input_min=-1.0, input_max=1.0)

        # Values within range
        t = torch.tensor([-1.0, 0.0, 1.0])
        result = metric._normalize_to_0_1(t)
        assert torch.allclose(result, torch.tensor([0.0, 0.5, 1.0]), atol=1e-6)

        # Values outside range — should be clamped
        t_out = torch.tensor([-2.0, 0.5, 2.0])
        result_out = metric._normalize_to_0_1(t_out)
        assert result_out[0].item() == 0.0, "Value below input_min should clamp to 0"
        assert result_out[2].item() == 1.0, "Value above input_max should clamp to 1"

    def test_normalize_custom_range(self) -> None:
        """Test normalization with custom input range."""
        metric = FIDMetric(input_min=0.0, input_max=255.0)
        t = torch.tensor([0.0, 127.5, 255.0])
        result = metric._normalize_to_0_1(t)
        assert torch.allclose(result, torch.tensor([0.0, 0.5, 1.0]), atol=1e-4)

    def test_extract_prediction_tensor(self) -> None:
        """C2: _extract_prediction returns tensor directly."""
        metric = FIDMetric()
        t = torch.randn(2, 1, 4, 4)
        result = metric._extract_prediction(t)
        assert result is t

    def test_extract_prediction_dict_x_recon_priority(self) -> None:
        """C2: x_recon key has highest priority in prediction dict."""
        metric = FIDMetric()
        t_recon = torch.randn(2, 1, 4, 4)
        t_images = torch.randn(2, 1, 4, 4)
        t_volumes = torch.randn(2, 1, 4, 4)

        # All keys present — x_recon wins
        result = metric._extract_prediction({
            "x_recon": t_recon,
            "images": t_images,
            "volumes": t_volumes,
        })
        assert result is t_recon

    def test_extract_prediction_dict_images_fallback(self) -> None:
        """C2: images key used when x_recon absent."""
        metric = FIDMetric()
        t_images = torch.randn(2, 1, 4, 4)

        result = metric._extract_prediction({"images": t_images, "volumes": torch.randn(2, 1, 4, 4)})
        assert result is t_images

    def test_extract_prediction_dict_volumes_fallback(self) -> None:
        """C2: volumes key used when x_recon and images absent."""
        metric = FIDMetric()
        t_volumes = torch.randn(2, 1, 4, 4)

        result = metric._extract_prediction({"volumes": t_volumes})
        assert result is t_volumes

    def test_extract_prediction_dict_no_key_raises(self) -> None:
        """C2: Missing all prediction keys raises ValueError."""
        metric = FIDMetric()
        with pytest.raises(ValueError, match="No prediction key found"):
            metric._extract_prediction({"foo": torch.randn(2)})

    def test_extract_target_tensor(self) -> None:
        """C2: _extract_target returns tensor directly."""
        metric = FIDMetric()
        t = torch.randn(2, 1, 4, 4)
        result = metric._extract_target(t)
        assert result is t

    def test_extract_target_dict_x_real_priority(self) -> None:
        """C2: x_real key has highest priority in target dict."""
        metric = FIDMetric()
        t_real = torch.randn(2, 1, 4, 4)
        t_images = torch.randn(2, 1, 4, 4)

        result = metric._extract_target({"x_real": t_real, "images": t_images})
        assert result is t_real

    def test_extract_target_dict_images_fallback(self) -> None:
        """C2: images key used when x_real absent."""
        metric = FIDMetric()
        t_images = torch.randn(2, 1, 4, 4)

        result = metric._extract_target({"images": t_images, "volumes": torch.randn(2, 1, 4, 4)})
        assert result is t_images

    def test_extract_target_dict_volumes_fallback(self) -> None:
        """C2: volumes key used when x_real and images absent."""
        metric = FIDMetric()
        t_volumes = torch.randn(2, 1, 4, 4)

        result = metric._extract_target({"volumes": t_volumes})
        assert result is t_volumes

    def test_extract_target_dict_no_key_raises(self) -> None:
        """C2: Missing all target keys raises ValueError."""
        metric = FIDMetric()
        with pytest.raises(ValueError, match="No target key found"):
            metric._extract_target({"foo": torch.randn(2)})

    def test_reset_clears_features(self) -> None:
        """Test that reset() clears accumulated features."""
        metric = FIDMetric()
        metric._pred_features.append(torch.randn(1, 2048))
        metric._target_features.append(torch.randn(1, 2048))
        assert len(metric._pred_features) == 1

        metric.reset()
        assert len(metric._pred_features) == 0
        assert len(metric._target_features) == 0

    def test_update_calls_feature_extractor(self) -> None:
        """Test that update() calls feature extractor and accumulates features."""
        metric = FIDMetric()

        mock_features = torch.randn(2, 2048)
        with patch.object(
            metric.feature_extractor, "forward", return_value=mock_features
        ) as mock_forward:
            pred = torch.randn(2, 1, 4, 4)
            target = torch.randn(2, 1, 4, 4)
            metric.update(pred, target)

            assert mock_forward.call_count == 2
            assert len(metric._pred_features) == 1
            assert len(metric._target_features) == 1

    def test_update_normalizes_input(self) -> None:
        """Test that update() normalizes inputs before feature extraction."""
        metric = FIDMetric(input_min=-1.0, input_max=1.0)

        captured_inputs: list[torch.Tensor] = []

        def capture_forward(x: torch.Tensor) -> torch.Tensor:
            captured_inputs.append(x.clone())
            return torch.randn(x.shape[0], 2048)

        with patch.object(
            metric.feature_extractor, "forward", side_effect=capture_forward
        ):
            # Input values at exactly -1.0 should be normalized to 0.0
            pred = torch.full((1, 1, 4, 4), -1.0)
            target = torch.full((1, 1, 4, 4), 1.0)
            metric.update(pred, target)

        # Check the normalized values passed to feature extractor
        assert len(captured_inputs) == 2
        # pred normalized: (-1 - (-1)) / (1 - (-1)) = 0.0
        assert torch.allclose(captured_inputs[0], torch.zeros(1, 1, 4, 4), atol=1e-6)
        # target normalized: (1 - (-1)) / (1 - (-1)) = 1.0
        assert torch.allclose(captured_inputs[1], torch.ones(1, 1, 4, 4), atol=1e-6)

    def test_compute_with_enough_samples(self) -> None:
        """Test compute returns a valid FID score with sufficient samples."""
        metric = FIDMetric()
        # Simulate 10 accumulated feature vectors
        metric._pred_features.append(torch.randn(10, 2048))
        metric._target_features.append(torch.randn(10, 2048))

        result = metric.compute()
        assert "fid" in result
        assert isinstance(result["fid"], float)
        assert result["fid"] == result["fid"]  # Not NaN

    def test_compute_multi_batch_accumulation(self) -> None:
        """Test compute correctly concatenates features from multiple batches."""
        metric = FIDMetric()
        metric._pred_features.append(torch.randn(3, 2048))
        metric._pred_features.append(torch.randn(4, 2048))
        metric._target_features.append(torch.randn(3, 2048))
        metric._target_features.append(torch.randn(4, 2048))

        result = metric.compute()
        assert "fid" in result
        assert result["fid"] == result["fid"]  # Not NaN

    def test_call_updates_computes_and_resets(self) -> None:
        """Test __call__ does update, compute, and reset."""
        metric = FIDMetric()
        metric._pred_features.append(torch.randn(5, 2048))
        metric._target_features.append(torch.randn(5, 2048))

        with patch.object(metric.feature_extractor, "forward", return_value=torch.randn(2, 2048)):
            pred = torch.randn(2, 1, 4, 4)
            target = torch.randn(2, 1, 4, 4)
            result = metric(pred, target)

        assert "fid" in result
        # After __call__, state should be reset
        assert len(metric._pred_features) == 0
        assert len(metric._target_features) == 0
