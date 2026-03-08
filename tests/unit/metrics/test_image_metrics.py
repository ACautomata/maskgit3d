"""Tests for image metrics."""

import pytest
import torch

from maskgit3d.metrics.image_metrics import ImageMetrics


class TestImageMetrics:
    """Tests for ImageMetrics class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        metrics = ImageMetrics()

        assert metrics.data_range == 1.0
        assert metrics.spatial_dims == 3
        assert metrics.input_min == -1.0
        assert metrics.input_max == 1.0

    def test_init_custom_params(self) -> None:
        """Test custom initialization parameters."""
        metrics = ImageMetrics(
            data_range=2.0,
            spatial_dims=2,
            input_min=0.0,
            input_max=1.0,
        )

        assert metrics.data_range == 2.0
        assert metrics.spatial_dims == 2
        assert metrics.input_min == 0.0
        assert metrics.input_max == 1.0

    def test_init_invalid_range(self) -> None:
        """Test that invalid input range raises error."""
        with pytest.raises(ValueError, match="input_max must be greater than input_min"):
            ImageMetrics(input_min=1.0, input_max=1.0)

        with pytest.raises(ValueError, match="input_max must be greater than input_min"):
            ImageMetrics(input_min=1.0, input_max=0.0)

    def test_update_with_tensors(self) -> None:
        """Test update with tensor inputs."""
        metrics = ImageMetrics()

        pred = torch.randn(2, 1, 16, 16, 16)
        target = torch.randn(2, 1, 16, 16, 16)

        metrics.update(pred, target)

        assert metrics._num_updates == 1

    def test_update_with_different_devices(self) -> None:
        """Test update handles different device tensors."""
        metrics = ImageMetrics()

        pred = torch.randn(2, 1, 16, 16, 16)
        target = torch.randn(2, 1, 16, 16, 16)

        # Should work without error even if devices differ
        metrics.update(pred, target)
        assert metrics._num_updates == 1

    def test_update_with_dict_input(self) -> None:
        """Test update with dictionary input."""
        metrics = ImageMetrics()

        pred_dict = {"images": torch.randn(2, 1, 16, 16, 16)}
        target_dict = {"images": torch.randn(2, 1, 16, 16, 16)}

        metrics.update(pred_dict, target_dict)

        assert metrics._num_updates == 1

    def test_update_with_volumes_key(self) -> None:
        """Test update with 'volumes' key in dict."""
        metrics = ImageMetrics()

        pred_dict = {"volumes": torch.randn(2, 1, 16, 16, 16)}
        target_dict = {"volumes": torch.randn(2, 1, 16, 16, 16)}

        metrics.update(pred_dict, target_dict)

        assert metrics._num_updates == 1

    def test_update_with_invalid_dict_key(self) -> None:
        """Test that invalid dict key raises error."""
        metrics = ImageMetrics()

        pred_dict = {"invalid": torch.randn(2, 1, 16, 16, 16)}
        target_dict = {"invalid": torch.randn(2, 1, 16, 16, 16)}

        with pytest.raises(ValueError, match="Unsupported metric input keys"):
            metrics.update(pred_dict, target_dict)

    def test_compute_without_updates(self) -> None:
        """Test compute returns zeros when no updates."""
        metrics = ImageMetrics()

        result = metrics.compute()

        assert result["psnr"] == 0.0
        assert result["ssim"] == 0.0

    def test_compute_with_updates(self) -> None:
        """Test compute after updates."""
        metrics = ImageMetrics()

        pred = torch.randn(2, 1, 16, 16, 16)
        target = pred.clone()  # Identical inputs for high metrics

        metrics.update(pred, target)
        result = metrics.compute()

        assert "psnr" in result
        assert "ssim" in result
        assert isinstance(result["psnr"], float)
        assert isinstance(result["ssim"], float)

    def test_call_method(self) -> None:
        """Test __call__ method."""
        metrics = ImageMetrics()

        pred = torch.randn(2, 1, 16, 16, 16)
        target = torch.randn(2, 1, 16, 16, 16)

        result = metrics(pred, target)

        assert "psnr" in result
        assert "ssim" in result
        assert metrics._num_updates == 1

    def test_reset(self) -> None:
        """Test reset clears state."""
        metrics = ImageMetrics()

        pred = torch.randn(2, 1, 16, 16, 16)
        target = torch.randn(2, 1, 16, 16, 16)

        metrics.update(pred, target)
        assert metrics._num_updates == 1

        metrics.reset()
        assert metrics._num_updates == 0

    def test_normalize_to_data_range(self) -> None:
        """Test normalization to data range."""
        metrics = ImageMetrics(data_range=1.0, input_min=-1.0, input_max=1.0)

        # Input in [-1, 1] should map to [0, 1]
        tensor = torch.tensor([-1.0, 0.0, 1.0])
        normalized = metrics._normalize_to_data_range(tensor)

        assert torch.allclose(normalized, torch.tensor([0.0, 0.5, 1.0]), atol=1e-5)

    def test_to_tensor_with_tensor(self) -> None:
        """Test _to_tensor with tensor input."""
        metrics = ImageMetrics()

        tensor = torch.randn(2, 1, 16, 16, 16)
        result = metrics._to_tensor(tensor)

        assert result is tensor

    def test_to_tensor_with_array(self) -> None:
        """Test _to_tensor with array input."""
        import numpy as np

        metrics = ImageMetrics()

        array = np.random.randn(2, 1, 16, 16, 16).astype(np.float32)
        result = metrics._to_tensor(array)

        assert isinstance(result, torch.Tensor)

    def test_to_scalar_with_tensor(self) -> None:
        """Test _to_scalar with tensor input."""
        metrics = ImageMetrics()

        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = metrics._to_scalar(tensor)

        assert isinstance(result, float)
        assert result == 2.0  # Mean of [1, 2, 3]

    def test_to_scalar_with_tuple(self) -> None:
        """Test _to_scalar with tuple input."""
        metrics = ImageMetrics()

        value = (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0]))
        result = metrics._to_scalar(value)

        assert isinstance(result, float)

    def test_multiple_updates_accumulate(self) -> None:
        """Test multiple updates accumulate correctly."""
        metrics = ImageMetrics()

        for _ in range(3):
            pred = torch.randn(2, 1, 16, 16, 16)
            target = torch.randn(2, 1, 16, 16, 16)
            metrics.update(pred, target)

        assert metrics._num_updates == 3

        result = metrics.compute()
        assert "psnr" in result
