"""Tests for data/transforms.py."""

import pytest
import torch
from monai.transforms.compose import Compose
from monai.transforms.intensity.array import NormalizeIntensity
from monai.transforms.spatial.array import Resize
from monai.transforms.utility.array import EnsureChannelFirst, EnsureType

from maskgit3d.data.transforms import create_3d_preprocessing, normalize_to_neg_one_one


class TestCreate3DPreprocessing:
    """Tests for create_3d_preprocessing function."""

    def test_create_3d_preprocessing_default(self) -> None:
        pipeline = create_3d_preprocessing()

        assert isinstance(pipeline, Compose)
        assert len(pipeline.transforms) == 5
        assert isinstance(pipeline.transforms[0], EnsureType)
        assert isinstance(pipeline.transforms[1], EnsureChannelFirst)
        assert isinstance(pipeline.transforms[-1], Resize)

    def test_create_3d_preprocessing_minmax(self) -> None:
        pipeline = create_3d_preprocessing(normalize_mode="minmax")

        assert isinstance(pipeline, Compose)

    def test_create_3d_preprocessing_zscore(self) -> None:
        pipeline = create_3d_preprocessing(normalize_mode="zscore")

        assert isinstance(pipeline, Compose)
        normalize_found = any(isinstance(t, NormalizeIntensity) for t in pipeline.transforms)
        assert normalize_found

    def test_create_3d_preprocessing_custom_size(self) -> None:
        pipeline = create_3d_preprocessing(spatial_size=(128, 128, 128))

        assert pipeline.transforms[-1].spatial_size == (128, 128, 128)

    def test_create_3d_preprocessing_custom_output_range(self) -> None:
        pipeline = create_3d_preprocessing(output_range=(0.0, 1.0))

        assert isinstance(pipeline, Compose)

    def test_create_3d_preprocessing_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="normalize_mode must be"):
            create_3d_preprocessing(normalize_mode="invalid")


class TestNormalizeToNegOneOne:
    """Tests for normalize_to_neg_one_one function."""

    def test_normalize_zero(self) -> None:
        result = normalize_to_neg_one_one(torch.tensor(0.0))
        assert result.item() == -1.0

    def test_normalize_one(self) -> None:
        result = normalize_to_neg_one_one(torch.tensor(1.0))
        assert result.item() == 1.0

    def test_normalize_half(self) -> None:
        result = normalize_to_neg_one_one(torch.tensor(0.5))
        assert result.item() == 0.0

    def test_normalize_tensor(self) -> None:
        x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        result = normalize_to_neg_one_one(x)

        expected = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        assert torch.allclose(result, expected)

    def test_normalize_3d_tensor(self) -> None:
        x = torch.rand(1, 16, 16, 16)
        result = normalize_to_neg_one_one(x)

        assert result.min() >= -1.0
        assert result.max() <= 1.0
