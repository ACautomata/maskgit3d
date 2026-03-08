"""Tests for MedMNIST transforms."""

import warnings

import torch

from src.maskgit3d.data.medmnist.config import MedMNISTConfig, MedMNISTDatasetName
from src.maskgit3d.data.medmnist.transforms import (
    create_inference_transforms,
    create_medmnist_inference_preprocessing,
    create_medmnist_preprocessing,
    create_medmnist_training_preprocessing,
    create_training_transforms,
)


def _assert_tensor(output: object) -> torch.Tensor:
    assert isinstance(output, torch.Tensor)
    return output


class TestCreateTrainingTransforms:
    def test_returns_callable(self):
        config = MedMNISTConfig(dataset_name=MedMNISTDatasetName.ORGAN)
        transform = create_training_transforms(config)
        assert callable(transform)

    def test_transform_pipeline(self):
        """Test that transform correctly processes input."""
        config = MedMNISTConfig(
            dataset_name=MedMNISTDatasetName.ORGAN,
            crop_size=(32, 32, 32),
        )
        transform = create_training_transforms(config)

        # Input: [C, D, H, W] with values in [0, 255]
        input_tensor = torch.randint(0, 256, (1, 28, 28, 28), dtype=torch.float32)

        output = _assert_tensor(transform(input_tensor))

        # Output should be [C, D, H, W] with values in [-1, 1]
        assert output.shape == (1, 32, 32, 32)
        assert output.min() >= -1.0
        assert output.max() <= 1.0

    def test_invalid_crop_size_warns(self):
        """Test that invalid crop_size triggers warning."""
        config = MedMNISTConfig(
            dataset_name=MedMNISTDatasetName.ORGAN,
            crop_size=(28, 28, 28),  # Not divisible by 16
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_training_transforms(config)
            assert len(w) == 1
            assert "16" in str(w[0].message)


class TestCreateInferenceTransforms:
    def test_returns_callable(self):
        config = MedMNISTConfig(dataset_name=MedMNISTDatasetName.ORGAN)
        transform = create_inference_transforms(config)
        assert callable(transform)

    def test_no_cropping(self):
        """Test that inference transform does not crop."""
        config = MedMNISTConfig(
            dataset_name=MedMNISTDatasetName.ORGAN,
            crop_size=(32, 32, 32),
        )
        transform = create_inference_transforms(config)

        # Input: [C, D, H, W]
        input_tensor = torch.randint(0, 256, (1, 28, 28, 28), dtype=torch.float32)

        output = _assert_tensor(transform(input_tensor))

        # Output should keep original size (no cropping)
        assert output.shape == (1, 28, 28, 28)

    def test_normalization(self):
        """Test that values are normalized to [-1, 1]."""
        config = MedMNISTConfig(dataset_name=MedMNISTDatasetName.ORGAN)
        transform = create_inference_transforms(config)

        input_tensor = torch.full((1, 28, 28, 28), 255.0)
        output = _assert_tensor(transform(input_tensor))

        assert output.max() <= 1.0
        assert output.min() >= -1.0


class TestCreateMedMNISTPreprocessing:
    def test_returns_compose(self):
        transform = create_medmnist_preprocessing()
        assert callable(transform)

    def test_default_spatial_size(self):
        transform = create_medmnist_preprocessing()
        input_tensor = torch.rand(1, 28, 28, 28)
        output = _assert_tensor(transform(input_tensor))
        assert output.shape == (1, 64, 64, 64)

    def test_custom_spatial_size(self):
        transform = create_medmnist_preprocessing(spatial_size=(32, 32, 32))
        input_tensor = torch.rand(1, 28, 28, 28)
        output = _assert_tensor(transform(input_tensor))
        assert output.shape == (1, 32, 32, 32)

    def test_same_input_output_size(self):
        transform = create_medmnist_preprocessing(spatial_size=(28, 28, 28))
        input_tensor = torch.rand(1, 28, 28, 28)
        output = _assert_tensor(transform(input_tensor))
        assert output.shape == (1, 28, 28, 28)


class TestCreateMedMNISTTrainingPreprocessing:
    def test_returns_compose(self):
        transform = create_medmnist_training_preprocessing()
        assert callable(transform)

    def test_default_crop_size(self):
        transform = create_medmnist_training_preprocessing()
        input_tensor = torch.rand(1, 28, 28, 28)
        output = _assert_tensor(transform(input_tensor))
        assert output.shape == (1, 32, 32, 32)

    def test_custom_crop_size(self):
        transform = create_medmnist_training_preprocessing(crop_size=(16, 16, 16))
        input_tensor = torch.rand(1, 28, 28, 28)
        output = _assert_tensor(transform(input_tensor))
        assert output.shape == (1, 16, 16, 16)

    def test_warns_on_invalid_crop_size(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_medmnist_training_preprocessing(crop_size=(28, 28, 28))
            assert len(w) == 1
            assert "divisible" in str(w[0].message)


class TestCreateMedMNISTInferencePreprocessing:
    def test_returns_compose(self):
        transform = create_medmnist_inference_preprocessing()
        assert callable(transform)

    def test_no_resize(self):
        transform = create_medmnist_inference_preprocessing()
        input_tensor = torch.rand(1, 28, 28, 28)
        output = _assert_tensor(transform(input_tensor))
        assert output.shape == (1, 28, 28, 28)

    def test_normalization(self):
        transform = create_medmnist_inference_preprocessing()
        input_tensor = torch.rand(1, 28, 28, 28)
        output = _assert_tensor(transform(input_tensor))
        assert output.max() <= 1.0
        assert output.min() >= -1.0
