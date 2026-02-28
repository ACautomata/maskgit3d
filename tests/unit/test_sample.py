"""
Sample unit tests for the maskgit3d framework.

These tests verify the basic functionality of the core components.
"""
import pytest
import torch
from unittest.mock import MagicMock, patch

from maskgit3d.domain.interfaces import (
    DataProvider,
    InferenceStrategy,
    Metrics,
    ModelInterface,
    OptimizerFactory,
    TrainingStrategy,
)


class TestModelInterface:
    """Tests for ModelInterface."""

    def test_model_interface_is_abstract(self):
        """Verify ModelInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ModelInterface()

    def test_model_interface_declares_module_runtime_methods(self):
        """ModelInterface must declare nn.Module runtime methods used by pipelines."""
        required = ["to", "parameters", "train", "eval", "state_dict", "load_state_dict"]
        for name in required:
            assert hasattr(ModelInterface, name), f"ModelInterface missing {name}"


class TestMockModel:
    """Tests using a mock model."""

    def test_mock_model_forward(self):
        """Test forward pass with mock model."""
        mock_model = MagicMock(spec=ModelInterface)
        mock_model.forward.return_value = torch.randn(1, 10, 32, 32, 32)

        input_tensor = torch.randn(1, 1, 32, 32, 32)
        output = mock_model.forward(input_tensor)

        assert output.shape == (1, 10, 32, 32, 32)
        mock_model.forward.assert_called_once_with(input_tensor)

    def test_mock_model_checkpoint(self):
        """Test checkpoint save/load with mock model."""
        mock_model = MagicMock(spec=ModelInterface)

        # Test save
        mock_model.save_checkpoint("/tmp/test_model.pth")
        mock_model.save_checkpoint.assert_called_once()

        # Test load
        mock_model.load_checkpoint("/tmp/test_model.pth")
        mock_model.load_checkpoint.assert_called_once()


class TestTrainingStrategy:
    """Tests for TrainingStrategy."""

    def test_training_strategy_is_abstract(self):
        """Verify TrainingStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TrainingStrategy()

    def test_mock_training_step(self):
        """Test training step with mock strategy."""
        mock_strategy = MagicMock(spec=TrainingStrategy)
        mock_model = MagicMock(spec=ModelInterface)
        mock_optimizer = MagicMock()

        batch = (
            torch.randn(2, 1, 32, 32, 32),
            torch.randint(0, 2, (2, 1, 32, 32, 32)),
        )

        expected_metrics = {
            "loss": 0.5,
            "dice_score": 0.8,
            "ce_loss": 0.3,
        }

        mock_strategy.train_step.return_value = expected_metrics

        result = mock_strategy.train_step(mock_model, batch, mock_optimizer)

        assert result == expected_metrics
        mock_strategy.train_step.assert_called_once_with(
            mock_model, batch, mock_optimizer
        )

    def test_mock_validation_step(self):
        """Test validation step with mock strategy."""
        mock_strategy = MagicMock(spec=TrainingStrategy)
        mock_model = MagicMock(spec=ModelInterface)

        batch = (
            torch.randn(2, 1, 32, 32, 32),
            torch.randint(0, 2, (2, 1, 32, 32, 32)),
        )

        expected_metrics = {
            "val_loss": 0.4,
            "val_dice_score": 0.85,
        }

        mock_strategy.validate_step.return_value = expected_metrics

        result = mock_strategy.validate_step(mock_model, batch)

        assert result == expected_metrics
        mock_strategy.validate_step.assert_called_once_with(mock_model, batch)


class TestDataProvider:
    """Tests for DataProvider."""

    def test_data_provider_is_abstract(self):
        """Verify DataProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataProvider()

    def test_mock_data_provider(self):
        """Test data provider with mocks."""
        mock_provider = MagicMock(spec=DataProvider)

        # Create mock loaders
        train_loader = [
            (torch.randn(2, 1, 32, 32, 32), torch.randint(0, 2, (2, 1, 32, 32, 32)))
            for _ in range(3)
        ]
        val_loader = [
            (torch.randn(2, 1, 32, 32, 32), torch.randint(0, 2, (2, 1, 32, 32, 32)))
            for _ in range(2)
        ]

        mock_provider.train_loader.return_value = iter(train_loader)
        mock_provider.val_loader.return_value = iter(val_loader)
        mock_provider.test_loader.return_value = iter([])

        # Test train loader
        train_iter = mock_provider.train_loader()
        batch = next(train_iter)
        assert batch[0].shape == (2, 1, 32, 32, 32)
        assert batch[1].shape == (2, 1, 32, 32, 32)

        # Test val loader
        val_iter = mock_provider.val_loader()
        batch = next(val_iter)
        assert batch[0].shape == (2, 1, 32, 32, 32)


class TestInferenceStrategy:
    """Tests for InferenceStrategy."""

    def test_inference_strategy_is_abstract(self):
        """Verify InferenceStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            InferenceStrategy()

    def test_mock_inference(self):
        """Test inference with mock strategy."""
        mock_strategy = MagicMock(spec=InferenceStrategy)
        mock_model = MagicMock(spec=ModelInterface)

        # Mock raw predictions
        mock_model.return_value = torch.randn(2, 2, 32, 32, 32)
        raw_preds = torch.randn(2, 2, 32, 32, 32)

        # Mock processed predictions
        expected_processed = {
            "masks": torch.randint(0, 2, (2, 1, 32, 32, 32)).numpy(),
            "probs": torch.softmax(torch.randn(2, 2, 32, 32, 32), dim=1).numpy(),
        }

        mock_strategy.predict.return_value = raw_preds
        mock_strategy.post_process.return_value = expected_processed

        # Test predict
        batch = torch.randn(2, 1, 32, 32, 32)
        result = mock_strategy.predict(mock_model, batch)

        assert result.shape == (2, 2, 32, 32, 32)
        mock_strategy.predict.assert_called_once_with(mock_model, batch)

        # Test post_process
        processed = mock_strategy.post_process(result)
        assert "masks" in processed
        assert "probs" in processed


class TestMetrics:
    """Tests for Metrics."""

    def test_metrics_is_abstract(self):
        """Verify Metrics cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Metrics()

    def test_mock_metrics(self):
        """Test metrics with mocks."""
        mock_metrics = MagicMock(spec=Metrics)

        # Test update
        predictions = torch.randint(0, 2, (2, 1, 32, 32, 32))
        targets = torch.randint(0, 2, (2, 1, 32, 32, 32))

        mock_metrics.update(predictions, targets)
        mock_metrics.update.assert_called_once_with(predictions, targets)

        # Test compute
        expected_result = {"dice_score": 0.85, "iou": 0.75}
        mock_metrics.compute.return_value = expected_result

        result = mock_metrics.compute()
        assert result == expected_result
        mock_metrics.compute.assert_called_once()

        # Test reset
        mock_metrics.reset()
        mock_metrics.reset.assert_called_once()


class TestOptimizerFactory:
    """Tests for OptimizerFactory."""

    def test_optimizer_factory_is_abstract(self):
        """Verify OptimizerFactory cannot be instantiated directly."""
        with pytest.raises(TypeError):
            OptimizerFactory()

    def test_mock_optimizer_creation(self):
        """Test optimizer creation with mock factory."""
        mock_factory = MagicMock(spec=OptimizerFactory)
        mock_optimizer = MagicMock()

        # Create mock parameters
        params = [torch.randn(10, 10) for _ in range(3)]

        mock_factory.create.return_value = mock_optimizer

        result = mock_factory.create(iter(params))

        assert result == mock_optimizer
        mock_factory.create.assert_called_once()


# =============================================================================
# Integration-style tests (with real components)
# =============================================================================


class TestSimpleDataProvider:
    """Tests using SimpleDataProvider."""

    def test_simple_data_provider_creation(self):
        """Test creating SimpleDataProvider."""
        from maskgit3d.infrastructure.data.dataset import SimpleDataProvider

        provider = SimpleDataProvider(
            num_train=10,
            num_val=5,
            num_test=5,
            batch_size=2,
            in_channels=1,
            out_channels=2,
            spatial_size=(16, 16, 16),
        )

        assert provider.num_train == 10
        assert provider.num_val == 5
        assert provider.batch_size == 2
        assert provider.in_channels == 1
        assert provider.out_channels == 2

    def test_simple_data_provider_train_batch(self):
        """Test getting a training batch."""
        from maskgit3d.infrastructure.data.dataset import SimpleDataProvider

        provider = SimpleDataProvider(
            num_train=4,
            num_val=2,
            batch_size=2,
            in_channels=1,
            out_channels=2,
            spatial_size=(8, 8, 8),
        )

        train_iter = provider.train_loader()
        batch = next(train_iter)

        inputs, targets = batch
        assert inputs.shape == (2, 1, 8, 8, 8)
        assert targets.shape == (2, 2, 8, 8, 8)  # out_channels=2


# =============================================================================
# Pytest configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
