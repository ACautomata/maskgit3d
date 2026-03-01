"""Tests for LightningTrainingPipeline as LightningModule adapter."""

from unittest.mock import MagicMock, patch

import pytest
import torch

# Try importing pytorch_lightning, skip if not available
try:
    import pytorch_lightning as pl

    PL_AVAILABLE = True
except ImportError:
    PL_AVAILABLE = False

from maskgit3d.application.pipeline import LightningTrainingPipeline

pytestmark = pytest.mark.skipif(not PL_AVAILABLE, reason="pytorch_lightning not installed")


def test_lightning_training_pipeline_is_lightning_module():
    """Test that LightningTrainingPipeline is a subclass of LightningModule."""
    # Create mock dependencies
    mock_model = MagicMock()
    mock_data_provider = MagicMock()
    mock_training_strategy = MagicMock()
    mock_optimizer_factory = MagicMock()
    mock_optimizer = MagicMock()
    mock_optimizer_factory.create.return_value = mock_optimizer

    # Create the pipeline
    pipeline = LightningTrainingPipeline(
        model=mock_model,
        data_provider=mock_data_provider,
        training_strategy=mock_training_strategy,
        optimizer_factory=mock_optimizer_factory,
    )

    # Assert it's a LightningModule
    assert isinstance(pipeline, pl.LightningModule)


def test_lightning_training_pipeline_has_required_methods():
    """Test that LightningTrainingPipeline has Lightning-compatible methods."""
    mock_model = MagicMock()
    mock_data_provider = MagicMock()
    mock_training_strategy = MagicMock()
    mock_optimizer_factory = MagicMock()
    mock_optimizer = MagicMock()
    mock_optimizer_factory.create.return_value = mock_optimizer

    pipeline = LightningTrainingPipeline(
        model=mock_model,
        data_provider=mock_data_provider,
        training_strategy=mock_training_strategy,
        optimizer_factory=mock_optimizer_factory,
    )

    # Check required methods exist
    assert hasattr(pipeline, "training_step")
    assert hasattr(pipeline, "validation_step")
    assert hasattr(pipeline, "configure_optimizers")


def test_lightning_training_pipeline_training_step():
    """Test that training_step returns proper Lightning format."""
    mock_model = MagicMock()
    mock_data_provider = MagicMock()
    mock_training_strategy = MagicMock()
    mock_optimizer_factory = MagicMock()
    mock_optimizer = MagicMock()
    mock_optimizer_factory.create.return_value = mock_optimizer

    # Mock training strategy to return metrics
    mock_training_strategy.train_step.return_value = {"loss": 0.5}

    pipeline = LightningTrainingPipeline(
        model=mock_model,
        data_provider=mock_data_provider,
        training_strategy=mock_training_strategy,
        optimizer_factory=mock_optimizer_factory,
    )

    # Create a dummy batch
    batch = (torch.randn(1, 1, 64, 64, 64), torch.randn(1, 1, 64, 64, 64))

    # Mock optimizers to return the mock optimizer
    with patch.object(pipeline, "_move_batch_to_device", return_value=batch), \
         patch.object(pipeline, "optimizers", return_value=mock_optimizer):
        result = pipeline.training_step(batch, 0)

    # Check result has loss key
    assert "loss" in result


def test_lightning_training_pipeline_configure_optimizers():
    """Test that configure_optimizers returns the optimizer."""
    mock_model = MagicMock()
    mock_data_provider = MagicMock()
    mock_training_strategy = MagicMock()
    mock_optimizer_factory = MagicMock()
    mock_optimizer = MagicMock()
    mock_optimizer_factory.create.return_value = mock_optimizer

    pipeline = LightningTrainingPipeline(
        model=mock_model,
        data_provider=mock_data_provider,
        training_strategy=mock_training_strategy,
        optimizer_factory=mock_optimizer_factory,
    )

    optimizer = pipeline.configure_optimizers()

    # Should return the optimizer created by the factory
    assert optimizer is mock_optimizer
