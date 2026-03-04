"""Tests for FabricTrainingPipeline as Lightning Fabric adapter."""

from unittest.mock import MagicMock, patch

import pytest

# Try importing lightning, skip if not available
try:
    import lightning as L

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

from maskgit3d.application.pipeline import FabricTrainingPipeline

pytestmark = pytest.mark.skipif(not LIGHTNING_AVAILABLE, reason="lightning not installed")


def test_fabric_training_pipeline_initialization():
    """Test that FabricTrainingPipeline initializes correctly."""
    # Create mock dependencies
    mock_model = MagicMock()
    mock_data_provider = MagicMock()
    mock_training_strategy = MagicMock()
    mock_optimizer_factory = MagicMock()
    mock_optimizer = MagicMock()
    mock_optimizer_factory.create.return_value = mock_optimizer

    # Create the pipeline
    pipeline = FabricTrainingPipeline(
        model=mock_model,
        data_provider=mock_data_provider,
        training_strategy=mock_training_strategy,
        optimizer_factory=mock_optimizer_factory,
    )

    # Check attributes are set
    assert pipeline.model is mock_model
    assert pipeline.data_provider is mock_data_provider
    assert pipeline.training_strategy is mock_training_strategy
    assert pipeline.optimizer_factory is mock_optimizer_factory


def test_fabric_training_pipeline_has_required_methods():
    """Test that FabricTrainingPipeline has required methods."""
    mock_model = MagicMock()
    mock_data_provider = MagicMock()
    mock_training_strategy = MagicMock()
    mock_optimizer_factory = MagicMock()
    mock_optimizer = MagicMock()
    mock_optimizer_factory.create.return_value = mock_optimizer

    pipeline = FabricTrainingPipeline(
        model=mock_model,
        data_provider=mock_data_provider,
        training_strategy=mock_training_strategy,
        optimizer_factory=mock_optimizer_factory,
    )

    # Check required methods exist
    assert hasattr(pipeline, "run")
    assert hasattr(pipeline, "_train_epoch")
    assert hasattr(pipeline, "_validate_epoch")
    assert hasattr(pipeline, "_save_checkpoint")
    assert hasattr(pipeline, "_load_checkpoint")


def test_fabric_training_pipeline_configurable_params():
    """Test that FabricTrainingPipeline accepts Fabric configuration."""
    mock_model = MagicMock()
    mock_data_provider = MagicMock()
    mock_training_strategy = MagicMock()
    mock_optimizer_factory = MagicMock()

    # Create with custom Fabric parameters
    pipeline = FabricTrainingPipeline(
        model=mock_model,
        data_provider=mock_data_provider,
        training_strategy=mock_training_strategy,
        optimizer_factory=mock_optimizer_factory,
        accelerator="cuda",
        devices=2,
        strategy="ddp",
        precision="16-mixed",
    )

    # Check Fabric parameters are stored
    assert pipeline._accelerator == "cuda"
    assert pipeline._devices == 2
    assert pipeline._strategy == "ddp"
    assert pipeline._precision == "16-mixed"


def test_fabric_training_pipeline_import_error():
    """Test that FabricTrainingPipeline raises ImportError when lightning not available."""
    with patch("maskgit3d.application.pipeline.LIGHTNING_AVAILABLE", False):
        mock_model = MagicMock()
        mock_data_provider = MagicMock()
        mock_training_strategy = MagicMock()
        mock_optimizer_factory = MagicMock()

        with pytest.raises(ImportError, match="Lightning is not installed"):
            FabricTrainingPipeline(
                model=mock_model,
                data_provider=mock_data_provider,
                training_strategy=mock_training_strategy,
                optimizer_factory=mock_optimizer_factory,
            )


def test_fabric_training_pipeline_checkpoint_dir():
    """Test that FabricTrainingPipeline creates checkpoint directory."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        mock_model = MagicMock()
        mock_data_provider = MagicMock()
        mock_training_strategy = MagicMock()
        mock_optimizer_factory = MagicMock()

        checkpoint_dir = f"{tmpdir}/checkpoints"

        pipeline = FabricTrainingPipeline(
            model=mock_model,
            data_provider=mock_data_provider,
            training_strategy=mock_training_strategy,
            optimizer_factory=mock_optimizer_factory,
            checkpoint_dir=checkpoint_dir,
        )

        # Check that checkpoint_dir is set
        assert str(pipeline.checkpoint_dir) == checkpoint_dir
