"""Extended tests for pipeline module to improve coverage."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from maskgit3d.application.pipeline import TrainingPipeline, TestPipeline


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x

    def parameters(self):
        return [self.param]

    def train(self, mode=True):
        pass

    def eval(self):
        pass

    def to(self, device):
        return self

    def state_dict(self):
        return {"param": self.param.data}

    def load_state_dict(self, state_dict):
        self.param.data = state_dict["param"]


class SimpleDataProvider:
    """Simple data provider for testing."""

    def __init__(self, num_batches=1):
        self.num_batches = num_batches

    def train_loader(self):
        for _ in range(self.num_batches):
            yield (torch.randn(2, 1, 8, 8, 8),)

    def val_loader(self):
        for _ in range(self.num_batches):
            yield (torch.randn(2, 1, 8, 8, 8),)

    def test_loader(self):
        for _ in range(self.num_batches):
            yield (torch.randn(2, 1, 8, 8, 8),)


class SimpleTrainingStrategy:
    """Simple training strategy for testing."""

    def train_step(self, model, batch, optimizer):
        return {"loss": 0.5, "acc": 0.9}

    def validate_step(self, model, batch):
        return {"val_loss": 0.4, "val_acc": 0.95}


class SimpleOptimizerFactory:
    """Simple optimizer factory for testing."""

    def create(self, params):
        return torch.optim.SGD(params, lr=0.01)


class SimpleInferenceStrategy:
    """Simple inference strategy for testing."""

    def predict(self, model, batch):
        return torch.randn(2, 1, 8, 8, 8)

    def post_process(self, predictions):
        return {
            "images": predictions.cpu().numpy(),
            "masks": predictions.cpu().numpy(),
            "probs": predictions.cpu().numpy(),
        }
        return {"images": predictions.cpu().numpy(), "masks": predictions.cpu().numpy()}


class SimpleMetrics:
    """Simple metrics for testing."""

    def __init__(self):
        self.values = []

    def reset(self):
        self.values = []

    def update(self, predictions, targets):
        self.values.append((predictions, targets))

    def compute(self):
        return {"accuracy": 0.95}


class TestTrainingPipelineExtended:
    """Extended tests for TrainingPipeline."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create a training pipeline."""
        model = SimpleModel()
        data_provider = SimpleDataProvider(num_batches=1)
        training_strategy = SimpleTrainingStrategy()
        optimizer_factory = SimpleOptimizerFactory()

        return TrainingPipeline(
            model=model,
            data_provider=data_provider,
            training_strategy=training_strategy,
            optimizer_factory=optimizer_factory,
            device=torch.device("cpu"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_interval=1,
        )

    def test_init(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.model is not None
        assert pipeline.data_provider is not None
        assert pipeline.training_strategy is not None
        assert pipeline.optimizer is not None
        assert pipeline.checkpoint_dir.exists()

    def test_run(self, pipeline):
        """Test running the pipeline."""
        history = pipeline.run(num_epochs=1, val_frequency=1)

        assert isinstance(history, dict)

    def test_run_without_validation(self, pipeline):
        """Test running without validation."""
        history = pipeline.run(num_epochs=1, val_frequency=2)

        assert isinstance(history, dict)

    def test_save_checkpoint(self, pipeline, tmp_path):
        """Test saving checkpoint."""
        train_metrics = {"train_loss": [0.5, 0.4]}
        val_metrics = {"val_loss": [0.45]}

        pipeline._save_checkpoint(0, train_metrics, val_metrics)

        checkpoint_files = list(pipeline.checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0


class TestTestPipelineExtended:
    """Extended tests for TestPipeline."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create a test pipeline."""
        model = SimpleModel()
        data_provider = SimpleDataProvider(num_batches=1)
        inference_strategy = SimpleInferenceStrategy()
        metrics = SimpleMetrics()

        return TestPipeline(
            model=model,
            data_provider=data_provider,
            inference_strategy=inference_strategy,
            metrics=metrics,
            device=torch.device("cpu"),
            output_dir=str(tmp_path / "outputs"),
        )

    def test_init(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.model is not None
        assert pipeline.data_provider is not None
        assert pipeline.inference_strategy is not None
        assert pipeline.metrics is not None
        assert pipeline.output_dir.exists()

    def test_run(self, pipeline):
        """Test running the pipeline."""
        results = pipeline.run()

        assert isinstance(results, dict)
        assert "accuracy" in results

    def test_run_save_predictions(self, pipeline):
        """Test running with saving predictions."""
        results = pipeline.run(save_predictions=True)

        # Check that predictions were saved
        output_files = list(pipeline.output_dir.glob("*.npy"))
        assert len(output_files) > 0
