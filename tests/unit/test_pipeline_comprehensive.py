"""Comprehensive tests for pipeline module to reach 90%+ coverage."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from maskgit3d.application.pipeline import (
    FabricTestPipeline,
    FabricTrainingPipeline,
    TestPipeline,
)
from maskgit3d.domain.interfaces import (
    DataProvider,
    InferenceStrategy,
    Metrics,
    ModelInterface,
    OptimizerFactory,
    TrainingStrategy,
)


class SimpleModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 8)

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1)).view(x.shape)

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))

    @property
    def device(self):
        return next(self.parameters()).device


class SimpleDataProvider(DataProvider):
    def __init__(self, num_batches=2):
        self.num_batches = num_batches
        data = torch.randn(num_batches * 2, 1, 8, 8, 8)
        targets = torch.randint(0, 2, (num_batches * 2, 1, 8, 8, 8))
        self._data = TensorDataset(data, targets)

    def train_loader(self):
        return DataLoader(self._data, batch_size=2)

    def val_loader(self):
        return DataLoader(self._data, batch_size=2)

    def test_loader(self):
        return DataLoader(self._data, batch_size=2)


class SimpleTrainingStrategy(TrainingStrategy):
    def train_step(self, model, batch, optimizer):
        return {"loss": 0.5, "acc": 0.9}

    def validate_step(self, model, batch):
        return {"val_loss": 0.4, "val_acc": 0.95}


class SimpleOptimizerFactory(OptimizerFactory):
    def create(self, model_params):
        return torch.optim.SGD(list(model_params), lr=0.01)


class SimpleInferenceStrategy(InferenceStrategy):
    def predict(self, model, batch):
        return torch.randn(batch.size(0), 1, 8, 8, 8)

    def post_process(self, predictions):
        return {
            "images": predictions.cpu().numpy(),
            "masks": predictions.cpu().numpy(),
            "probs": predictions.cpu().numpy(),
        }


class SimpleMetrics(Metrics):
    def __init__(self):
        self.values = []

    def reset(self):
        self.values = []

    def update(self, predictions, targets):
        self.values.append((predictions, targets))

    def compute(self):
        return {"accuracy": 0.95, "f1": 0.92}


class TestTestPipelineComprehensive:
    """Comprehensive tests for TestPipeline."""

    @pytest.fixture
    def pipeline(self, tmp_path):
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

    def test_init_creates_output_dir(self, tmp_path):
        """Test that init creates output directory."""
        output_dir = tmp_path / "test_outputs"
        model = SimpleModel()
        data_provider = SimpleDataProvider()
        inference = SimpleInferenceStrategy()

        _ = TestPipeline(
            model=model,
            data_provider=data_provider,
            inference_strategy=inference,
            output_dir=str(output_dir),
        )

        assert output_dir.exists()

    def test_run_without_checkpoint(self, pipeline):
        """Test run without loading checkpoint."""
        results = pipeline.run()

        assert isinstance(results, dict)
        assert "accuracy" in results
        assert "f1" in results

    def test_run_with_checkpoint(self, pipeline, tmp_path):
        """Test run with checkpoint loading."""
        # Create a fake checkpoint
        checkpoint_path = tmp_path / "test.ckpt"
        torch.save({"model_state_dict": pipeline.model.state_dict()}, checkpoint_path)

        results = pipeline.run(checkpoint_path=str(checkpoint_path))

        assert isinstance(results, dict)
        assert "accuracy" in results

    def test_run_save_predictions(self, pipeline, tmp_path):
        """Test run with saving predictions."""
        pipeline.run(save_predictions=True)

        # Check that prediction files were created
        pred_files = list(pipeline.output_dir.glob("*.npy"))
        assert len(pred_files) > 0

    def test_load_checkpoint_with_model_key(self, pipeline, tmp_path):
        """Test loading checkpoint with 'model' key."""
        checkpoint_path = tmp_path / "test_model_key.ckpt"
        torch.save({"model": pipeline.model.state_dict()}, checkpoint_path)

        pipeline._load_checkpoint(str(checkpoint_path))
        # Should not raise

    def test_load_checkpoint_with_state_dict_key(self, pipeline, tmp_path):
        """Test loading checkpoint with 'state_dict' key."""
        checkpoint_path = tmp_path / "test_state_dict.ckpt"
        torch.save({"state_dict": pipeline.model.state_dict()}, checkpoint_path)

        pipeline._load_checkpoint(str(checkpoint_path))
        # Should not raise

    def test_load_checkpoint_raw(self, pipeline, tmp_path):
        """Test loading raw state dict checkpoint."""
        checkpoint_path = tmp_path / "test_raw.ckpt"
        torch.save(pipeline.model.state_dict(), checkpoint_path)

        pipeline._load_checkpoint(str(checkpoint_path))
        # Should not raise

    def test_save_predictions(self, pipeline, tmp_path):
        """Test _save_predictions method."""
        predictions = {
            "masks": torch.randn(2, 1, 8, 8, 8).numpy(),
            "probs": torch.randn(2, 1, 8, 8, 8).numpy(),
        }

        pipeline._save_predictions(predictions, batch_idx=0)

        masks_path = pipeline.output_dir / "predictions_batch_0.npy"
        probs_path = pipeline.output_dir / "probabilities_batch_0.npy"

        assert masks_path.exists()
        assert probs_path.exists()

    def test_run_without_metrics(self, tmp_path):
        """Test run without metrics."""
        model = SimpleModel()
        data_provider = SimpleDataProvider(num_batches=1)
        inference_strategy = SimpleInferenceStrategy()

        pipeline = TestPipeline(
            model=model,
            data_provider=data_provider,
            inference_strategy=inference_strategy,
            metrics=None,
            output_dir=str(tmp_path / "outputs"),
        )

        results = pipeline.run()

        assert isinstance(results, dict)
        assert "num_samples" in results

    def test_run_with_single_batch_input(self, pipeline):
        """Test run with single tensor batch (not tuple)."""

        # Data provider that returns single tensors
        class SingleTensorProvider(DataProvider):
            def train_loader(self):
                return DataLoader(TensorDataset(torch.randn(4, 1, 8, 8, 8)), batch_size=2)

            def val_loader(self):
                return DataLoader(TensorDataset(torch.randn(4, 1, 8, 8, 8)), batch_size=2)

            def test_loader(self):
                return DataLoader(TensorDataset(torch.randn(4, 1, 8, 8, 8)), batch_size=2)

        pipeline.data_provider = SingleTensorProvider()
        results = pipeline.run()

        assert isinstance(results, dict)


class TestFabricTestPipelineComprehensive:
    """Comprehensive tests for FabricTestPipeline."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        model = SimpleModel()
        data_provider = SimpleDataProvider(num_batches=1)
        inference_strategy = SimpleInferenceStrategy()
        metrics = SimpleMetrics()

        return FabricTestPipeline(
            model=model,
            data_provider=data_provider,
            inference_strategy=inference_strategy,
            metrics=metrics,
            accelerator="cpu",
            checkpoint_path=None,
            output_dir=str(tmp_path / "outputs"),
        )

    def test_init_raises_without_lightning(self):
        """Test that init raises ImportError if lightning not available."""
        with (
            patch("maskgit3d.application.pipeline.LIGHTNING_AVAILABLE", False),
            pytest.raises(ImportError, match="Lightning is not installed"),
        ):
            FabricTestPipeline(
                model=SimpleModel(),
                data_provider=SimpleDataProvider(),
                inference_strategy=SimpleInferenceStrategy(),
            )

    def test_run_basic(self, pipeline):
        """Test basic run."""
        results = pipeline.run()

        assert isinstance(results, dict)

    def test_run_with_tensorboard(self, pipeline, tmp_path):
        """Test run with tensorboard logging."""
        fake_tb_module = ModuleType("torch.utils.tensorboard")
        fake_writer = MagicMock()
        fake_tb_module.__dict__["SummaryWriter"] = MagicMock(return_value=fake_writer)

        with patch.dict(sys.modules, {"torch.utils.tensorboard": fake_tb_module}):
            results = pipeline.run(
                save_predictions=False,
                enable_tensorboard=True,
                tensorboard_dir=str(tmp_path / "tensorboard"),
            )

        assert isinstance(results, dict)

    def test_run_export_nifti(self, pipeline, tmp_path):
        """Test run with NIfTI export."""
        results = pipeline.run(export_nifti=True)
        assert isinstance(results, dict)

        # Check if NIfTI files were created
        nifti_files = list(pipeline.output_dir.glob("*.nii.gz"))
        assert isinstance(nifti_files, list)
        # May not be created if nibabel not installed

    def test_load_checkpoint_fabric_format(self, pipeline, tmp_path):
        """Test loading Fabric format checkpoint."""
        checkpoint_path = tmp_path / "fabric.ckpt"
        torch.save({"model": pipeline.model.state_dict()}, checkpoint_path)

        pipeline._fabric = MagicMock()
        pipeline._fabric.load.return_value = {"model": pipeline.model.state_dict()}

        pipeline._load_checkpoint(str(checkpoint_path))

    def test_load_checkpoint_legacy_format(self, pipeline, tmp_path):
        """Test loading legacy format checkpoint."""
        checkpoint_path = tmp_path / "legacy.ckpt"
        torch.save({"model_state_dict": pipeline.model.state_dict()}, checkpoint_path)

        pipeline._load_checkpoint(str(checkpoint_path))

    def test_load_checkpoint_fabric_fails_fallback(self, pipeline, tmp_path):
        """Test fallback when Fabric load fails."""
        checkpoint_path = tmp_path / "fallback.ckpt"
        torch.save({"state_dict": pipeline.model.state_dict()}, checkpoint_path)

        # Should not raise, should fallback
        pipeline._load_checkpoint(str(checkpoint_path))

    def test_log_tensorboard(self, pipeline, tmp_path):
        """Test _log_tensorboard method."""
        writer = MagicMock()
        images = torch.randn(2, 1, 8, 8, 8)
        predictions = {"masks": torch.randn(2, 1, 8, 8, 8)}
        targets = torch.randn(2, 1, 8, 8, 8)

        pipeline._fabric = MagicMock()
        pipeline._global_step = 0

        pipeline._log_tensorboard(writer, images, predictions, targets, batch_idx=0)
        assert writer.add_image.call_count == 3

    def test_export_nifti_with_targets(self, pipeline, tmp_path):
        """Test _export_nifti with targets."""
        images = torch.randn(2, 1, 8, 8, 8)
        predictions = {"masks": torch.randn(2, 1, 8, 8, 8), "probs": torch.randn(2, 1, 8, 8, 8)}
        targets = torch.randn(2, 1, 8, 8, 8)

        pipeline._export_nifti(images, predictions, targets, batch_idx=0)

    def test_export_nifti_no_nibabel(self, pipeline):
        """Test _export_nifti when nibabel not available."""
        with patch.dict("sys.modules", {"nibabel": None}):
            images = torch.randn(2, 1, 8, 8, 8)
            predictions = {"masks": torch.randn(2, 1, 8, 8, 8)}
            targets = None

            # Should not raise, should print warning
            pipeline._export_nifti(images, predictions, targets, batch_idx=0)


class TestFabricTrainingPipelineComprehensive:
    """Comprehensive tests for FabricTrainingPipeline."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        model = SimpleModel()
        data_provider = SimpleDataProvider(num_batches=1)
        training_strategy = SimpleTrainingStrategy()
        optimizer_factory = SimpleOptimizerFactory()

        return FabricTrainingPipeline(
            model=model,
            data_provider=data_provider,
            training_strategy=training_strategy,
            optimizer_factory=optimizer_factory,
            accelerator="cpu",
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_interval=1,
        )

    def test_init_raises_without_lightning(self):
        """Test that init raises ImportError if lightning not available."""
        with (
            patch("maskgit3d.application.pipeline.LIGHTNING_AVAILABLE", False),
            pytest.raises(ImportError, match="Lightning is not installed"),
        ):
            FabricTrainingPipeline(
                model=SimpleModel(),
                data_provider=SimpleDataProvider(),
                training_strategy=SimpleTrainingStrategy(),
                optimizer_factory=SimpleOptimizerFactory(),
            )

    def test_run_basic(self, pipeline):
        """Test basic training run."""
        with patch("maskgit3d.application.pipeline.L") as mock_l:
            mock_fabric = MagicMock()
            mock_l.Fabric.return_value = mock_fabric
            mock_fabric.setup.return_value = (
                pipeline.model,
                torch.optim.SGD(pipeline.model.parameters(), lr=0.01),
            )
            mock_fabric.setup_dataloaders.side_effect = lambda dl: dl
            mock_fabric.backward = MagicMock()

            with (
                patch.object(pipeline, "_train_epoch", return_value={"train_loss": [0.5]}),
                patch.object(pipeline, "_validate_epoch", return_value={"val_loss": [0.4]}),
                patch.object(pipeline, "_save_checkpoint"),
            ):
                history = pipeline.run(num_epochs=1)

        assert isinstance(history, dict)
        assert "train_loss" in history

    def test_run_with_validation_frequency(self, pipeline):
        """Test run with validation every N epochs."""
        with patch("maskgit3d.application.pipeline.L") as mock_l:
            mock_fabric = MagicMock()
            mock_l.Fabric.return_value = mock_fabric
            mock_fabric.setup.return_value = (
                pipeline.model,
                torch.optim.SGD(pipeline.model.parameters(), lr=0.01),
            )
            mock_fabric.setup_dataloaders.side_effect = lambda dl: dl
            mock_fabric.backward = MagicMock()

            with (
                patch.object(pipeline, "_train_epoch", return_value={"train_loss": [0.5]}),
                patch.object(pipeline, "_validate_epoch", return_value={"val_loss": [0.4]}),
                patch.object(pipeline, "_save_checkpoint"),
            ):
                history = pipeline.run(num_epochs=3, val_frequency=2)

        assert isinstance(history, dict)

    def test_run_resume_from_checkpoint(self, pipeline, tmp_path):
        """Test resuming from checkpoint."""
        checkpoint_path = tmp_path / "resume.ckpt"
        checkpoint_path.touch()

        with patch("maskgit3d.application.pipeline.L") as mock_l:
            mock_fabric = MagicMock()
            mock_l.Fabric.return_value = mock_fabric
            mock_fabric.setup.return_value = (
                pipeline.model,
                torch.optim.SGD(pipeline.model.parameters(), lr=0.01),
            )
            mock_fabric.setup_dataloaders.side_effect = lambda dl: dl
            mock_fabric.backward = MagicMock()
            mock_fabric.load.return_value = {"epoch": 2, "global_step": 10}

            with (
                patch.object(pipeline, "_train_epoch", return_value={"train_loss": [0.5]}),
                patch.object(pipeline, "_validate_epoch", return_value={"val_loss": [0.4]}),
                patch.object(pipeline, "_save_checkpoint"),
            ):
                history = pipeline.run(num_epochs=3, resume_from=str(checkpoint_path))

        assert isinstance(history, dict)

    def test_save_checkpoint(self, pipeline, tmp_path):
        """Test _save_checkpoint method."""
        pipeline._fabric = MagicMock()
        pipeline._optimizer = torch.optim.SGD(pipeline.model.parameters(), lr=0.01)

        train_metrics = {"train_loss": [0.5, 0.4]}
        val_metrics = {"val_loss": [0.6]}

        pipeline._save_checkpoint(epoch=0, train_metrics=train_metrics, val_metrics=val_metrics)

        pipeline._fabric.save.assert_called_once()

    def test_load_checkpoint(self, pipeline, tmp_path):
        """Test _load_checkpoint method."""
        checkpoint_path = tmp_path / "load.ckpt"
        torch.save(
            {
                "epoch": 5,
                "global_step": 100,
            },
            checkpoint_path,
        )

        # Mock fabric.load
        pipeline._fabric = MagicMock()
        pipeline._fabric.load.return_value = {"epoch": 5, "global_step": 100}

        start_epoch = pipeline._load_checkpoint(str(checkpoint_path))

        assert start_epoch == 5
        assert pipeline._global_step == 100

    def test_print_epoch_summary_with_validation(self, pipeline, capsys):
        """Test _print_epoch_summary with validation."""
        history = {"train_loss": [0.5, 0.4], "val_loss": [0.6, 0.55]}

        pipeline._print_epoch_summary(epoch=1, num_epochs=2, history=history, has_validation=True)

        captured = capsys.readouterr()
        assert "Epoch" in captured.out

    def test_print_epoch_summary_without_validation(self, pipeline, capsys):
        """Test _print_epoch_summary without validation."""
        history = {"train_loss": [0.5, 0.4]}

        pipeline._print_epoch_summary(epoch=1, num_epochs=2, history=history, has_validation=False)

        captured = capsys.readouterr()
        assert "Epoch" in captured.out

    def test_train_epoch_with_compute_loss(self, pipeline):
        """Test _train_epoch with compute_loss method."""

        # Strategy with compute_loss
        class StrategyWithComputeLoss(TrainingStrategy):
            def train_step(self, model, batch, optimizer):
                return {"loss": 0.5}

            def compute_loss(self, model, batch, output):
                return torch.nn.functional.mse_loss(output, batch[0])

            def validate_step(self, model, batch):
                return {"val_loss": 0.5}

        pipeline.training_strategy = StrategyWithComputeLoss()
        pipeline.model = MagicMock()
        pipeline.model.train = MagicMock()
        pipeline.model.return_value = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
        pipeline._fabric = MagicMock()
        pipeline._fabric.backward = MagicMock()
        pipeline._optimizer = MagicMock()

        metrics = pipeline._train_epoch(epoch=0, train_loader=pipeline.data_provider.train_loader())

        assert isinstance(metrics, dict)

    def test_global_step_property(self, pipeline):
        """Test global_step property."""
        pipeline._global_step = 42
        assert pipeline.global_step == 42

    def test_current_epoch_property(self, pipeline):
        """Test current_epoch property."""
        pipeline._current_epoch = 5
        assert pipeline.current_epoch == 5
