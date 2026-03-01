"""
Unit tests for dynamic pipeline metrics.

These tests verify that the pipeline collects metrics dynamically from
strategy return values instead of hardcoded dice keys.
"""

from unittest.mock import MagicMock

import torch

from maskgit3d.application.pipeline import TrainingPipeline


class TestDynamicPipelineMetrics:
    """Tests for dynamic metric collection in TrainingPipeline."""

    def test_training_pipeline_collects_dynamic_metric_keys(self):
        """Test that pipeline collects metrics dynamically from strategy return values.

        The strategy returns custom metrics (token_acc) that are not hardcoded
        as dice. The pipeline should collect these dynamically.
        """
        # Create mock components
        mock_model = MagicMock()
        mock_model.to = MagicMock()

        # Create a strategy that returns custom metrics (not just dice)
        class CustomStrategy:
            def train_step(self, model, batch, optimizer):
                return {"loss": 1.0, "token_acc": 0.8}

            def validate_step(self, model, batch):
                return {"val_loss": 0.9, "val_token_acc": 0.81}

        # Create mock data provider with one batch
        mock_data_provider = MagicMock()
        mock_train_loader = MagicMock()
        mock_train_loader.__iter__ = MagicMock(
            return_value=iter([(torch.randn(1, 1, 8, 8, 8), torch.randn(1, 1, 8, 8, 8))])
        )
        mock_train_loader.__len__ = MagicMock(return_value=1)
        mock_data_provider.train_loader.return_value = mock_train_loader

        # Also need validation loader
        mock_val_loader = MagicMock()
        mock_val_loader.__iter__ = MagicMock(
            return_value=iter([(torch.randn(1, 1, 8, 8, 8), torch.randn(1, 1, 8, 8, 8))])
        )
        mock_val_loader.__len__ = MagicMock(return_value=1)
        mock_data_provider.val_loader.return_value = mock_val_loader

        mock_optimizer = MagicMock()

        # Create pipeline with custom strategy
        pipeline = TrainingPipeline(
            model=mock_model,
            data_provider=mock_data_provider,
            training_strategy=CustomStrategy(),
            optimizer_factory=MagicMock(create=MagicMock(return_value=mock_optimizer)),
            device=torch.device("cpu"),
            checkpoint_dir="/tmp/test_checkpoints",
            log_interval=1,
        )

        # Mock _save_checkpoint to avoid pickling issues with mock model
        pipeline._save_checkpoint = MagicMock()

        # Run for 1 epoch with 1 batch
        history = pipeline.run(num_epochs=1)

        # Assert that dynamic metric keys are collected
        # The pipeline should have collected "token_acc" from the strategy
        assert "train_token_acc" in history, (
            f"Expected 'train_token_acc' in history but got: {list(history.keys())}"
        )
        assert "val_token_acc" in history, (
            f"Expected 'val_token_acc' in history but got: {list(history.keys())}"
        )

        # Verify values were collected
        assert len(history["train_token_acc"]) == 1
        assert history["train_token_acc"][0] == 0.8
        assert len(history["val_token_acc"]) == 1
        assert history["val_token_acc"][0] == 0.81
