"""Tests for training time callback."""

import time

import pytest

from maskgit3d.callbacks.training_time import TrainingTimeCallback


class TestTrainingTimeCallback:
    """Test suite for TrainingTimeCallback."""

    def test_callback_initialization(self):
        """Test callback can be initialized with different parameters."""
        callback = TrainingTimeCallback(log_every_n_epochs=1)
        assert callback.log_every_n_epochs == 1
        assert callback.estimate_time_to_completion is True

        callback = TrainingTimeCallback(log_every_n_epochs=5, estimate_time_to_completion=False)
        assert callback.log_every_n_epochs == 5
        assert callback.estimate_time_to_completion is False

    def test_time_formatting(self):
        """Test time formatting function."""
        assert TrainingTimeCallback._format_time(30) == "30.0s"
        assert TrainingTimeCallback._format_time(90) == "1.5m"
        assert TrainingTimeCallback._format_time(3600) == "1.00h"
        assert TrainingTimeCallback._format_time(7200) == "2.00h"

    def test_etc_estimation(self):
        """Test time to completion estimation."""
        callback = TrainingTimeCallback()
        callback._epoch_times = [100.0, 110.0, 105.0]

        class MockTrainer:
            max_epochs = 10
            current_epoch = 3

        etc = callback._estimate_etc(MockTrainer())
        assert etc is not None
        assert etc > 0

        # Should be roughly 7 epochs * ~105 seconds
        assert 600 < etc < 800

    def test_etc_with_no_epochs(self):
        """Test ETC estimation with no epoch history."""
        callback = TrainingTimeCallback()
        callback._epoch_times = []

        class MockTrainer:
            max_epochs = 10
            current_epoch = 0

        etc = callback._estimate_etc(MockTrainer())
        assert etc is None

    def test_average_epoch_time(self):
        """Test average epoch time calculation."""
        callback = TrainingTimeCallback()
        callback._epoch_times = [100.0, 110.0, 105.0, 95.0, 100.0]

        avg = callback._get_avg_epoch_time()
        assert avg == pytest.approx(102.0, abs=0.1)

        # Should use last 5 epochs
        callback._epoch_times = [100.0, 110.0, 105.0, 95.0, 100.0, 90.0]
        avg = callback._get_avg_epoch_time()
        assert avg == pytest.approx(100.0, abs=0.1)

    def test_state_dict(self):
        """Test state dict saving and loading."""
        callback = TrainingTimeCallback()
        callback._total_train_time = 1000.0
        callback._total_validation_time = 100.0
        callback._epoch_times = [100.0, 110.0, 105.0]

        state = callback.state_dict()
        assert state["total_train_time"] == 1000.0
        assert state["total_validation_time"] == 100.0
        assert state["epoch_times"] == [100.0, 110.0, 105.0]

        new_callback = TrainingTimeCallback()
        new_callback.load_state_dict(state)
        assert new_callback._total_train_time == 1000.0
        assert new_callback._total_validation_time == 100.0
        assert new_callback._epoch_times == [100.0, 110.0, 105.0]

    def test_train_start_tracking(self):
        """Test training start time tracking."""
        callback = TrainingTimeCallback()
        assert callback._train_start_time is None

        callback.on_train_start(None, None)
        assert callback._train_start_time is not None

    def test_epoch_start_tracking(self):
        """Test epoch start time tracking."""
        callback = TrainingTimeCallback()
        assert callback._epoch_start_time is None

        callback.on_train_epoch_start(None, None)
        assert callback._epoch_start_time is not None
