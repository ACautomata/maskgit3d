"""Tests for FIDCallback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from lightning.pytorch import LightningModule

from maskgit3d.callbacks.fid_logging import FIDCallback
from maskgit3d.metrics.fid import FIDMetric


class _DummyModule(LightningModule):
    """Minimal LightningModule for testing callbacks."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class TestFIDCallback:
    """Tests for FIDCallback."""

    def test_init_default_params(self) -> None:
        """Test default initialization."""
        cb = FIDCallback()
        assert cb.input_min == -1.0
        assert cb.input_max == 1.0
        assert cb._fid_metric is None

    def test_init_custom_params(self) -> None:
        """I3: Verify input_min/input_max are passed through correctly."""
        cb = FIDCallback(input_min=0.0, input_max=255.0)
        assert cb.input_min == 0.0
        assert cb.input_max == 255.0

    def test_get_fid_metric_creates_with_correct_params(self) -> None:
        """I3: FIDMetric is created with correct input_min/input_max from callback."""
        cb = FIDCallback(input_min=0.0, input_max=1.0)
        pl_module = _DummyModule()

        metric = cb._get_fid_metric(pl_module)

        assert isinstance(metric, FIDMetric)
        assert metric.input_min == 0.0
        assert metric.input_max == 1.0

    def test_get_fid_metric_lazy_init(self) -> None:
        """Test that FIDMetric is lazily initialized and reused."""
        cb = FIDCallback()
        pl_module = _DummyModule()

        assert cb._fid_metric is None
        metric1 = cb._get_fid_metric(pl_module)
        assert cb._fid_metric is not None
        metric2 = cb._get_fid_metric(pl_module)
        assert metric1 is metric2

    def test_extract_batch_pair_with_both_keys(self) -> None:
        """Test _extract_batch_pair returns pair when both keys present."""
        x_recon = torch.randn(2, 1, 4, 4)
        x_real = torch.randn(2, 1, 4, 4)
        outputs = {"x_recon": x_recon, "x_real": x_real}

        result = FIDCallback._extract_batch_pair(outputs)
        assert result is not None
        assert result[0] is x_recon
        assert result[1] is x_real

    def test_extract_batch_pair_missing_key_returns_none(self) -> None:
        """Test _extract_batch_pair returns None when keys missing."""
        assert FIDCallback._extract_batch_pair(None) is None
        assert FIDCallback._extract_batch_pair("string") is None
        assert FIDCallback._extract_batch_pair({}) is None
        assert FIDCallback._extract_batch_pair({"x_recon": torch.randn(2)}) is None
        assert FIDCallback._extract_batch_pair({"x_real": torch.randn(2)}) is None

    def test_on_validation_batch_end_accumulates(self) -> None:
        """Test that validation batch end accumulates features."""
        cb = FIDCallback()
        pl_module = _DummyModule()
        outputs = {
            "x_recon": torch.randn(2, 1, 4, 4),
            "x_real": torch.randn(2, 1, 4, 4),
        }

        with patch.object(cb, "_get_fid_metric") as mock_get:
            mock_metric = MagicMock(spec=FIDMetric)
            mock_get.return_value = mock_metric

            cb.on_validation_batch_end(
                trainer=MagicMock(),
                pl_module=pl_module,
                outputs=outputs,
                batch=MagicMock(),
                batch_idx=0,
            )

            mock_metric.update.assert_called_once()

    def test_on_validation_batch_end_no_outputs(self) -> None:
        """Test that validation batch end skips when outputs is None."""
        cb = FIDCallback()
        pl_module = _DummyModule()

        cb.on_validation_batch_end(
            trainer=MagicMock(),
            pl_module=pl_module,
            outputs=None,
            batch=MagicMock(),
            batch_idx=0,
        )

        assert cb._fid_metric is None  # Not initialized

    def test_on_validation_epoch_end_computes_and_logs(self) -> None:
        """Test that validation epoch end computes FID and logs."""
        cb = FIDCallback()
        pl_module = _DummyModule()
        pl_module.log = MagicMock()  # type: ignore[assignment]

        mock_metric = MagicMock(spec=FIDMetric)
        mock_metric.compute.return_value = {"fid": 42.5}
        cb._fid_metric = mock_metric

        cb.on_validation_epoch_end(trainer=MagicMock(), pl_module=pl_module)

        mock_metric.compute.assert_called_once()
        pl_module.log.assert_called_once_with("val_fid", 42.5, prog_bar=True)
        mock_metric.reset.assert_called_once()

    def test_on_validation_epoch_end_no_metric(self) -> None:
        """Test that validation epoch end is a no-op when no metric."""
        cb = FIDCallback()
        pl_module = _DummyModule()
        pl_module.log = MagicMock()  # type: ignore[assignment]

        cb.on_validation_epoch_end(trainer=MagicMock(), pl_module=pl_module)
        pl_module.log.assert_not_called()

    def test_on_test_batch_end_accumulates(self) -> None:
        """Test that test batch end accumulates features."""
        cb = FIDCallback()
        pl_module = _DummyModule()
        outputs = {
            "x_recon": torch.randn(2, 1, 4, 4),
            "x_real": torch.randn(2, 1, 4, 4),
        }

        with patch.object(cb, "_get_fid_metric") as mock_get:
            mock_metric = MagicMock(spec=FIDMetric)
            mock_get.return_value = mock_metric

            cb.on_test_batch_end(
                trainer=MagicMock(),
                pl_module=pl_module,
                outputs=outputs,
                batch=MagicMock(),
                batch_idx=0,
            )

            mock_metric.update.assert_called_once()

    def test_on_test_epoch_end_computes_and_logs(self) -> None:
        """Test that test epoch end logs with 'fid:test' key."""
        cb = FIDCallback()
        pl_module = _DummyModule()
        pl_module.log = MagicMock()  # type: ignore[assignment]

        mock_metric = MagicMock(spec=FIDMetric)
        mock_metric.compute.return_value = {"fid": 99.9}
        cb._fid_metric = mock_metric

        cb.on_test_epoch_end(trainer=MagicMock(), pl_module=pl_module)

        mock_metric.compute.assert_called_once()
        pl_module.log.assert_called_once_with("fid:test", 99.9, prog_bar=True)
        mock_metric.reset.assert_called_once()

    def test_device_from_pl_module(self) -> None:
        """Test that device is extracted from pl_module."""
        cb = FIDCallback()
        pl_module = _DummyModule()

        with patch("maskgit3d.callbacks.fid_logging.FIDMetric") as MockFID:
            mock_instance = MagicMock()
            MockFID.return_value = mock_instance

            _ = cb._get_fid_metric(pl_module)

            MockFID.assert_called_once()
            call_kwargs = MockFID.call_args
            # Device should come from pl_module.device
            assert call_kwargs.kwargs.get("device") is not None or len(call_kwargs.args) > 0
