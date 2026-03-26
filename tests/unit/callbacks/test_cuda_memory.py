"""Tests for CUDA memory cleanup callback."""

from unittest.mock import MagicMock, patch

from lightning.pytorch import Trainer

from maskgit3d.callbacks.cuda_memory import CUDAMemoryCleanupCallback


class TestCUDAMemoryCleanupCallback:
    def test_on_train_epoch_end_with_cuda(self) -> None:
        callback = CUDAMemoryCleanupCallback()
        trainer = MagicMock(spec=Trainer)
        pl_module = MagicMock()

        with (
            patch("maskgit3d.callbacks.cuda_memory.torch.cuda.is_available", return_value=True),
            patch("maskgit3d.callbacks.cuda_memory.gc.collect") as mock_gc,
            patch("maskgit3d.callbacks.cuda_memory.torch.cuda.empty_cache") as mock_empty,
        ):
            callback.on_train_epoch_end(trainer, pl_module)
            mock_gc.assert_called_once()
            mock_empty.assert_called_once()

    def test_on_train_epoch_end_without_cuda(self) -> None:
        callback = CUDAMemoryCleanupCallback()
        trainer = MagicMock(spec=Trainer)
        pl_module = MagicMock()

        with (
            patch("maskgit3d.callbacks.cuda_memory.torch.cuda.is_available", return_value=False),
            patch("maskgit3d.callbacks.cuda_memory.gc.collect") as mock_gc,
            patch("maskgit3d.callbacks.cuda_memory.torch.cuda.empty_cache") as mock_empty,
        ):
            callback.on_train_epoch_end(trainer, pl_module)
            mock_gc.assert_not_called()
            mock_empty.assert_not_called()

    def test_on_validation_epoch_end_with_cuda(self) -> None:
        callback = CUDAMemoryCleanupCallback()
        trainer = MagicMock(spec=Trainer)
        pl_module = MagicMock()

        with (
            patch("maskgit3d.callbacks.cuda_memory.torch.cuda.is_available", return_value=True),
            patch("maskgit3d.callbacks.cuda_memory.gc.collect") as mock_gc,
            patch("maskgit3d.callbacks.cuda_memory.torch.cuda.empty_cache") as mock_empty,
        ):
            callback.on_validation_epoch_end(trainer, pl_module)
            mock_gc.assert_called_once()
            mock_empty.assert_called_once()

    def test_on_validation_epoch_end_without_cuda(self) -> None:
        callback = CUDAMemoryCleanupCallback()
        trainer = MagicMock(spec=Trainer)
        pl_module = MagicMock()

        with (
            patch("maskgit3d.callbacks.cuda_memory.torch.cuda.is_available", return_value=False),
            patch("maskgit3d.callbacks.cuda_memory.gc.collect") as mock_gc,
            patch("maskgit3d.callbacks.cuda_memory.torch.cuda.empty_cache") as mock_empty,
        ):
            callback.on_validation_epoch_end(trainer, pl_module)
            mock_gc.assert_not_called()
            mock_empty.assert_not_called()
