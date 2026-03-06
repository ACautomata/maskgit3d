"""Integration tests for axial slice visualization callback."""

from unittest.mock import MagicMock

import pytest
import torch

from maskgit3d.infrastructure.training.callbacks import AxialSliceVisualizationCallback


class TestSliceVisualizationIntegration:
    """Integration tests for slice visualization callback."""

    def test_callback_with_mock_validation_batch(self, tmp_path):
        """Test callback with realistic validation batch."""
        callback = AxialSliceVisualizationCallback(
            num_samples=2,
            slice_range=3,
            output_dir=str(tmp_path / "slices"),
            enable_tensorboard=False,
        )

        # Create realistic 3D batch (MedMNIST-like: B, C, D, H, W)
        batch = {"volumes": torch.randn(8, 1, 28, 28, 28)}

        # Mock trainer with global_step
        mock_trainer = MagicMock()
        mock_trainer.global_step = 0
        callback._trainer = mock_trainer

        # Call the internal method directly to test
        callback._process_batch(batch, 0, "val")

        # Verify files were created
        slices_dir = tmp_path / "slices"
        assert slices_dir.exists()

        # Should have 2 sample files (num_samples=2)
        png_files = list(slices_dir.glob("val_batch0_sample*.png"))
        assert len(png_files) == 2

        # Verify file naming pattern
        filenames = [f.name for f in png_files]
        assert "val_batch0_sample0.png" in filenames
        assert "val_batch0_sample1.png" in filenames

    def test_callback_tensorboard_logging(self, tmp_path):
        """Test TensorBoard logging."""
        pytest.importorskip("tensorboard", reason="tensorboard required")
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = tmp_path / "tensorboard"

        callback = AxialSliceVisualizationCallback(
            num_samples=1,
            slice_range=3,
            output_dir=str(tmp_path / "slices"),
            enable_tensorboard=True,
        )

        # Set writer manually
        writer = SummaryWriter(str(tb_dir))
        callback._tensorboard_writer = writer

        # Process batch
        batch = {"volumes": torch.randn(4, 1, 28, 28, 28)}
        mock_trainer = MagicMock()
        mock_trainer.global_step = 0
        callback._trainer = mock_trainer

        callback._process_batch(batch, 0, "val")

        writer.close()

        # Verify TensorBoard event files exist
        tb_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(tb_files) > 0

    def test_callback_with_different_batch_indices(self, tmp_path):
        """Test callback handles different batch indices correctly."""
        callback = AxialSliceVisualizationCallback(
            num_samples=1,
            slice_range=3,
            output_dir=str(tmp_path / "slices"),
            enable_tensorboard=False,
        )

        batch = {"volumes": torch.randn(4, 1, 28, 28, 28)}
        mock_trainer = MagicMock()
        mock_trainer.global_step = 0
        callback._trainer = mock_trainer

        # Process multiple batches
        callback._process_batch(batch, 0, "val")
        callback._process_batch(batch, 1, "val")
        callback._process_batch(batch, 2, "val")

        # Verify files with different batch indices
        slices_dir = tmp_path / "slices"
        png_files = list(slices_dir.glob("val_batch*.png"))
        assert len(png_files) == 3

        filenames = [f.name for f in png_files]
        assert "val_batch0_sample0.png" in filenames
        assert "val_batch1_sample0.png" in filenames
        assert "val_batch2_sample0.png" in filenames

    def test_callback_creates_valid_png_files(self, tmp_path):
        """Test that saved PNG files are valid images."""
        callback = AxialSliceVisualizationCallback(
            num_samples=1,
            slice_range=3,
            output_dir=str(tmp_path / "slices"),
            enable_tensorboard=False,
        )

        batch = {"volumes": torch.randn(2, 1, 16, 16, 16)}
        mock_trainer = MagicMock()
        mock_trainer.global_step = 0
        callback._trainer = mock_trainer

        callback._process_batch(batch, 0, "val")

        # Verify PNG file is valid
        slices_dir = tmp_path / "slices"
        png_files = list(slices_dir.glob("val_batch0_sample*.png"))
        assert len(png_files) == 1

        # Try to open with PIL
        from PIL import Image

        img = Image.open(png_files[0])
        assert img.format == "PNG"
        assert img.mode == "L"  # Grayscale

    def test_callback_respects_num_samples(self, tmp_path):
        """Test that callback respects num_samples parameter."""
        callback = AxialSliceVisualizationCallback(
            num_samples=3,
            slice_range=2,
            output_dir=str(tmp_path / "slices"),
            enable_tensorboard=False,
        )

        batch = {"volumes": torch.randn(8, 1, 28, 28, 28)}
        mock_trainer = MagicMock()
        mock_trainer.global_step = 0
        callback._trainer = mock_trainer

        callback._process_batch(batch, 0, "val")

        # Should have exactly 3 samples despite batch size of 8
        slices_dir = tmp_path / "slices"
        png_files = list(slices_dir.glob("val_batch0_sample*.png"))
        assert len(png_files) == 3
