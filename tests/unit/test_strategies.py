"""
Unit tests for training strategies.

These tests verify the functionality of training and inference strategies.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from maskgit3d.infrastructure.training.strategies import (
    AdamOptimizerFactory,
    AdamWOptimizerFactory,
    MaskGITInference,
    MaskGITTrainingStrategy,
    SGDOptimizerFactory,
    VQGANInference,
    VQGANMetrics,
    VQGANOptimizerFactory,
    VQGANTrainingStrategy,
)


class TestVQGANTrainingStrategy:
    """Tests for VQGANTrainingStrategy."""

    def test_vqgan_training_strategy_creation(self):
        """Test creating VQGANTrainingStrategy."""
        strategy = VQGANTrainingStrategy(
            codebook_weight=1.0,
            pixel_loss_weight=1.0,
            perceptual_weight=0.0,
        )
        assert strategy.codebook_weight == 1.0
        assert strategy.pixel_loss_weight == 1.0

    def test_vqgan_train_step_with_mock_model(self):
        """Test training step with mock model."""
        strategy = VQGANTrainingStrategy(perceptual_weight=0.0)
        mock_model = MagicMock()

        # Mock model output: (reconstructed, quantizer_loss)
        # Use tensors that require grad for backward compatibility
        mock_model.forward_with_loss.return_value = (
            torch.randn(2, 1, 8, 8, 8, requires_grad=True),  # xrec
            torch.tensor(0.1, requires_grad=True),  # qloss
        )

        batch = (torch.randn(2, 1, 8, 8, 8),)
        optimizer = MagicMock()

        # Use patch to avoid actual backward pass
        with patch.object(optimizer, "zero_grad"), patch.object(optimizer, "step"):
            metrics = strategy.train_step(mock_model, batch, optimizer)

        assert "loss" in metrics
        assert "rec_loss" in metrics
        assert "codebook_loss" in metrics

    def test_vqgan_validate_step(self):
        """Test validation step."""
        strategy = VQGANTrainingStrategy(perceptual_weight=0.0)
        mock_model = MagicMock()

        mock_model.forward_with_loss.return_value = (
            torch.randn(2, 1, 8, 8, 8),
            torch.tensor(0.1),
        )

        batch = (torch.randn(2, 1, 8, 8, 8),)

        metrics = strategy.validate_step(mock_model, batch)

        assert "val_loss" in metrics
        assert "val_rec_loss" in metrics
        assert "val_codebook_loss" in metrics
        # Metrics should be Python floats
        assert isinstance(metrics["val_loss"], float)


class TestVQGANInference:
    """Tests for VQGANInference."""

    def test_vqgan_inference_creation(self):
        """Test creating VQGANInference."""
        inference = VQGANInference(
            mode="reconstruct",
            num_samples=1,
            temperature=1.0,
        )
        assert inference.mode == "reconstruct"
        assert inference.temperature == 1.0

    def test_vqgan_inference_reconstruct(self):
        """Test reconstruction mode."""
        inference = VQGANInference(mode="reconstruct")
        mock_model = MagicMock()

        # VQGANInference.reconstruct calls vq_model.encode(x=batch) which returns (quantized, _, _)
        # Then calls vq_model.decode(quantized)
        mock_model.encode.return_value = (torch.randn(2, 1, 8, 8, 8), None, None)
        mock_model.decode.return_value = torch.randn(2, 1, 8, 8, 8)
        # It expects the model to return a tuple where first element is the reconstruction
        mock_model.return_value = (torch.randn(2, 1, 8, 8, 8),)

        batch = torch.randn(2, 1, 8, 8, 8)
        result = inference.predict(mock_model, batch)

        assert result.shape == batch.shape
        mock_model.encode.assert_called_once()

    def test_vqgan_inference_post_process(self):
        """Test post-processing."""
        inference = VQGANInference()
        # Input in range [-1, 1]
        predictions = torch.randn(2, 1, 8, 8, 8)

        result = inference.post_process(predictions)

        assert "images" in result
        assert result["images"].shape == predictions.shape
        # Should be normalized to [0, 1]
        assert result["images"].min() >= 0
        assert result["images"].max() <= 1

    def test_vqgan_inference_unknown_mode(self):
        """Test unknown inference mode raises error."""
        inference = VQGANInference(mode="unknown")
        mock_model = MagicMock()

        with pytest.raises(ValueError, match="Unknown mode"):
            inference.predict(mock_model, torch.randn(2, 1, 8, 8, 8))


class TestVQGANMetrics:
    """Tests for VQGANMetrics."""

    def test_vqgan_metrics_creation(self):
        """Test creating VQGANMetrics."""
        # VQGANMetrics may fail if MONAI is not available with correct API
        # We test that it can be instantiated
        try:
            metrics = VQGANMetrics(data_range=1.0, spatial_dims=3)
            assert metrics.data_range == 1.0
            assert metrics.spatial_dims == 3
        except TypeError:
            # MONAI API may have changed - skip if instantiation fails
            pytest.skip("MONAI SSIMMetric API changed")

    def test_vqgan_metrics_update_and_compute(self):
        """Test metrics update and compute."""
        try:
            metrics = VQGANMetrics(data_range=1.0, spatial_dims=3)
        except TypeError:
            pytest.skip("MONAI SSIMMetric API changed")

        # Create test predictions and targets in [-1, 1] range
        predictions = torch.randn(2, 1, 8, 8, 8)
        targets = torch.randn(2, 1, 8, 8, 8)

        metrics.update(predictions, targets)
        result = metrics.compute()

        assert "psnr" in result
        assert "ssim" in result

    def test_vqgan_metrics_reset(self):
        """Test metrics reset."""
        try:
            metrics = VQGANMetrics(data_range=1.0, spatial_dims=3)
        except TypeError:
            pytest.skip("MONAI SSIMMetric API changed")

        predictions = torch.randn(2, 1, 8, 8, 8)
        targets = torch.randn(2, 1, 8, 8, 8)

        metrics.update(predictions, targets)
        metrics.reset()

        # After reset, compute should return initial values
        result = metrics.compute()
        assert result["psnr"] == 0.0

    def test_vqgan_metrics_with_lpips(self):
        """Test metrics with LPIPS enabled."""
        try:
            metrics = VQGANMetrics(data_range=1.0, spatial_dims=3, enable_lpips=True)
        except TypeError:
            pytest.skip("MONAI SSIMMetric API changed")

        predictions = torch.randn(2, 1, 8, 8, 8)
        targets = torch.randn(2, 1, 8, 8, 8)

        metrics.update(predictions, targets)
        result = metrics.compute()

        assert "psnr" in result
        assert "ssim" in result

    def test_vqgan_metrics_compute_with_stats(self):
        """Test compute_with_stats method."""
        try:
            metrics = VQGANMetrics(data_range=1.0, spatial_dims=3)
        except TypeError:
            pytest.skip("MONAI SSIMMetric API changed")

        predictions = torch.randn(2, 1, 8, 8, 8)
        targets = torch.randn(2, 1, 8, 8, 8)

        metrics.update(predictions, targets)
        stats = metrics.compute_with_stats()

        assert "psnr" in stats
        assert "mean" in stats["psnr"]
        assert "std" in stats["psnr"]
        assert "min" in stats["psnr"]
        assert "max" in stats["psnr"]
        assert "count" in stats["psnr"]

    def test_vqgan_metrics_export_json(self, tmp_path):
        """Test export to JSON."""
        try:
            metrics = VQGANMetrics(data_range=1.0, spatial_dims=3)
        except TypeError:
            pytest.skip("MONAI SSIMMetric API changed")

        predictions = torch.randn(2, 1, 8, 8, 8)
        targets = torch.randn(2, 1, 8, 8, 8)

        metrics.update(predictions, targets)
        json_path = tmp_path / "metrics.json"
        metrics.export_json(str(json_path))

        assert json_path.exists()
        import json

        with open(json_path) as f:
            data = json.load(f)
        assert "summary" in data
        assert "detailed_stats" in data

    def test_vqgan_metrics_export_csv(self, tmp_path):
        """Test export to CSV."""
        try:
            metrics = VQGANMetrics(data_range=1.0, spatial_dims=3)
        except TypeError:
            pytest.skip("MONAI SSIMMetric API changed")

        predictions = torch.randn(2, 1, 8, 8, 8)
        targets = torch.randn(2, 1, 8, 8, 8)

        metrics.update(predictions, targets)
        csv_path = tmp_path / "metrics.csv"
        metrics.export_csv(str(csv_path))

        assert csv_path.exists()
        import csv

        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) >= 2  # Header + at least one data row


class TestAdamOptimizerFactory:
    """Tests for AdamOptimizerFactory."""

    def test_adam_factory_creation(self):
        """Test creating AdamOptimizerFactory."""
        factory = AdamOptimizerFactory(
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )
        assert factory.lr == 1e-3
        assert factory.weight_decay == 1e-4

    def test_adam_factory_create(self):
        """Test creating Adam optimizer."""
        factory = AdamOptimizerFactory(lr=1e-3)
        params = [torch.randn(10, 10)]

        optimizer = factory.create(iter(params))

        assert optimizer.param_groups[0]["lr"] == 1e-3


class TestSGDOptimizerFactory:
    """Tests for SGDOptimizerFactory."""

    def test_sgd_factory_creation(self):
        """Test creating SGDOptimizerFactory."""
        factory = SGDOptimizerFactory(
            lr=1e-2,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True,
        )
        assert factory.lr == 1e-2
        assert factory.momentum == 0.9
        assert factory.nesterov is True

    def test_sgd_factory_create(self):
        """Test creating SGD optimizer."""
        factory = SGDOptimizerFactory(lr=1e-2, momentum=0.9)
        params = [torch.randn(10, 10)]

        optimizer = factory.create(iter(params))

        assert optimizer.param_groups[0]["lr"] == 1e-2
        assert optimizer.param_groups[0]["momentum"] == 0.9


class TestAdamWOptimizerFactory:
    """Tests for AdamWOptimizerFactory."""

    def test_adamw_factory_creation(self):
        """Test creating AdamWOptimizerFactory."""
        factory = AdamWOptimizerFactory(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        assert factory.lr == 1e-3
        assert factory.weight_decay == 0.01

    def test_adamw_factory_create(self):
        """Test creating AdamW optimizer."""
        factory = AdamWOptimizerFactory(lr=1e-3, weight_decay=0.01)
        params = [torch.randn(10, 10)]

        optimizer = factory.create(iter(params))

        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["weight_decay"] == 0.01


class TestVQGANOptimizerFactory:
    """Tests for VQGANOptimizerFactory."""

    def test_vqgan_optimizer_factory_creation(self):
        """Test creating VQGANOptimizerFactory."""
        factory = VQGANOptimizerFactory(
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.5, 0.9),
        )
        assert factory.lr == 1e-4

    def test_vqgan_optimizer_factory_implements_gan_optimizer_factory(self):
        """VQGANOptimizerFactory must implement GANOptimizerFactory."""
        from maskgit3d.domain.interfaces import GANOptimizerFactory
        from maskgit3d.infrastructure.training.strategies import VQGANOptimizerFactory

        factory = VQGANOptimizerFactory()
        assert isinstance(factory, GANOptimizerFactory)

    def test_vqgan_optimizer_factory_creates_dual_optimizers(self):
        """VQGANOptimizerFactory.create() must return two optimizers with same lr."""
        from maskgit3d.infrastructure.training.strategies import VQGANOptimizerFactory

        factory = VQGANOptimizerFactory(lr=1e-4)
        params_g = [torch.nn.Parameter(torch.randn(10, 10))]
        params_d = [torch.nn.Parameter(torch.randn(5, 5))]
        opt_g, opt_d = factory.create(iter(params_g), iter(params_d))
        assert opt_g.param_groups[0]["lr"] == 1e-4  # type: ignore[union-attr]
        assert opt_d.param_groups[0]["lr"] == 1e-4  # type: ignore[union-attr]  # Same lr

    def test_vqgan_optimizer_factory_creates_generator_only(self):
        """VQGANOptimizerFactory.create() can return just generator optimizer."""
        from maskgit3d.infrastructure.training.strategies import VQGANOptimizerFactory

        factory = VQGANOptimizerFactory(lr=1e-4)
        params_g = [torch.nn.Parameter(torch.randn(10, 10))]
        opt_g, opt_d = factory.create(iter(params_g), None)
        assert opt_g.param_groups[0]["lr"] == 1e-4  # type: ignore[union-attr]
        assert opt_d is None


class TestMaskGITTrainingStrategy:
    """Tests for MaskGITTrainingStrategy."""

    def test_maskgit_strategy_creation(self):
        """Test creating MaskGITTrainingStrategy."""
        strategy = MaskGITTrainingStrategy(
            mask_schedule_type="cosine",
            reconstruction_weight=1.0,
        )
        assert strategy.mask_schedule_type == "cosine"

    def test_maskgit_validate_step(self):
        """Test validation step."""
        strategy = MaskGITTrainingStrategy(mask_schedule_type="cosine")
        mock_model = MagicMock()

        # Mock model(x) - returns reconstruction
        mock_model.return_value = torch.randn(2, 1, 8, 8, 8)

        # Mock model.encode_tokens(x) - returns token indices
        mock_model.encode_tokens.return_value = torch.randint(0, 1024, (2, 1, 4, 4, 4))

        # Mock model.decode_tokens(tokens) - returns reconstruction
        # validate_step calls: x_rec = decode_tokens(encode_tokens(x))
        mock_model.decode_tokens.return_value = torch.randn(2, 1, 8, 8, 8)
        mock_model.encode_tokens.return_value = torch.randint(0, 1024, (2, 1, 4, 4, 4))

        # Mock model.transformer.encode() - returns logits for token prediction
        # Shape should be (B, num_tokens, codebook_size) for argmax to work
        mock_transformer = MagicMock()
        mock_transformer.encode.return_value = torch.randn(2, 64, 1024)
        mock_model.transformer = mock_transformer

        batch = (torch.randn(2, 1, 8, 8, 8),)

        metrics = strategy.validate_step(mock_model, batch)

        assert "val_loss" in metrics
        assert "val_token_acc" in metrics
        # Metrics should be Python floats
        assert isinstance(metrics["val_loss"], float)
        assert isinstance(metrics["val_token_acc"], float)

    def test_maskgit_train_step_backprop_updates_parameters(self):
        """Train step should backpropagate through live loss tensor."""
        strategy = MaskGITTrainingStrategy(mask_ratio=0.5)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def encode_tokens(self, x):
                return torch.zeros((x.shape[0], 1, 1, 4), dtype=torch.long, device=x.device)

            @property
            def transformer(self):
                outer = self

                class _T:
                    @staticmethod
                    def forward(tokens_flat, mask_indices):
                        logits = torch.zeros(
                            tokens_flat.shape[0],
                            tokens_flat.shape[1],
                            8,
                            device=tokens_flat.device,
                        )
                        logits[..., 0] = outer.weight
                        return logits

                return _T()

        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        batch = (torch.randn(2, 1, 4, 4, 4),)

        before = model.weight.detach().clone()
        strategy.train_step(model, batch, optimizer)  # type: ignore[arg-type]
        after = model.weight.detach().clone()

        assert not torch.equal(before, after)


class TestMaskGITInference:
    """Tests for MaskGITInference."""

    def test_maskgit_inference_creation(self):
        """Test creating MaskGITInference."""
        inference = MaskGITInference(
            mode="generate",
            num_iterations=12,
            temperature=1.0,
        )
        assert inference.mode == "generate"
        assert inference.num_iterations == 12

    def test_maskgit_inference_reconstruct(self):
        """Test reconstruction mode."""
        inference = MaskGITInference(mode="reconstruct")
        # MaskGITInference.reconstruct calls encode_tokens(batch) then decode_tokens(tokens)
        mock_model = MagicMock()
        mock_model.encode_tokens.return_value = torch.randint(0, 1024, (2, 1, 4, 4, 4))
        mock_model.decode_tokens.return_value = torch.randn(2, 1, 8, 8, 8)

        batch = torch.randn(2, 1, 8, 8, 8)
        result = inference.predict(mock_model, batch)

        assert result.shape == batch.shape

    def test_maskgit_inference_post_process(self):
        """Test post-processing."""
        inference = MaskGITInference()
        predictions = torch.randn(2, 1, 8, 8, 8)

        result = inference.post_process(predictions)

        assert "volumes" in result
        assert result["volumes"].shape == predictions.shape

        assert "volumes" in result
        assert result["volumes"].shape == predictions.shape


def test_strategy_annotations_use_specific_model_interfaces():
    """Verify training strategies use specific model interfaces in type hints."""
    import inspect

    from maskgit3d.infrastructure.training.strategies import (
        MaskGITTrainingStrategy,
        VQGANTrainingStrategy,
    )

    vq_sig = inspect.signature(VQGANTrainingStrategy.train_step)
    mg_sig = inspect.signature(MaskGITTrainingStrategy.train_step)

    # VQGANTrainingStrategy should use VQModelInterface
    vq_model_annotation = str(vq_sig.parameters["model"].annotation)
    assert "VQModelInterface" in vq_model_annotation, (
        f"Expected VQModelInterface in annotation, got: {vq_model_annotation}"
    )

    # MaskGITTrainingStrategy should use MaskGITModelInterface
    mg_model_annotation = str(mg_sig.parameters["model"].annotation)
    assert "MaskGITModelInterface" in mg_model_annotation, (
        f"Expected MaskGITModelInterface in annotation, got: {mg_model_annotation}"
    )
