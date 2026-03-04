"""Extended tests for strategies module to improve coverage."""

import pytest
import torch
import torch.nn as nn

from maskgit3d.infrastructure.training.strategies import (
    AdamOptimizerFactory,
    AdamWOptimizerFactory,
    MaskGITInference,
    MixedPrecisionTrainer,
    SGDOptimizerFactory,
    VQGANInference,
    VQGANOptimizerFactory,
)


class TestMixedPrecisionTrainerExtended:
    """Extended tests for MixedPrecisionTrainer."""

    def test_init_disabled(self):
        """Test initialization when disabled."""
        trainer = MixedPrecisionTrainer(enabled=False)
        assert trainer.enabled is False

    def test_autocast_context_disabled(self):
        """Test autocast context when disabled."""
        trainer = MixedPrecisionTrainer(enabled=False)
        context = trainer.autocast_context()
        assert context is not None

    def test_scale_loss_not_enabled(self):
        """Test scale_loss when not enabled."""
        trainer = MixedPrecisionTrainer(enabled=False)
        loss = torch.tensor(1.0)
        scaled = trainer.scale_loss(loss)
        assert scaled == loss

    def test_state_dict_disabled(self):
        """Test state_dict when disabled."""
        trainer = MixedPrecisionTrainer(enabled=False)
        state = trainer.state_dict()
        assert state["enabled"] is False
        assert "scaler" not in state

    def test_load_state_dict_disabled(self):
        """Test load_state_dict when disabled."""
        trainer = MixedPrecisionTrainer(enabled=False)
        state = {"enabled": False, "dtype": "float16"}
        trainer.load_state_dict(state)
        assert trainer.enabled is False


class TestAdamOptimizerFactory:
    """Tests for AdamOptimizerFactory."""

    def test_create(self):
        """Test creating Adam optimizer."""
        factory = AdamOptimizerFactory(lr=1e-3, weight_decay=1e-5)
        model = nn.Linear(10, 10)
        optimizer = factory.create(model.parameters())

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == 1e-3
        assert optimizer.defaults["weight_decay"] == 1e-5


class TestSGDOptimizerFactory:
    """Tests for SGDOptimizerFactory."""

    def test_create(self):
        """Test creating SGD optimizer."""
        factory = SGDOptimizerFactory(lr=1e-2, momentum=0.9, nesterov=True)
        model = nn.Linear(10, 10)
        optimizer = factory.create(model.parameters())

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults["lr"] == 1e-2
        assert optimizer.defaults["momentum"] == 0.9
        assert optimizer.defaults["nesterov"] is True


class TestAdamWOptimizerFactory:
    """Tests for AdamWOptimizerFactory."""

    def test_create(self):
        """Test creating AdamW optimizer."""
        factory = AdamWOptimizerFactory(lr=1e-4, weight_decay=0.01)
        model = nn.Linear(10, 10)
        optimizer = factory.create(model.parameters())

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 1e-4
        assert optimizer.defaults["weight_decay"] == 0.01


class TestVQGANOptimizerFactory:
    """Tests for VQGANOptimizerFactory."""

    def test_create_with_discriminator(self):
        """Test creating optimizers with discriminator (same lr for G and D)."""
        factory = VQGANOptimizerFactory(lr=1e-4)
        gen_model = nn.Linear(10, 10)
        disc_model = nn.Linear(10, 1)

        opt_g, opt_d = factory.create(gen_model.parameters(), disc_model.parameters())

        assert isinstance(opt_g, torch.optim.AdamW)
        assert isinstance(opt_d, torch.optim.AdamW)
        assert opt_g.defaults["lr"] == 1e-4
        assert opt_d.defaults["lr"] == 1e-4

    def test_create_without_discriminator(self):
        """Test creating optimizers without discriminator."""
        factory = VQGANOptimizerFactory()
        gen_model = nn.Linear(10, 10)

        opt_g, opt_d = factory.create(gen_model.parameters(), None)

        assert isinstance(opt_g, torch.optim.AdamW)
        assert opt_d is None


class TestVQGANInference:
    """Tests for VQGANInference."""

    @pytest.fixture
    def mock_vqmodel(self):
        """Create a mock VQ model."""

        class MockVQModel:
            def __init__(self):
                self._codebook_size = 512
                self._latent_shape = (256, 4, 4, 4)

            @property
            def codebook_size(self):
                return self._codebook_size

            @property
            def latent_shape(self):
                return self._latent_shape

            def encode(self, x):
                batch_size = x.shape[0]
                z = torch.randn(batch_size, 256, 4, 4, 4)
                return z, torch.tensor(0.1), (None, None, None)

            def decode(self, quant):
                batch_size = quant.shape[0]
                return torch.randn(batch_size, 1, 16, 16, 16)

            def decode_code(self, codes):
                batch_size = codes.shape[0]
                return torch.randn(batch_size, 1, 16, 16, 16)

            def eval(self):
                pass

            def train(self):
                pass

        return MockVQModel()

    def test_predict_reconstruct(self, mock_vqmodel):
        """Test predict with reconstruct mode."""
        inference = VQGANInference(mode="reconstruct")
        batch = torch.randn(2, 1, 16, 16, 16)

        result = inference.predict(mock_vqmodel, batch)

        assert result.shape[0] == 2

    def test_predict_generate(self, mock_vqmodel):
        """Test predict with generate mode."""
        inference = VQGANInference(mode="generate", num_samples=2)
        batch = torch.randn(2, 1, 16, 16, 16)

        result = inference.predict(mock_vqmodel, batch)

        assert result.shape[0] == 2

    def test_predict_decode_code(self, mock_vqmodel):
        """Test predict with decode_code mode."""
        inference = VQGANInference(mode="decode_code")
        batch = torch.randint(0, 512, (2, 4, 4, 4))

        result = inference.predict(mock_vqmodel, batch)

        assert result.shape[0] == 2

    def test_predict_unknown_mode(self, mock_vqmodel):
        """Test predict with unknown mode raises ValueError."""
        inference = VQGANInference(mode="unknown")
        batch = torch.randn(2, 1, 16, 16, 16)

        with pytest.raises(ValueError, match="Unknown mode"):
            inference.predict(mock_vqmodel, batch)

    def test_post_process(self):
        """Test post_process method."""
        inference = VQGANInference()
        predictions = torch.randn(2, 1, 16, 16, 16)

        result = inference.post_process(predictions)

        assert "images" in result


class TestMaskGITInference:
    """Tests for MaskGITInference."""

    @pytest.fixture
    def mock_maskgit_model(self):
        """Create a mock MaskGIT model."""

        class MockMaskGITModel:
            def __init__(self):
                self._latent_shape = (1, 4, 4, 4)

            @property
            def latent_shape(self):
                return self._latent_shape

            def encode_tokens(self, x):
                batch_size = x.shape[0]
                return torch.randint(0, 512, (batch_size, 4, 4, 4))

            def decode_tokens(self, tokens):
                batch_size = tokens.shape[0]
                return torch.randn(batch_size, 1, 16, 16, 16)

            def generate(self, shape, temperature, num_iterations):
                return torch.randn(shape[0], 1, 16, 16, 16)

            def eval(self):
                pass

        return MockMaskGITModel()

    def test_predict_reconstruct(self, mock_maskgit_model):
        """Test predict with reconstruct mode."""
        inference = MaskGITInference(mode="reconstruct")
        batch = torch.randn(2, 1, 16, 16, 16)

        result = inference.predict(mock_maskgit_model, batch)

        assert result.shape[0] == 2

    def test_predict_generate(self, mock_maskgit_model):
        """Test predict with generate mode."""
        inference = MaskGITInference(mode="generate", num_iterations=2)
        batch = torch.randn(2, 1, 16, 16, 16)

        result = inference.predict(mock_maskgit_model, batch)

        assert result.shape[0] == 2

    def test_predict_unknown_mode(self, mock_maskgit_model):
        """Test predict with unknown mode raises ValueError."""
        inference = MaskGITInference(mode="unknown")
        batch = torch.randn(2, 1, 16, 16, 16)

        with pytest.raises(ValueError, match="Unknown mode"):
            inference.predict(mock_maskgit_model, batch)

    def test_post_process(self):
        """Test post_process method."""
        inference = MaskGITInference()
        predictions = torch.randn(2, 1, 16, 16, 16)

        result = inference.post_process(predictions)

        assert "volumes" in result
