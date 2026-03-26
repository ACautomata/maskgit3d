"""Tests for shared VQVAE reconstruction helpers."""

import pytest
import torch

from maskgit3d.inference.reconstruction import VQVAEReconstructor


class DummyQuantizer:
    def __init__(self, embedding_dim: int = 256) -> None:
        self.last_indices: torch.Tensor | None = None
        self.embedding_dim = embedding_dim

    def __call__(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, D, H, W = z_e.shape
        indices = torch.arange(D * H * W, dtype=torch.long).view(1, D, H, W).repeat(B, 1, 1, 1)
        z_q = z_e + 0.1
        return z_q, torch.tensor(0.0), indices

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        self.last_indices = indices.clone()
        return indices.unsqueeze(1).float().expand(-1, self.embedding_dim, -1, -1, -1)


class DummyVQVAE:
    def __init__(self, embedding_dim: int = 256) -> None:
        self.quantizer = DummyQuantizer(embedding_dim)
        self.decode_inputs: list[torch.Tensor] = []
        self.forward_inputs: list[torch.Tensor] = []
        self.embedding_dim = embedding_dim

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_inputs.append(x.clone())
        return x + 10.0, torch.tensor(0.0)

    def encode(
        self, patch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, D, H, W = patch.shape
        z_e = torch.randn(B, self.embedding_dim, D, H, W)
        z_q = z_e + 0.1
        indices = torch.arange(D * H * W, dtype=torch.long).view(1, D, H, W).repeat(B, 1, 1, 1)
        return z_q, torch.tensor(0.0), indices, z_e

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(latent.clone())
        return latent[:, :1, ...] + 0.5

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.decode_from_indices(indices)


class RecordingInferer:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self.calls: list[torch.Tensor] = []

    def __call__(self, x: torch.Tensor, fn):
        self.calls.append(x.clone())
        fn(x)
        return self.output.clone()


class RecordingDecoderInferer:
    instances: list["RecordingDecoderInferer"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls: list[torch.Tensor] = []
        RecordingDecoderInferer.instances.append(self)

    def __call__(self, x: torch.Tensor, fn):
        self.calls.append(x.clone())
        return fn(x)


def test_extract_input_tensor_supports_tensor_and_tuple_batches() -> None:
    reconstructor = VQVAEReconstructor(sliding_window={"enabled": False}, downsampling_factor=4)
    x = torch.randn(1, 1, 8, 8, 8)

    assert reconstructor.extract_input_tensor(x) is x
    assert reconstructor.extract_input_tensor((x, x.clone())) is x


def test_extract_input_tensor_supports_dict_batches() -> None:
    reconstructor = VQVAEReconstructor(sliding_window={"enabled": False}, downsampling_factor=4)
    x = torch.randn(2, 4, 128, 128, 128)

    result = reconstructor.extract_input_tensor({"image": x, "case_id": "test"})
    assert result is x

    result = reconstructor.extract_input_tensor({"image": x, "label": None})
    assert result is x


def test_extract_input_tensor_raises_on_invalid_input() -> None:
    reconstructor = VQVAEReconstructor(sliding_window={"enabled": False}, downsampling_factor=4)

    with pytest.raises(TypeError):
        reconstructor.extract_input_tensor({"label": torch.randn(1, 1, 8, 8, 8)})

    with pytest.raises(TypeError):
        reconstructor.extract_input_tensor("invalid")


def test_reconstruct_without_sliding_window_uses_direct_forward_path() -> None:
    reconstructor = VQVAEReconstructor(sliding_window={"enabled": False}, downsampling_factor=4)
    model = DummyVQVAE()
    x = torch.randn(1, 1, 8, 8, 8)

    result, vq_loss = reconstructor.reconstruct(model, x)

    assert torch.allclose(result, x + 10.0)
    assert len(model.forward_inputs) == 1
    assert model.quantizer.last_indices is None


class UpscalingDummyVQVAE(DummyVQVAE):
    def __init__(self, embedding_dim: int = 256, upsample_factor: int = 4) -> None:
        super().__init__(embedding_dim)
        self.upsample_factor = upsample_factor

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(latent.clone())
        B, C, D, H, W = latent.shape
        upsampled = latent[:, :1, ...].repeat(
            1, 1, self.upsample_factor, self.upsample_factor, self.upsample_factor
        )
        return upsampled + 0.5


def test_reconstruct_with_sliding_window_pads_and_crops_latent() -> None:
    z_e_padded = torch.randn(1, 256, 3, 3, 3)
    inferer = RecordingInferer(output=z_e_padded)
    reconstructor = VQVAEReconstructor(
        sliding_window={"enabled": True, "roi_size": [8, 8, 8], "sw_batch_size": 2},
        downsampling_factor=4,
        inferer=inferer,
    )
    model = UpscalingDummyVQVAE()
    x = torch.randn(1, 1, 10, 9, 11)

    result, vq_loss = reconstructor.reconstruct(model, x)

    assert inferer.calls[0].shape == (1, 1, 12, 12, 12)
    assert len(model.decode_inputs) >= 1
    assert result.shape == (1, 1, 10, 9, 11)


class WeightedAverageRecorder:
    """Records what values are being averaged in sliding window."""

    def __init__(self) -> None:
        self.encode_fn_outputs: list[torch.Tensor] = []
        self.final_output: torch.Tensor | None = None

    def __call__(self, x: torch.Tensor, fn):
        # Record what the encode_fn returns for each patch
        result = fn(x)
        self.encode_fn_outputs.append(result.clone())
        # Simulate Gaussian weighted average - just return mean for simplicity
        # The real MONAI SlidingWindowInferer does weighted average in overlap regions
        self.final_output = result
        return result


def test_sliding_window_averages_in_latent_space_not_indices() -> None:
    """Sliding window weighted average must be done in continuous latent space, not discrete indices.

    This is CRITICAL: averaging discrete indices (e.g., (100 + 200) / 2 = 150) has no semantic meaning
    in codebook space. We must average continuous latent vectors (z_q) and THEN quantize.
    """
    recorder = WeightedAverageRecorder()
    reconstructor = VQVAEReconstructor(
        sliding_window={"enabled": True, "roi_size": [8, 8, 8], "sw_batch_size": 1},
        downsampling_factor=4,
        inferer=recorder,
    )
    model = DummyVQVAE()
    x = torch.randn(1, 1, 16, 16, 16)

    _ = reconstructor.reconstruct(model, x)

    # The encode_fn should return continuous latent vectors (z_q), NOT discrete indices
    # z_q has shape [B, C, D, H, W] where C is the latent channel dimension (e.g., 256)
    # indices has shape [B, D, H, W] - no channel dimension
    for output in recorder.encode_fn_outputs:
        # If sliding window operates on indices, output will have shape [B, 1, D, H, W]
        # (because indices.unsqueeze(1) was called)
        # If sliding window operates on latent, output will have shape [B, C, D, H, W]
        # where C is the embedding/latent dimension
        assert output.dim() == 5, f"Expected 5D tensor, got {output.dim()}D"
        # The channel dimension should be > 1 for latent space (e.g., 256 channels)
        # NOT 1 which would indicate unsqueezed indices
        assert output.shape[1] > 1, (
            f"encode_fn returned shape {output.shape} - appears to be indices (C=1). "
            f"Sliding window MUST operate on continuous latent space (z_q), not discrete indices. "
            f"Expected C > 1 (e.g., 256 for embedding dimension)."
        )


def test_reconstruct_with_large_latent_uses_decoder_sliding_window(monkeypatch) -> None:
    RecordingDecoderInferer.instances.clear()
    monkeypatch.setattr(
        "maskgit3d.inference.reconstruction.SlidingWindowInferer",
        RecordingDecoderInferer,
    )
    indices_padded = torch.arange(64, dtype=torch.float32).view(1, 1, 4, 4, 4)
    inferer = RecordingInferer(output=indices_padded)
    reconstructor = VQVAEReconstructor(
        sliding_window={"enabled": True, "roi_size": [4, 4, 4], "sw_batch_size": 3, "overlap": 0.1},
        downsampling_factor=2,
        inferer=inferer,
    )
    model = DummyVQVAE()
    x = torch.randn(1, 1, 8, 8, 8)

    result, vq_loss = reconstructor.reconstruct(model, x)

    assert len(RecordingDecoderInferer.instances) == 1
    decoder_inferer = RecordingDecoderInferer.instances[0]
    assert decoder_inferer.kwargs["roi_size"] == (2, 2, 2)
    assert decoder_inferer.kwargs["sw_batch_size"] == 3
    assert decoder_inferer.calls[0].shape == (1, 1, 4, 4, 4)
    assert torch.allclose(result, decoder_inferer.calls[0] + 0.5)
