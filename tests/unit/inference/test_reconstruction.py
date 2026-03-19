"""Tests for shared VQVAE reconstruction helpers."""

import torch

from src.maskgit3d.inference.reconstruction import VQVAEReconstructor


class DummyQuantizer:
    def __init__(self) -> None:
        self.last_indices: torch.Tensor | None = None

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        self.last_indices = indices.clone()
        return indices.unsqueeze(1).float()


class DummyVQVAE:
    def __init__(self) -> None:
        self.quantizer = DummyQuantizer()
        self.decode_inputs: list[torch.Tensor] = []
        self.forward_inputs: list[torch.Tensor] = []

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_inputs.append(x.clone())
        return x + 10.0, torch.tensor(0.0)

    def encode(self, patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.arange(
            patch.shape[2] * patch.shape[3] * patch.shape[4], dtype=torch.float32
        )
        indices = indices.view(1, patch.shape[2], patch.shape[3], patch.shape[4]).repeat(
            patch.shape[0], 1, 1, 1
        )
        return patch, torch.tensor(0.0), indices.long()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(latent.clone())
        return latent + 0.5

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


def test_reconstruct_without_sliding_window_uses_direct_forward_path() -> None:
    reconstructor = VQVAEReconstructor(sliding_window={"enabled": False}, downsampling_factor=4)
    model = DummyVQVAE()
    x = torch.randn(1, 1, 8, 8, 8)

    result = reconstructor.reconstruct(model, x)

    assert torch.allclose(result, x + 10.0)
    assert len(model.forward_inputs) == 1
    assert model.quantizer.last_indices is None


def test_reconstruct_with_sliding_window_pads_and_crops_latent_indices() -> None:
    indices_padded = torch.arange(64, dtype=torch.float32).view(1, 1, 4, 4, 4)
    inferer = RecordingInferer(output=indices_padded)
    reconstructor = VQVAEReconstructor(
        sliding_window={"enabled": True, "roi_size": [8, 8, 8], "sw_batch_size": 2},
        downsampling_factor=4,
        inferer=inferer,
    )
    model = DummyVQVAE()
    x = torch.randn(1, 1, 10, 9, 11)

    result = reconstructor.reconstruct(model, x)

    assert inferer.calls[0].shape == (1, 1, 12, 12, 12)
    assert model.quantizer.last_indices is not None
    assert model.quantizer.last_indices.shape == (1, 2, 2, 2)
    expected_latent = indices_padded[0, 0, :2, :2, :2].unsqueeze(0)
    assert torch.equal(model.quantizer.last_indices, expected_latent.long())
    assert torch.allclose(result, expected_latent.unsqueeze(1) + 0.5)


def test_reconstruct_with_large_latent_uses_decoder_sliding_window(monkeypatch) -> None:
    RecordingDecoderInferer.instances.clear()
    monkeypatch.setattr(
        "src.maskgit3d.inference.reconstruction.SlidingWindowInferer",
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

    result = reconstructor.reconstruct(model, x)

    assert len(RecordingDecoderInferer.instances) == 1
    decoder_inferer = RecordingDecoderInferer.instances[0]
    assert decoder_inferer.kwargs["roi_size"] == (2, 2, 2)
    assert decoder_inferer.kwargs["sw_batch_size"] == 3
    assert decoder_inferer.calls[0].shape == (1, 1, 4, 4, 4)
    assert torch.allclose(result, decoder_inferer.calls[0] + 0.5)
