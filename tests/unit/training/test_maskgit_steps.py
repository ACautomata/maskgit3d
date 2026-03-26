"""Tests for extracted MaskGIT training steps."""

from __future__ import annotations

from typing import Any

import torch

from maskgit3d.training.maskgit_steps import MaskGITTrainingSteps


class RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []

    def __call__(self, name: str, value: Any, **kwargs: Any) -> None:
        self.calls.append((name, value, kwargs))


class RecordingMaskScheduler:
    def __init__(self, mask_ratio: float = 0.6) -> None:
        self.mask_ratio = mask_ratio
        self.calls = 0

    def sample_mask_ratio(self) -> float:
        self.calls += 1
        return self.mask_ratio


class RecordingTransformer:
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls: list[tuple[torch.Tensor, float]] = []

    def predict_masked(
        self, tokens_flat: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.calls.append((tokens_flat.clone(), mask_ratio))
        masked_targets = torch.tensor([0, 1], dtype=torch.long)
        masked_logits = torch.tensor(
            [[4.0, 0.5, -1.0], [0.25, 3.0, -2.0]],
            dtype=torch.float32,
        )
        mask = torch.tensor([True, True])
        return masked_logits, masked_targets, mask

    def parameters(self):
        return [self.weight]


class RecordingMaskGIT:
    def __init__(self, generated_offset: float = 5.0) -> None:
        self.mask_scheduler = RecordingMaskScheduler()
        self.transformer = RecordingTransformer()
        self.generated_offset = generated_offset
        self.generate_calls: list[dict[str, Any]] = []

    def generate(self, shape: torch.Size, temperature: float, num_iterations: int) -> torch.Tensor:
        self.generate_calls.append(
            {
                "shape": shape,
                "temperature": temperature,
                "num_iterations": num_iterations,
            }
        )
        return torch.full(shape, self.generated_offset, dtype=torch.float32)

    def parameters(self):
        return self.transformer.parameters()


def test_compute_masked_loss_returns_metrics_and_uses_explicit_mask_ratio() -> None:
    service = MaskGITTrainingSteps(maskgit=RecordingMaskGIT())
    tokens = torch.randint(0, 8, (2, 2, 2, 2))

    loss, raw_data = service.compute_masked_loss(tokens, mask_ratio=0.25)

    assert isinstance(loss, torch.Tensor)
    assert service.maskgit.transformer.calls[0][0].shape == (2, 8)
    assert service.maskgit.transformer.calls[0][1] == 0.25
    assert raw_data == {"correct": 2, "total": 2, "mask_ratio": 0.25}


def test_training_step_returns_loss_dict() -> None:
    service = MaskGITTrainingSteps(maskgit=RecordingMaskGIT())
    batch = [torch.randn(1, 1, 8, 8, 8)]
    captured: list[torch.Tensor] = []

    def encode_images_to_tokens(x: torch.Tensor) -> torch.Tensor:
        captured.append(x)
        return torch.randint(0, 16, (1, 2, 2, 2))

    outputs = service.training_step(
        batch=batch,
        encode_images_to_tokens_fn=encode_images_to_tokens,
    )

    assert captured[0].shape == (1, 1, 8, 8, 8)
    assert isinstance(outputs, dict)
    assert "loss" in outputs
    assert isinstance(outputs["loss"], torch.Tensor)


def test_validation_step_returns_raw_and_model_outputs() -> None:
    logger = RecordingLogger()
    service = MaskGITTrainingSteps(maskgit=RecordingMaskGIT(), log_fn=logger)

    outputs = service.validation_step(
        batch=torch.randn(1, 1, 8, 8, 8),
        encode_images_to_tokens_fn=lambda x: torch.randint(0, 16, (1, 2, 2, 2)),
    )

    assert logger.calls == []
    assert outputs["x_real"].shape == (1, 1, 8, 8, 8)
    assert outputs["generated_images"].shape == (1, 2, 2, 2)
    assert outputs["masked_logits"].shape == (2, 3)
    assert outputs["masked_targets"].shape == (2,)
    assert service.maskgit.generate_calls[0]["num_iterations"] == 12
