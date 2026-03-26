"""Tests for InContextTrainingSteps.

Following TDD: Write failing tests first, then implement minimal code to pass.
"""

from __future__ import annotations

from typing import Any

import torch

from src.maskgit3d.training.incontext_steps import InContextTrainingSteps


class RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []

    def __call__(self, name: str, value: Any, **kwargs: Any) -> None:
        self.calls.append((name, value, kwargs))


class MockInContextMaskGIT:
    def __init__(
        self,
        loss_value: float = 2.5,
        mask_acc: float = 0.75,
        generated_value: float = 5.0,
    ) -> None:
        self._loss_value = loss_value
        self._mask_acc = mask_acc
        self._generated_value = generated_value
        self.compute_loss_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []

    def compute_loss(
        self,
        context_images: list[torch.Tensor],
        context_modality_ids: list[int],
        target_image: torch.Tensor,
        target_modality_id: int,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        self.compute_loss_calls.append(
            {
                "context_images": [t.shape for t in context_images],
                "context_modality_ids": context_modality_ids,
                "target_image_shape": target_image.shape,
                "target_modality_id": target_modality_id,
                "mask_ratio": mask_ratio,
            }
        )
        loss = torch.tensor(self._loss_value, requires_grad=True)
        metrics = {"mask_acc": self._mask_acc, "mask_ratio": 0.5}
        return loss, metrics

    def generate(
        self,
        context_images: list[torch.Tensor],
        context_modality_ids: list[int],
        target_modality_id: int,
        target_shape: tuple[int, int, int, int],
        temperature: float = 1.0,
        num_iterations: int = 12,
    ) -> torch.Tensor:
        self.generate_calls.append(
            {
                "context_images": [t.shape for t in context_images],
                "context_modality_ids": context_modality_ids,
                "target_modality_id": target_modality_id,
                "target_shape": target_shape,
                "temperature": temperature,
                "num_iterations": num_iterations,
            }
        )
        B, D, H, W = target_shape
        return torch.full((B, 1, D, H, W), self._generated_value)


def test_training_step_returns_correct_loss() -> None:
    model = MockInContextMaskGIT(loss_value=3.0)
    steps = InContextTrainingSteps(model=model)

    batch = {
        "context_images": [torch.randn(2, 1, 16, 16, 16)],
        "context_modality_ids": [0],
        "target_image": torch.randn(2, 1, 16, 16, 16),
        "target_modality_id": 1,
    }

    def encode_context_fn(images: list[torch.Tensor], ids: list[int]) -> list[torch.Tensor]:
        return images

    def encode_target_fn(image: torch.Tensor, modality_id: int) -> torch.Tensor:
        return image

    result = steps.training_step(
        batch=batch,
        encode_context_fn=encode_context_fn,
        encode_target_fn=encode_target_fn,
    )

    assert isinstance(result, dict)
    assert "loss" in result
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].item() == 3.0


def test_training_step_uses_encode_functions() -> None:
    model = MockInContextMaskGIT()
    steps = InContextTrainingSteps(model=model)

    context_images = [torch.randn(2, 1, 16, 16, 16)]
    target_image = torch.randn(2, 1, 16, 16, 16)

    batch = {
        "context_images": context_images,
        "context_modality_ids": [0],
        "target_image": target_image,
        "target_modality_id": 1,
    }

    context_encoded: list[torch.Tensor] = []
    target_encoded: list[torch.Tensor] = []

    def encode_context_fn(images: list[torch.Tensor], ids: list[int]) -> list[torch.Tensor]:
        context_encoded.extend(images)
        return images

    def encode_target_fn(image: torch.Tensor, modality_id: int) -> torch.Tensor:
        target_encoded.append(image)
        return image

    steps.training_step(
        batch=batch,
        encode_context_fn=encode_context_fn,
        encode_target_fn=encode_target_fn,
    )

    assert len(context_encoded) == 1
    assert len(target_encoded) == 1
    assert model.compute_loss_calls[0]["context_images"] == [(2, 1, 16, 16, 16)]
    assert model.compute_loss_calls[0]["target_image_shape"] == (2, 1, 16, 16, 16)


def test_validation_step_returns_correct_dict() -> None:
    model = MockInContextMaskGIT(loss_value=1.5, generated_value=7.0)
    steps = InContextTrainingSteps(model=model)

    batch = {
        "context_images": [torch.randn(2, 1, 16, 16, 16)],
        "context_modality_ids": [0],
        "target_image": torch.randn(2, 1, 16, 16, 16),
        "target_modality_id": 1,
    }

    result = steps.validation_step(
        batch=batch,
        encode_context_fn=lambda images, ids: images,
        encode_target_fn=lambda image, modality_id: image,
    )

    assert isinstance(result, dict)
    assert "context_images" in result
    assert "target_image" in result
    assert "generated_image" in result
    assert "loss" in result
    assert result["loss"].item() == 1.5
    assert result["generated_image"].shape == result["target_image"].shape


def test_predict_step_returns_generated_images() -> None:
    model = MockInContextMaskGIT(generated_value=9.0)
    steps = InContextTrainingSteps(model=model)

    batch = {
        "context_images": [torch.randn(2, 1, 16, 16, 16)],
        "context_modality_ids": [0],
        "target_image": torch.randn(2, 1, 16, 16, 16),
        "target_modality_id": 1,
    }

    result = steps.predict_step(
        batch=batch,
        encode_context_fn=lambda images, ids: images,
        encode_target_fn=lambda image, modality_id: image,
    )

    assert isinstance(result, dict)
    assert "context_images" in result
    assert "target_image" in result
    assert "generated_image" in result
    assert "loss" not in result
    assert result["generated_image"].shape == result["target_image"].shape


def test_loss_is_differentiable() -> None:
    model = MockInContextMaskGIT()
    steps = InContextTrainingSteps(model=model)

    batch = {
        "context_images": [torch.randn(2, 1, 16, 16, 16)],
        "context_modality_ids": [0],
        "target_image": torch.randn(2, 1, 16, 16, 16),
        "target_modality_id": 1,
    }

    result = steps.training_step(
        batch=batch,
        encode_context_fn=lambda images, ids: images,
        encode_target_fn=lambda image, modality_id: image,
    )

    loss = result["loss"]
    assert loss.requires_grad, "Loss should require gradients"
    loss.backward()


def test_metrics_are_computed_correctly() -> None:
    model = MockInContextMaskGIT(mask_acc=0.85)
    steps = InContextTrainingSteps(model=model)

    batch = {
        "context_images": [torch.randn(2, 1, 16, 16, 16)],
        "context_modality_ids": [0],
        "target_image": torch.randn(2, 1, 16, 16, 16),
        "target_modality_id": 1,
    }

    steps.training_step(
        batch=batch,
        encode_context_fn=lambda images, ids: images,
        encode_target_fn=lambda image, modality_id: image,
    )

    assert len(model.compute_loss_calls) == 1
    call = model.compute_loss_calls[0]
    assert call["context_modality_ids"] == [0]
    assert call["target_modality_id"] == 1


def test_multiple_context_images() -> None:
    model = MockInContextMaskGIT()
    steps = InContextTrainingSteps(model=model)

    batch = {
        "context_images": [
            torch.randn(2, 1, 16, 16, 16),
            torch.randn(2, 1, 16, 16, 16),
        ],
        "context_modality_ids": [0, 1],
        "target_image": torch.randn(2, 1, 16, 16, 16),
        "target_modality_id": 2,
    }

    result = steps.training_step(
        batch=batch,
        encode_context_fn=lambda images, ids: images,
        encode_target_fn=lambda image, modality_id: image,
    )

    assert "loss" in result
    assert model.compute_loss_calls[0]["context_modality_ids"] == [0, 1]
    assert model.compute_loss_calls[0]["target_modality_id"] == 2


def test_validation_step_logs_with_log_fn() -> None:
    logger = RecordingLogger()
    model = MockInContextMaskGIT()
    steps = InContextTrainingSteps(model=model, log_fn=logger)

    batch = {
        "context_images": [torch.randn(2, 1, 16, 16, 16)],
        "context_modality_ids": [0],
        "target_image": torch.randn(2, 1, 16, 16, 16),
        "target_modality_id": 1,
    }

    steps.validation_step(
        batch=batch,
        encode_context_fn=lambda images, ids: images,
        encode_target_fn=lambda image, modality_id: image,
    )


def test_predict_step_generation_params() -> None:
    model = MockInContextMaskGIT()
    steps = InContextTrainingSteps(model=model)

    context_images = [torch.randn(2, 1, 16, 16, 16)]
    target_image = torch.randn(2, 1, 16, 16, 16)

    batch = {
        "context_images": context_images,
        "context_modality_ids": [0],
        "target_image": target_image,
        "target_modality_id": 1,
    }

    steps.predict_step(
        batch=batch,
        encode_context_fn=lambda images, ids: images,
        encode_target_fn=lambda image, modality_id: image,
    )

    assert len(model.generate_calls) == 1
    call = model.generate_calls[0]
    assert call["target_modality_id"] == 1
    assert call["target_shape"] == (2, 16, 16, 16)
