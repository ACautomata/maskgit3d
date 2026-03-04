"""Unit tests for MaskGIT training and sliding-window strategy paths."""

from unittest.mock import MagicMock, patch

import torch

from maskgit3d.infrastructure.training.strategies import (
    MaskGITTrainingStrategy,
    SlidingWindowVQGANInference,
    SlidingWindowVQGANLatentExtractor,
)


def test_maskgit_train_step_uses_compute_maskgit_loss_when_available():
    strategy = MaskGITTrainingStrategy(mask_schedule_type="cosine")
    model = MagicMock()
    optimizer = MagicMock()

    loss = torch.tensor(1.25, requires_grad=True)
    model.compute_maskgit_loss.return_value = (
        loss,
        {"loss": 1.25, "mask_acc": 0.5, "mask_ratio": 0.3},
    )

    metrics = strategy.train_step(model, (torch.randn(2, 1, 4, 4, 4),), optimizer)

    model.compute_maskgit_loss.assert_called_once()
    assert metrics["loss"] == 1.25
    optimizer.step.assert_called_once()
    optimizer.zero_grad.assert_called_once()


def test_maskgit_train_step_manual_path_with_transformer_forward():
    strategy = MaskGITTrainingStrategy(mask_schedule_type="cosine")
    optimizer = MagicMock()

    model = MagicMock()
    model.compute_maskgit_loss = None
    tokens = torch.tensor([[[[0, 1], [2, 3]]]], dtype=torch.long)
    model.encode_tokens.return_value = tokens

    logits = torch.randn(1, 4, 8, requires_grad=True)
    model.transformer = MagicMock()
    model.transformer.forward.return_value = logits

    metrics = strategy.train_step(model, (torch.randn(1, 1, 2, 2, 2),), optimizer)

    model.transformer.forward.assert_called_once()
    assert 0.0 <= metrics["mask_acc"] <= 1.0
    assert 0.0 < metrics["mask_ratio"] <= 1.0


def test_maskgit_train_step_manual_path_masks_at_least_one_token_per_sample():
    strategy = MaskGITTrainingStrategy(mask_schedule_type="cosine")
    optimizer = MagicMock()

    model = MagicMock()
    model.compute_maskgit_loss = None
    model.encode_tokens.return_value = torch.tensor(
        [
            [[[0, 1], [2, 3]]],
            [[[3, 2], [1, 0]]],
        ],
        dtype=torch.long,
    )
    model.transformer = MagicMock()
    model.transformer.forward.return_value = torch.randn(2, 4, 8, requires_grad=True)

    metrics = strategy.train_step(model, (torch.randn(2, 1, 2, 2, 2),), optimizer)

    assert 0.0 < metrics["mask_ratio"] <= 1.0


def test_maskgit_train_step_manual_path_falls_back_to_model_forward_when_no_transformer():
    strategy = MaskGITTrainingStrategy(mask_schedule_type="cosine")
    optimizer = MagicMock()

    model = MagicMock()
    model.compute_maskgit_loss = None
    model.encode_tokens.return_value = torch.tensor([[[[0, 1], [2, 3]]]], dtype=torch.long)
    model.transformer = None
    model.forward.return_value = torch.randn(1, 4, 8, requires_grad=True)

    _ = strategy.train_step(model, (torch.randn(1, 1, 2, 2, 2),), optimizer)

    model.forward.assert_called_once()


def test_maskgit_validate_step_flattens_tokens_when_dim_gt_2():
    strategy = MaskGITTrainingStrategy(mask_schedule_type="cosine")
    model = MagicMock()

    x = torch.randn(2, 1, 2, 2, 2)
    model.decode_tokens.return_value = x.clone()
    model.encode_tokens.side_effect = [
        torch.randint(0, 8, (2, 1, 2, 2), dtype=torch.long),
        torch.randint(0, 8, (2, 1, 2, 2), dtype=torch.long),
    ]
    model.transformer = MagicMock()
    model.transformer.encode.return_value = torch.randn(2, 4, 8)

    metrics = strategy.validate_step(model, (x,))

    model.transformer.encode.assert_called_once()
    assert "val_loss" in metrics
    assert "val_token_acc" in metrics


def test_maskgit_validate_step_reshapes_1d_tokens():
    strategy = MaskGITTrainingStrategy(mask_schedule_type="cosine")
    model = MagicMock()

    x = torch.randn(2, 1, 2, 2, 2)
    model.decode_tokens.return_value = x.clone()
    model.encode_tokens.side_effect = [
        torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
        torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
    ]
    model.transformer = MagicMock()
    model.transformer.encode.return_value = torch.randn(2, 3, 10)

    metrics = strategy.validate_step(model, (x,))

    assert "val_token_acc" in metrics


def test_maskgit_validate_step_uses_argmax_fallback_when_no_transformer():
    strategy = MaskGITTrainingStrategy(mask_schedule_type="cosine")
    model = MagicMock()

    x = torch.randn(2, 1, 2, 2, 2)
    two_d_tokens = torch.tensor([[1, 0], [1, 0]], dtype=torch.float32)
    model.decode_tokens.return_value = x.clone()
    model.encode_tokens.side_effect = [two_d_tokens, two_d_tokens]
    model.transformer = None

    metrics = strategy.validate_step(model, (x,))

    assert "val_token_acc" in metrics


def test_sliding_window_vqgan_inference_init_validates_roi_and_stores_params():
    with patch(
        "maskgit3d.infrastructure.data.padding.validate_roi_size",
        return_value=(32, 32, 32),
    ) as validate_mock:
        strategy = SlidingWindowVQGANInference(
            roi_size=(31, 31, 31),
            sw_batch_size=2,
            overlap=0.5,
            mode="constant",
            sigma_scale=0.2,
            progress=True,
            downsampling_factor=8,
            original_size=(30, 30, 30),
        )

    validate_mock.assert_called_once_with((31, 31, 31), 0.5, 8)
    assert strategy.roi_size == (32, 32, 32)
    assert strategy.sw_batch_size == 2
    assert strategy.original_size == (30, 30, 30)


def test_sliding_window_vqgan_inference_predict_without_original_size():
    strategy = SlidingWindowVQGANInference(roi_size=(16, 16, 16), original_size=None)
    model = MagicMock()
    model.device = torch.device("cpu")
    model.forward.return_value = torch.ones(1, 1, 16, 16, 16)

    batch = torch.randn(1, 1, 16, 16, 16)

    with patch("monai.inferers.utils.sliding_window_inference") as swi_mock:
        swi_mock.side_effect = lambda **kwargs: kwargs["predictor"](kwargs["inputs"])
        output = strategy.predict(model, batch)

    assert output.shape == (1, 1, 16, 16, 16)
    model.eval.assert_called_once()


def test_sliding_window_vqgan_inference_predict_with_original_size_crops_output():
    strategy = SlidingWindowVQGANInference(roi_size=(16, 16, 16), original_size=(8, 8, 8))
    model = MagicMock()
    model.device = torch.device("cpu")
    model.forward.return_value = torch.arange(4096, dtype=torch.float32).reshape(1, 1, 16, 16, 16)

    batch = torch.randn(1, 1, 16, 16, 16)

    with (
        patch("monai.inferers.utils.sliding_window_inference") as swi_mock,
        patch(
            "maskgit3d.infrastructure.data.padding.compute_output_crop",
            return_value=(slice(0, 8), slice(0, 8), slice(0, 8)),
        ) as crop_mock,
    ):
        swi_mock.side_effect = lambda **kwargs: kwargs["predictor"](kwargs["inputs"])
        output = strategy.predict(model, batch)

    crop_mock.assert_called_once_with((8, 8, 8), (16, 16, 16))
    assert output.shape == (1, 1, 8, 8, 8)


def test_sliding_window_vqgan_inference_post_process_normalizes_and_returns_numpy():
    strategy = SlidingWindowVQGANInference()
    predictions = torch.tensor([[[[[-1.0, 1.0]]]]])

    result = strategy.post_process(predictions)

    assert "images" in result
    assert result["images"].min() >= 0.0
    assert result["images"].max() <= 1.0


def test_sliding_window_latent_extractor_init_stores_params():
    strategy = SlidingWindowVQGANLatentExtractor(
        roi_size=(8, 8, 8),
        sw_batch_size=3,
        overlap=0.4,
        mode="constant",
        sigma_scale=0.3,
        progress=True,
    )

    assert strategy.roi_size == (8, 8, 8)
    assert strategy.sw_batch_size == 3
    assert strategy.overlap == 0.4
    assert strategy.mode == "constant"
    assert strategy.sigma_scale == 0.3
    assert strategy.progress is True


def test_sliding_window_latent_extractor_encode_patch_returns_quant_and_indices():
    strategy = SlidingWindowVQGANLatentExtractor()
    model = MagicMock()

    patch_tensor = torch.randn(1, 1, 2, 2, 2)
    encoded = torch.randn(1, 2, 2, 2, 2)
    quant = torch.randn(1, 2, 2, 2, 2)
    indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    model.encoder.return_value = encoded
    model.quant_conv.return_value = encoded
    model.quantize.return_value = (quant, None, (None, None, indices))

    out_quant, out_indices = strategy._encode_patch(model, patch_tensor)

    assert torch.equal(out_quant, quant)
    assert torch.equal(out_indices, indices)


def _build_latent_extractor_model_for_predict() -> MagicMock:
    model = MagicMock()
    model.device = torch.device("cpu")
    model.latent_shape = (2, 2, 2, 2)

    def encoder_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0], 2, 2, 2, 2)

    def quant_conv_fn(x: torch.Tensor) -> torch.Tensor:
        return x

    class QuantizeModule:
        def __init__(self):
            self.embedding = MagicMock()
            self.embedding.weight = torch.tensor(
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                dtype=torch.float32,
            )

        def __call__(self, h: torch.Tensor):
            b = h.shape[0]
            d, h_dim, w = 2, 2, 2
            indices = torch.zeros(b * d * h_dim * w, dtype=torch.long)
            quant = torch.zeros(b, 2, d, h_dim, w)
            return quant, None, (None, None, indices)

    model.encoder.side_effect = encoder_fn
    model.quant_conv.side_effect = quant_conv_fn
    model.quantize = QuantizeModule()
    return model


def _build_latent_extractor_model_for_extract() -> MagicMock:
    model = MagicMock()
    model.device = torch.device("cpu")

    def encoder_fn(x: torch.Tensor) -> torch.Tensor:
        return x.repeat(1, 2, 1, 1, 1)

    model.encoder.side_effect = encoder_fn
    model.quant_conv.side_effect = lambda x: x
    return model


def test_sliding_window_latent_extractor_predict_returns_spatial_indices():
    strategy = SlidingWindowVQGANLatentExtractor(roi_size=(2, 2, 2))
    model = _build_latent_extractor_model_for_predict()
    batch = torch.ones(1, 1, 2, 2, 2)

    with patch("monai.inferers.utils.sliding_window_inference") as swi_mock:
        swi_mock.side_effect = lambda **kwargs: kwargs["predictor"](kwargs["inputs"])
        indices = strategy.predict(model, batch)

    assert indices.shape == (1, 2, 2, 2)
    assert indices.dtype == torch.long


def test_sliding_window_latent_extractor_extract_latent_and_indices_returns_tuple():
    strategy = SlidingWindowVQGANLatentExtractor(roi_size=(2, 2, 2))
    model = _build_latent_extractor_model_for_extract()
    batch = torch.ones(1, 1, 2, 2, 2)

    quantized_latent = torch.full((1, 2, 2, 2, 2), 3.0)
    indices = torch.arange(8, dtype=torch.long)

    def quantize_call(x: torch.Tensor):
        return quantized_latent, None, (None, None, indices)

    model.quantize = MagicMock(side_effect=quantize_call)

    with patch("monai.inferers.utils.sliding_window_inference") as swi_mock:
        swi_mock.side_effect = lambda **kwargs: kwargs["predictor"](kwargs["inputs"])
        latent, out_indices = strategy.extract_latent_and_indices(model, batch)

    assert torch.equal(latent, quantized_latent)
    assert out_indices.shape == (1, 2, 2, 2)


def test_sliding_window_latent_extractor_post_process_returns_numpy_indices():
    strategy = SlidingWindowVQGANLatentExtractor()
    predictions = torch.tensor([[[[1, 2], [3, 4]]]])

    result = strategy.post_process(predictions)

    assert "indices" in result
    assert result["indices"].shape == (1, 1, 2, 2)
