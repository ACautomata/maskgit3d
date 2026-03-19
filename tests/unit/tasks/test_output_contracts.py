import importlib
import sys
from typing import get_type_hints

from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

output_contracts = importlib.import_module("maskgit3d.tasks.output_contracts")
MaskGITStepOutput = output_contracts.MaskGITStepOutput
VQVAEStepOutput = output_contracts.VQVAEStepOutput
VQVAETrainingOutput = output_contracts.VQVAETrainingOutput
VQVAEValidationOutput = output_contracts.VQVAEValidationOutput
MaskGITTrainingOutput = output_contracts.MaskGITTrainingOutput
MaskGITValidationOutput = output_contracts.MaskGITValidationOutput


def test_vqvae_step_output_contract_keys() -> None:
    hints = get_type_hints(VQVAEStepOutput)

    assert set(hints) == {
        "x_real",
        "x_recon",
        "vq_loss",
        "last_layer",
        "inference_time",
        "use_sliding_window",
    }
    assert VQVAEStepOutput.__required_keys__ == {"x_real", "x_recon", "vq_loss"}
    assert VQVAEStepOutput.__optional_keys__ == {
        "last_layer",
        "inference_time",
        "use_sliding_window",
    }
    assert hints["x_real"] is torch.Tensor
    assert hints["x_recon"] is torch.Tensor
    assert hints["vq_loss"] is torch.Tensor


def test_maskgit_step_output_contract_keys() -> None:
    hints = get_type_hints(MaskGITStepOutput)

    assert set(hints) == {
        "x_real",
        "generated_images",
        "masked_logits",
        "masked_targets",
        "mask_ratio",
        "input_shape",
        "token_shape",
    }
    assert MaskGITStepOutput.__required_keys__ == {"x_real", "generated_images"}
    assert MaskGITStepOutput.__optional_keys__ == {
        "masked_logits",
        "masked_targets",
        "mask_ratio",
        "input_shape",
        "token_shape",
    }
    assert hints["x_real"] is torch.Tensor
    assert hints["generated_images"] is torch.Tensor


def test_vqvae_training_output_fields() -> None:
    x_real = torch.randn(1, 1, 32, 32, 32)
    x_recon = torch.randn(1, 1, 32, 32, 32)
    vq_loss = torch.tensor(0.1)
    last_layer = torch.nn.Parameter(torch.randn(1, 1))

    output = VQVAETrainingOutput(x_real=x_real, x_recon=x_recon, vq_loss=vq_loss)
    assert output.x_real is x_real
    assert output.x_recon is x_recon
    assert output.vq_loss is vq_loss
    assert output.last_layer is None

    output_with_last_layer = VQVAETrainingOutput(
        x_real=x_real, x_recon=x_recon, vq_loss=vq_loss, last_layer=last_layer
    )
    assert output_with_last_layer.last_layer is last_layer


def test_vqvae_validation_output_fields() -> None:
    x_real = torch.randn(1, 1, 32, 32, 32)
    x_recon = torch.randn(1, 1, 32, 32, 32)
    vq_loss = torch.tensor(0.1)

    output = VQVAEValidationOutput(x_real=x_real, x_recon=x_recon, vq_loss=vq_loss)
    assert output.x_real is x_real
    assert output.x_recon is x_recon
    assert output.vq_loss is vq_loss
    assert output.inference_time is None
    assert output.use_sliding_window is None

    output_with_extras = VQVAEValidationOutput(
        x_real=x_real,
        x_recon=x_recon,
        vq_loss=vq_loss,
        inference_time=0.5,
        use_sliding_window=True,
    )
    assert output_with_extras.inference_time == 0.5
    assert output_with_extras.use_sliding_window is True


def test_maskgit_training_output_fields() -> None:
    tokens = torch.randint(0, 8192, (1, 2, 2, 2))
    masked_logits = torch.randn(4, 8192)
    masked_targets = torch.randint(0, 8192, (4,))
    mask_ratio = 0.5

    output = MaskGITTrainingOutput(
        tokens=tokens,
        masked_logits=masked_logits,
        masked_targets=masked_targets,
        mask_ratio=mask_ratio,
    )
    assert output.tokens is tokens
    assert output.masked_logits is masked_logits
    assert output.masked_targets is masked_targets
    assert output.mask_ratio == mask_ratio


def test_maskgit_validation_output_fields() -> None:
    x_real = torch.randn(1, 1, 32, 32, 32)
    generated_images = torch.randn(1, 1, 32, 32, 32)
    masked_logits = torch.randn(4, 8192)
    masked_targets = torch.randint(0, 8192, (4,))
    mask_ratio = 0.5
    token_shape = torch.Size([1, 2, 2, 2])

    output = MaskGITValidationOutput(
        x_real=x_real,
        generated_images=generated_images,
        masked_logits=masked_logits,
        masked_targets=masked_targets,
        mask_ratio=mask_ratio,
    )
    assert output.x_real is x_real
    assert output.generated_images is generated_images
    assert output.masked_logits is masked_logits
    assert output.masked_targets is masked_targets
    assert output.mask_ratio == mask_ratio
    assert output.token_shape is None

    output_with_token_shape = MaskGITValidationOutput(
        x_real=x_real,
        generated_images=generated_images,
        masked_logits=masked_logits,
        masked_targets=masked_targets,
        mask_ratio=mask_ratio,
        token_shape=token_shape,
    )
    assert output_with_token_shape.token_shape == token_shape
