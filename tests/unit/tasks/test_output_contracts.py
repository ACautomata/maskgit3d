import importlib
import sys
from typing import get_type_hints

from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

output_contracts = importlib.import_module("maskgit3d.tasks.output_contracts")
VQVAETrainingStepOutput = output_contracts.VQVAETrainingStepOutput
VQVAEEvalStepOutput = output_contracts.VQVAEEvalStepOutput
MaskGITTrainingStepOutput = output_contracts.MaskGITTrainingStepOutput
MaskGITEvalStepOutput = output_contracts.MaskGITEvalStepOutput


def test_vqvae_training_step_output_contract_keys() -> None:
    hints = get_type_hints(VQVAETrainingStepOutput)

    assert set(hints) == {
        "loss",
        "loss_g",
        "loss_d",
        "nll_loss",
        "rec_loss",
        "p_loss",
        "g_loss",
        "vq_loss",
        "disc_loss",
    }
    assert hints["loss"] is torch.Tensor
    assert hints["loss_g"] is torch.Tensor
    assert hints["vq_loss"] is torch.Tensor


def test_vqvae_eval_step_output_contract_keys() -> None:
    hints = get_type_hints(VQVAEEvalStepOutput)

    assert set(hints) == {"x_real", "x_recon"}
    assert hints["x_real"] is torch.Tensor
    assert hints["x_recon"] is torch.Tensor


def test_maskgit_training_step_output_contract_keys() -> None:
    hints = get_type_hints(MaskGITTrainingStepOutput)

    assert set(hints) == {"loss"}
    assert hints["loss"] is torch.Tensor


def test_maskgit_eval_step_output_contract_keys() -> None:
    hints = get_type_hints(MaskGITEvalStepOutput)

    assert set(hints) == {
        "x_real",
        "generated_images",
        "masked_logits",
        "masked_targets",
    }
    assert hints["x_real"] is torch.Tensor
    assert hints["generated_images"] is torch.Tensor
    assert hints["masked_logits"] is torch.Tensor
    assert hints["masked_targets"] is torch.Tensor
