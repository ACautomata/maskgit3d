import importlib
import sys
from typing import get_type_hints

from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

output_contracts = importlib.import_module("maskgit3d.tasks.output_contracts")
MaskGITStepOutput = output_contracts.MaskGITStepOutput
VQVAEStepOutput = output_contracts.VQVAEStepOutput


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
