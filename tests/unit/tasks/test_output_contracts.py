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
    assert VQVAEStepOutput.__required_keys__ == {
        "x_real",
        "x_recon",
        "vq_loss",
        "last_layer",
    }
    assert VQVAEStepOutput.__optional_keys__ == {"inference_time", "use_sliding_window"}
    assert hints["x_real"] is torch.Tensor
    assert hints["x_recon"] is torch.Tensor
    assert hints["vq_loss"] is torch.Tensor


def test_maskgit_step_output_contract_keys() -> None:
    hints = get_type_hints(MaskGITStepOutput)
    log_data_type = hints["log_data"]
    log_data_hints = get_type_hints(log_data_type)

    assert set(hints) == {"loss", "log_data"}
    assert MaskGITStepOutput.__required_keys__ == {"loss"}
    assert MaskGITStepOutput.__optional_keys__ == {"log_data"}
    assert hints["loss"] is torch.Tensor
    assert log_data_type.__required_keys__ == set()
    assert log_data_type.__optional_keys__ == {"correct", "total", "mask_ratio", "mask_acc"}
    assert log_data_hints == {
        "correct": int,
        "total": int,
        "mask_ratio": float,
        "mask_acc": float,
    }
