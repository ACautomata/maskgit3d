from __future__ import annotations

import torch
from torch import nn


class GANLoss(nn.Module):
    def __init__(
        self,
        gan_mode: str = "lsgan",
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ) -> None:
        super().__init__()

        valid_modes = {"lsgan", "vanilla", "hinge"}
        if gan_mode not in valid_modes:
            raise ValueError(
                f"Unsupported gan_mode '{gan_mode}'. Expected one of {sorted(valid_modes)}"
            )

        self.gan_mode = gan_mode
        self.real_label = target_real_label
        self.fake_label = target_fake_label

        if gan_mode == "lsgan":
            self.loss: nn.Module | None = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = None

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_value = self.real_label if target_is_real else self.fake_label
        target_tensor = torch.full_like(prediction, fill_value=target_value)
        return target_tensor.expand(prediction.shape)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if self.gan_mode == "hinge":
            if target_is_real:
                return torch.relu(1.0 - prediction).mean()
            return torch.relu(1.0 + prediction).mean()

        target_tensor = self.get_target_tensor(prediction, target_is_real)
        if self.loss is None:
            raise RuntimeError("Loss module is not initialized")
        return self.loss(prediction, target_tensor)

    def discriminator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        loss_real = self(real_pred, target_is_real=True)
        loss_fake = self(fake_pred, target_is_real=False)
        return 0.5 * (loss_real + loss_fake)

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        if self.gan_mode == "hinge":
            return -fake_pred.mean()
        return self(fake_pred, target_is_real=True)
