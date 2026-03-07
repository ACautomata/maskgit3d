from __future__ import annotations

import torch
from torch import nn

from monai.losses import PerceptualLoss as MONAIPerceptualLoss


class PerceptualLoss(nn.Module):
    """Perceptual loss using features from pretrained deep neural networks.

    Wraps MONAI's PerceptualLoss for use in VQVAE training.

    Args:
        network: Network backbone for feature extraction.
            Options: "alex", "vgg", "squeeze", "radimagenet_resnet50",
            "medicalnet_resnet10_23datasets", "medicalnet_resnet50_23datasets",
            "resnet50". Default: "alex".
        pretrained: Whether to use pretrained weights. Default: True.
        spatial_dims: Number of spatial dimensions (2 or 3). Default: 3.
        is_fake_3d: Use 2.5D approach for 3D inputs. Default: True.
        fake_3d_ratio: Ratio of slices for fake 3D. Default: 0.5.
        enabled: If False, forward returns 0.0. Default: True.
    """

    def __init__(
        self,
        network: str = "alex",
        pretrained: bool = True,
        spatial_dims: int = 3,
        is_fake_3d: bool = True,
        fake_3d_ratio: float = 0.5,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = enabled

        if enabled:
            self.loss: MONAIPerceptualLoss | None = MONAIPerceptualLoss(
                spatial_dims=spatial_dims,
                network_type=network,
                is_fake_3d=is_fake_3d,
                fake_3d_ratio=fake_3d_ratio,
                pretrained=pretrained,
            )
            # Freeze LPIPS weights - they should not be trained
            for param in self.loss.parameters():
                param.requires_grad = False
        else:
            self.loss = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between prediction and target.

        Args:
            pred: Predicted tensor (B, C, [D,] H, W).
            target: Target tensor with same shape as pred.

        Returns:
            Scalar loss value.
        """
        if not self.enabled or self.loss is None:
            return torch.tensor(0.0, device=pred.device, requires_grad=False)

        return self.loss(pred, target)
