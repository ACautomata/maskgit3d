import torch
import torch.nn as nn
import torch.nn.functional as F


class VQLoss(nn.Module):
    def __init__(self, commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.commitment_cost = commitment_cost

    def forward(self, z: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z, z_q.detach())
        return codebook_loss + self.commitment_cost * commitment_loss
