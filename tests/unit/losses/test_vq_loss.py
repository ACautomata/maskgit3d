import torch

from src.maskgit3d.losses.vq_loss import VQLoss


def test_vq_loss_returns_scalar_non_negative() -> None:
    loss_fn = VQLoss(commitment_cost=0.25)

    z = torch.randn(2, 64, 4, 4, 4, requires_grad=True)
    z_q = z + 0.1 * torch.randn_like(z)

    loss = loss_fn(z, z_q)

    assert loss.dim() == 0
    assert loss.item() >= 0


def test_vq_loss_matches_codebook_plus_weighted_commitment() -> None:
    commitment_cost = 0.5
    loss_fn = VQLoss(commitment_cost=commitment_cost)

    z = torch.randn(2, 8, 2, 2, 2)
    z_q = z + 0.2 * torch.randn_like(z)

    loss = loss_fn(z, z_q)
    codebook_loss = torch.nn.functional.mse_loss(z_q, z.detach())
    commitment_loss = torch.nn.functional.mse_loss(z, z_q.detach())
    expected = codebook_loss + commitment_cost * commitment_loss

    assert torch.isclose(loss, expected)
