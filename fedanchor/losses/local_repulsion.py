import torch
import torch.nn as nn


class LocalNegativeRepulsionLoss(nn.Module):
    """
    Local negative-sampling repulsion (same form as FEDNE's local repulsion term).

    L_rep_local = E_i sum_{s=1}^{b} -log(1 - phi(z_i, z_neg_s)),  phi = 1 / (1 + ||z_i - z_neg||^2)

    Without this term the attraction loss has a trivial minimiser (all points collapse
    onto one location), which is what happened in the original FEDANCHOR runs
    (attraction ~1e-5, 1-2 anchors holding ~100% of the points, kNN acc ~20%).
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, z_i: torch.Tensor, z_neg: torch.Tensor) -> torch.Tensor:
        # z_i: [B, D], z_neg: [B, b, D]
        d2 = torch.sum((z_i.unsqueeze(1) - z_neg) ** 2, dim=-1)
        phi = torch.clamp(1.0 / (1.0 + d2), max=1.0 - self.eps)
        return torch.mean(-torch.sum(torch.log(1.0 - phi), dim=-1))
