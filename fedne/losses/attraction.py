import torch
import torch.nn as nn

class AttractionLoss(nn.Module):
    """
    Computes the attraction loss for pairs of neighboring data points.
    L_att = - E_ij log(phi(z_i, z_j))
    where phi(z_i, z_j) = 1 / (1 + ||z_i - z_j||^2).
    """
    def __init__(self, eps: float = 1e-7) -> None:
        super(AttractionLoss, self).__init__()
        self.eps = eps
        
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        z_i: Source embeddings [B, D]
        z_j: Target (neighbor) embeddings [B, D]
        """
        d2 = torch.sum((z_i - z_j) ** 2, dim=-1)
        phi = 1.0 / (1.0 + d2)
        # Clamp to avoid log(0)
        phi = torch.clamp(phi, min=self.eps, max=1.0)
        loss = -torch.log(phi)
        return torch.mean(loss)
