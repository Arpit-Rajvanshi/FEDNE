import torch
import torch.nn as nn

class RepulsionLoss(nn.Module):
    """
    Computes local repulsion loss using negative samples.
    L_rep = - E_i sum_{s=1}^b log(1 - phi(z_i, z_{neg, s}))
    where phi(z_i, z_j) = 1 / (1 + ||z_i - z_j||^2).
    """
    def __init__(self, eps: float = 1e-7) -> None:
        super(RepulsionLoss, self).__init__()
        self.eps = eps
        
    def forward(self, z_i: torch.Tensor, z_neg: torch.Tensor) -> torch.Tensor:
        """
        z_i: Source embeddings [B, D]
        z_neg: Negative samples embeddings [B, b, D]
        """
        # Expand z_i to [B, 1, D] to compute distance to all negative samples
        z_i_expanded = z_i.unsqueeze(1)
        
        # Compute squared Euclidean distances: [B, b]
        d2 = torch.sum((z_i_expanded - z_neg) ** 2, dim=-1)
        
        # phi(z_i, z_neg)
        phi = 1.0 / (1.0 + d2)
        
        # Clamp to avoid log(0) when computing log(1 - phi)
        phi = torch.clamp(phi, min=0.0, max=1.0 - self.eps)
        
        # Sum over the b negative samples for each source point
        rep_sum = -torch.sum(torch.log(1.0 - phi), dim=-1)  # [B]
        
        # Return average over batch
        return torch.mean(rep_sum)
