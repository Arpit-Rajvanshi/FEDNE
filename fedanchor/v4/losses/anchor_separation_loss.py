import torch
import torch.nn as nn
from typing import Tuple

class AnchorSeparationLoss(nn.Module):
    """
    Explicit Geometric Anchor Separation Penalty.
    
    L_sep = (1 / K(K-1)) * sum_{i != j} max(0, m - ||a_i - a_j||)^2
    """
    def __init__(self, margin: float = 0.1, eps: float = 1e-7) -> None:
        super(AnchorSeparationLoss, self).__init__()
        self.margin = float(margin)
        self.eps = float(eps)

    def forward(self, anchors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchors: torch.Tensor of shape [K, 2]
        Returns:
            loss_val: scalar torch.Tensor for separation loss.
        """
        K = anchors.shape[0]
        if K <= 1:
            return torch.tensor(0.0, device=anchors.device, requires_grad=True)
            
        # Pairwise differences
        # diff[i, j] = a_i - a_j
        diff = anchors.unsqueeze(1) - anchors.unsqueeze(0)  # [K, K, 2]
        
        # Add epsilon for numerical stability of sqrt
        sq_dists = torch.sum(diff ** 2, dim=2) + self.eps  # [K, K]
        dists = torch.sqrt(sq_dists)  # [K, K]
        
        # max(0, m - d)^2
        violation = torch.clamp(self.margin - dists, min=0.0)
        violation_sq = violation ** 2
        
        # Zero out the diagonal (i == j)
        mask = torch.eye(K, device=anchors.device).bool()
        violation_sq = violation_sq.masked_fill(mask, 0.0)
        
        loss_val = torch.sum(violation_sq) / (K * (K - 1))
        
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            raise ValueError("AnchorSeparationLoss evaluated to NaN or Inf.")
            
        return loss_val
