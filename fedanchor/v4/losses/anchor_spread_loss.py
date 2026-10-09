import torch
import torch.nn as nn
from typing import Tuple

class AnchorSpreadLoss(nn.Module):
    """
    Explicit anchor diversity / spread loss.
    Penalizes anchors becoming too close to each other.
    
    L_spread = 1/[K(K-1)] * sum_{i != j} 1 / (||a_i - a_j||^2 + epsilon)
    """
    def __init__(self, eps: float = 1e-7) -> None:
        super(AnchorSpreadLoss, self).__init__()
        self.eps = float(eps)

    def forward(self, anchors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchors: torch.Tensor of shape [K, 2]
        Returns:
            loss_val: scalar torch.Tensor
        """
        K = anchors.shape[0]
        if K <= 1:
            return torch.tensor(0.0, device=anchors.device, requires_grad=True)

        # Pairwise squared Euclidean distances [K, K]
        diff = anchors.unsqueeze(1) - anchors.unsqueeze(0)  # [K, K, 2]
        sq_dists = torch.sum(diff ** 2, dim=2)  # [K, K]
        
        # Calculate 1 / (||a_i - a_j||^2 + eps)
        inv_sq_dists = 1.0 / (sq_dists + self.eps)
        
        # Zero out the diagonal (i == j)
        mask = 1.0 - torch.eye(K, device=anchors.device)
        off_diag_inv = inv_sq_dists * mask
        
        # Average over K(K-1) off-diagonal elements
        loss_val = torch.sum(off_diag_inv) / (K * (K - 1))
        
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            raise ValueError("AnchorSpreadLoss evaluated to NaN or Inf.")
            
        return loss_val
