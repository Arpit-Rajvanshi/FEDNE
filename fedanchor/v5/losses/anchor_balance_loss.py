import torch
import torch.nn as nn
from typing import Tuple

class AnchorBalanceLoss(nn.Module):
    """
    Explicit Anchor Utilization Balancing Penalty.
    
    Computes empirical anchor utilization u_j = (1/N) * sum_i q_ij,
    and penalizes the mean squared error from the uniform target (1/K).
    
    L_balance = sum_j (u_j - 1/K)^2
    """
    def __init__(self, eps: float = 1e-7) -> None:
        super(AnchorBalanceLoss, self).__init__()
        self.eps = float(eps)

    def forward(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: torch.Tensor of shape [N, K] containing soft assignments.
        Returns:
            loss_val: scalar torch.Tensor for balance loss.
            u: torch.Tensor of shape [K] containing utilizations.
        """
        N, K = q.shape
        if N == 0 or K == 0:
            zero_loss = torch.tensor(0.0, device=q.device, requires_grad=True)
            zero_u = torch.zeros((K,), device=q.device)
            return zero_loss, zero_u
            
        # Empirical utilization of each anchor
        u = torch.mean(q, dim=0)  # Shape [K]
        
        # Target uniform utilization
        target_u = 1.0 / K
        
        # Balance penalty
        loss_val = torch.sum((u - target_u) ** 2)
        
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            raise ValueError("AnchorBalanceLoss evaluated to NaN or Inf.")
            
        return loss_val, u
