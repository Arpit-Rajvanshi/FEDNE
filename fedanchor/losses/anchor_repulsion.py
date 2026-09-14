import torch
import torch.nn as nn
from typing import Dict, Tuple

class AnchorRepulsionLoss(nn.Module):
    """
    Anchor-based cross-client repulsion loss with configurable scaling & normalization.
    
    Raw Repulsion:
    L_rep_raw = (1 / (N * |A_other|)) * sum_i sum_{a in A_other} log(1 + 1 / (||z_i - a||^2 + eps))
    
    If normalization == "scale":
    L_rep_scaled = scale * L_rep_raw
    """
    def __init__(
        self,
        normalization: str = "none",
        scale: float = 1.0,
        eps: float = 1e-7
    ) -> None:
        super(AnchorRepulsionLoss, self).__init__()
        self.normalization = normalization.lower()
        self.scale = float(scale)
        self.eps = float(eps)

    def forward(self, embeddings: torch.Tensor, other_anchors: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings: torch.Tensor of shape [N, 2]
            other_anchors: torch.Tensor of shape [K_other, 2]
        Returns:
            scaled_loss: scalar torch.Tensor for autograd
            breakdown: Dict containing 'repulsion_raw' and 'repulsion_scaled'
        """
        N = embeddings.shape[0]
        K_other = other_anchors.shape[0]
        
        if N == 0 or K_other == 0:
            zero_tensor = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            return zero_tensor, {"repulsion_raw": 0.0, "repulsion_scaled": 0.0}

        diff = embeddings.unsqueeze(1) - other_anchors.unsqueeze(0)  # [N, K_other, 2]
        sq_dists = torch.sum(diff ** 2, dim=2)  # [N, K_other]
        
        inv_sq_dist = 1.0 / (sq_dists + self.eps)
        loss_matrix = torch.log(1.0 + inv_sq_dist)
        
        raw_loss = torch.mean(loss_matrix)
        
        if self.normalization == "scale":
            scaled_loss = self.scale * raw_loss
        else:
            scaled_loss = raw_loss
            
        if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
            raise ValueError("AnchorRepulsionLoss evaluated to NaN or Inf.")
            
        breakdown = {
            "repulsion_raw": raw_loss.item(),
            "repulsion_scaled": scaled_loss.item()
        }
        
        return scaled_loss, breakdown
