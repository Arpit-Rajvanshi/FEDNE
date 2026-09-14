import torch
import torch.nn as nn

class AnchorCoverageLoss(nn.Module):
    """
    Local anchor coverage loss.
    Encourages learnable anchor parameters A_m to represent/cover the client's local embedding distribution.
    
    L_anchor = (1 / N) * sum_i min_j ||z_i - a_j||^2
    """
    def __init__(self, eps: float = 1e-7) -> None:
        super(AnchorCoverageLoss, self).__init__()
        self.eps = float(eps)

    def forward(self, embeddings: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: torch.Tensor of shape [N, 2]
            anchors: torch.Tensor of shape [K, 2]
        Returns:
            anchor_loss: scalar torch.Tensor
        """
        N = embeddings.shape[0]
        K = anchors.shape[0]
        if N == 0 or K == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Compute pairwise squared Euclidean distances [N, K]
        # ||z_i - a_j||^2 = ||z_i||^2 + ||a_j||^2 - 2 <z_i, a_j>
        diff = embeddings.unsqueeze(1) - anchors.unsqueeze(0)  # [N, K, 2]
        sq_dists = torch.sum(diff ** 2, dim=2)  # [N, K]
        
        # Minimum distance for each embedding point z_i to its nearest anchor
        min_sq_dists, _ = torch.min(sq_dists, dim=1)  # [N]
        
        loss_val = torch.mean(min_sq_dists)
        
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            raise ValueError("AnchorCoverageLoss evaluated to NaN or Inf.")
            
        return loss_val
