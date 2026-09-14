import torch
import torch.nn as nn

class AttractionLoss(nn.Module):
    """
    Local attraction loss for neighbor embedding.
    Encourages local kNN graph neighbors to remain close in 2D embedding space.
    
    L_att = - (1 / |E|) * sum_{(i,j) in E} log(phi(z_i, z_j))
    where phi(z_i, z_j) = 1 / (1 + ||z_i - z_j||^2).
    Equivalently:
    L_att = (1 / |E|) * sum_{(i,j) in E} log(1 + ||z_i - z_j||^2 + eps).
    """
    def __init__(self, eps: float = 1e-7) -> None:
        super(AttractionLoss, self).__init__()
        self.eps = float(eps)

    def forward(self, embeddings: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: torch.Tensor of shape [N, 2]
            edges: torch.Tensor of shape [E, 2] containing indices (src, dst)
        Returns:
            attraction_loss: scalar torch.Tensor
        """
        if edges.shape[0] == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        src_indices = edges[:, 0]
        dst_indices = edges[:, 1]
        
        z_src = embeddings[src_indices]
        z_dst = embeddings[dst_indices]
        
        # Squared Euclidean distance
        sq_dist = torch.sum((z_src - z_dst) ** 2, dim=1)
        
        # log(1 + sq_dist + eps)
        loss = torch.log(1.0 + sq_dist + self.eps)
        
        loss_val = torch.mean(loss)
        
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            raise ValueError("AttractionLoss evaluated to NaN or Inf.")
            
        return loss_val
