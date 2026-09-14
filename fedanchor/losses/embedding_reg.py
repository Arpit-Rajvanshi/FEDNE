import torch
import torch.nn as nn

class EmbeddingRegLoss(nn.Module):
    """
    Optional Embedding Regularization Objective.
    L_embedding_reg = (1 / N) * sum_i ||z_i||^2
    Prevents embeddings from expanding to infinitely large coordinate norms.
    """
    def __init__(self) -> None:
        super(EmbeddingRegLoss, self).__init__()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: torch.Tensor of shape [N, 2]
        Returns:
            reg_loss: scalar torch.Tensor
        """
        if embeddings.shape[0] == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        sq_norms = torch.sum(embeddings ** 2, dim=1)
        loss_val = torch.mean(sq_norms)
        
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            raise ValueError("EmbeddingRegLoss evaluated to NaN or Inf.")
            
        return loss_val
