import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class AnchorSoftLoss(nn.Module):
    """
    Local soft anchor assignment loss.
    Uses a Student-t-style similarity to construct differentiable soft assignments
    between embeddings and anchors, mitigating the hard winner-take-all min() collapse.
    
    q_ij = (1 + ||z_i - a_j||^2 / alpha)^(-(alpha+1)/2) / Z_i
    L_soft_anchor = (1 / N) * sum_i sum_j q_ij ||z_i - a_j||^2
    """
    def __init__(self, alpha: float = 1.0, eps: float = 1e-7) -> None:
        super(AnchorSoftLoss, self).__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, embeddings: torch.Tensor, anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: torch.Tensor of shape [N, 2]
            anchors: torch.Tensor of shape [K, 2]
        Returns:
            loss_val: scalar torch.Tensor
            q: torch.Tensor of shape [N, K] containing the soft assignment probabilities
        """
        N = embeddings.shape[0]
        K = anchors.shape[0]
        if N == 0 or K == 0:
            zero_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            zero_q = torch.zeros((N, K), device=embeddings.device)
            return zero_loss, zero_q

        # Pairwise squared Euclidean distances [N, K]
        diff = embeddings.unsqueeze(1) - anchors.unsqueeze(0)  # [N, K, 2]
        sq_dists = torch.sum(diff ** 2, dim=2)  # [N, K]
        
        # Numerically stable computation of q_ij via log-softmax
        # log(numerator) = -((alpha+1)/2) * log(1 + sq_dist / alpha)
        log_num = -0.5 * (self.alpha + 1.0) * torch.log(1.0 + sq_dists / self.alpha)
        
        # log_softmax computes: log(num_ij) - log(sum_k exp(log_num_ik))
        log_q = F.log_softmax(log_num, dim=1)
        q = torch.exp(log_q)
        
        # Stop gradient for q_ij?
        # The user's formula implies L = (1/N) \sum \sum q_ij ||z_i - a_j||^2
        # Typically, in t-SNE or DEC, we don't treat q_ij as a constant, but whether we backpropagate through q_ij depends on the formulation.
        # Wait, the user asked to replace min_j ||z_i-a_j||^2 with \sum_j q_ij ||z_i-a_j||^2.
        # If we just do `loss = torch.sum(q * sq_dists) / N`, PyTorch will backpropagate through both `q` and `sq_dists`.
        # This is correct.
        
        loss_val = torch.mean(torch.sum(q * sq_dists, dim=1))
        
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            raise ValueError("AnchorSoftLoss evaluated to NaN or Inf.")
            
        return loss_val, q
