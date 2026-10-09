"""
Embedding Repulsion Loss
========================

Introduced in V7 to prevent embedding collapse.

L_embed_rep = (1 / |P|) Σ_(i,j)∈P exp(-||z_i-z_j||² / τ)

where P is a sampled set of non-neighbor pairs.
"""

import torch
import torch.nn as nn

class EmbeddingRepulsionLoss(nn.Module):
    def __init__(self, tau: float = 1.0, max_pairs: int = 5000, eps: float = 1e-7):
        super(EmbeddingRepulsionLoss, self).__init__()
        self.tau = float(tau)
        self.max_pairs = int(max_pairs)
        self.eps = float(eps)

    def forward(self, embeddings: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        Compute repulsion for non-neighbor pairs.
        
        Args:
            embeddings: [N, 2] tensor
            edges: [E, 2] tensor of kNN edges (indices into embeddings)
            
        Returns:
            scalar loss
        """
        N = embeddings.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Determine number of candidate pairs to sample
        num_candidates = min(self.max_pairs * 2, (N * (N - 1)))
        
        # Generate random candidates
        idx1 = torch.randint(0, N, (num_candidates,), device=embeddings.device)
        idx2 = torch.randint(0, N, (num_candidates,), device=embeddings.device)
        
        # Filter i == j
        valid_mask = idx1 != idx2
        idx1 = idx1[valid_mask]
        idx2 = idx2[valid_mask]
        
        # Build edge set for fast lookup
        # Convert edge pairs to 1D keys: i * N + j
        edge_keys_1 = edges[:, 0] * N + edges[:, 1]
        edge_keys_2 = edges[:, 1] * N + edges[:, 0]
        edge_keys = torch.cat([edge_keys_1, edge_keys_2])
        
        cand_keys = idx1 * N + idx2
        
        # Filter candidates not in edges
        is_edge = torch.isin(cand_keys, edge_keys)
        idx1 = idx1[~is_edge]
        idx2 = idx2[~is_edge]
        
        if len(idx1) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        # Truncate to max_pairs
        if len(idx1) > self.max_pairs:
            idx1 = idx1[:self.max_pairs]
            idx2 = idx2[:self.max_pairs]
            
        # Compute exp(-||z_i - z_j||^2 / tau)
        z_i = embeddings[idx1]
        z_j = embeddings[idx2]
        
        sq_dist = torch.sum((z_i - z_j)**2, dim=1)
        repulsion = torch.exp(-sq_dist / self.tau)
        
        loss_val = torch.mean(repulsion)
        return loss_val
