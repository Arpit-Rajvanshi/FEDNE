import torch
import torch.nn as nn
from typing import List, Dict
from fedne.losses.attraction import AttractionLoss
from fedne.losses.repulsion import RepulsionLoss

class ContrastiveNeighborEmbeddingLoss(nn.Module):
    """
    Combined contrastive loss for FEDNE (Equation 10).
    L_m = L_att + (|D_m| / |D|) * L_rep_local + sum_{m' != m} (|D_m'| / |D|) * L_rep_surrogate_m'
    """
    def __init__(
        self,
        client_id: int,
        client_sizes: List[int],
        attraction_coeff: float = 1.0,
        local_rep_coeff: float = 1.0,
        surr_rep_coeff: float = 1.0,
        eps: float = 1e-7
    ) -> None:
        super(ContrastiveNeighborEmbeddingLoss, self).__init__()
        self.client_id = client_id
        self.client_sizes = client_sizes
        self.total_size = sum(client_sizes)
        self.attraction_coeff = attraction_coeff
        self.local_rep_coeff = local_rep_coeff
        self.surr_rep_coeff = surr_rep_coeff
        
        self.attraction_loss = AttractionLoss(eps=eps)
        self.repulsion_loss = RepulsionLoss(eps=eps)
        
        # Scaling factors: |D_m| / |D|
        self.local_scale = self.client_sizes[client_id] / self.total_size
        self.surr_scales = [size / self.total_size for size in client_sizes]
        
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        z_neg: torch.Tensor,
        surrogates: List[nn.Module],
        enable_surrogate: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        z_i: Source embeddings [B, D]
        z_j: Neighbor embeddings [B, D]
        z_neg: Negative sample embeddings [B, b, D]
        surrogates: List of surrogate models for all clients (length M)
        enable_surrogate: If False, surrogate repulsion is omitted.
        """
        # 1. Attraction Loss (no scale factor from client sizes)
        loss_att = self.attraction_loss(z_i, z_j)
        
        # 2. Local Repulsion Loss (scaled by |D_m| / |D|)
        loss_rep_local = self.repulsion_loss(z_i, z_neg)
        loss_rep_local_scaled = self.local_scale * loss_rep_local
        
        # 3. Surrogate Repulsion Loss (scaled by |D_m'| / |D|)
        loss_rep_surr = torch.tensor(0.0, device=z_i.device)
        if enable_surrogate and len(surrogates) > 1:
            # We loop over all other clients
            for m_prime, surrogate in enumerate(surrogates):
                if m_prime == self.client_id:
                    continue
                # Evaluate surrogate on source embeddings: [B, 2] -> [B, 1]
                # Clamp prediction to be non-negative to prevent negative loss explosion
                surr_pred = torch.clamp(surrogate(z_i), min=0.0)  # [B, 1]
                scale = self.surr_scales[m_prime]
                loss_rep_surr = loss_rep_surr + scale * torch.mean(surr_pred)
                
        # Combine losses using coefficients
        total_loss = (
            self.attraction_coeff * loss_att
            + self.local_rep_coeff * loss_rep_local_scaled
            + self.surr_rep_coeff * loss_rep_surr
        )
        
        return {
            "loss": total_loss,
            "attraction": loss_att,
            "repulsion_local": loss_rep_local,
            "repulsion_local_scaled": loss_rep_local_scaled,
            "repulsion_surrogate": loss_rep_surr
        }
