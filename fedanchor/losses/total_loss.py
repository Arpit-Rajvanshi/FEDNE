import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .attraction import AttractionLoss
from .anchor_loss import AnchorCoverageLoss
from .anchor_repulsion import AnchorRepulsionLoss
from .embedding_reg import EmbeddingRegLoss

class TotalLoss(nn.Module):
    """
    Combined objective function for Federated Anchor training with diagnostics, scaling, and optional initial balancing.
    L_total = lambda_att * L_att + lambda_anchor * L_anchor + lambda_rep_anchor * L_rep_scaled + lambda_emb_reg * L_emb_reg
    """
    def __init__(
        self,
        lambda_attraction: float = 1.0,
        lambda_anchor: float = 1.0,
        lambda_anchor_repulsion: float = 1.0,
        lambda_embedding_reg: float = 0.0,
        normalization: str = "none",
        scale: float = 1.0,
        eps: float = 1e-7,
        balancing_enabled: bool = False,
        use_attraction: bool = True,
        use_anchor: bool = True,
        use_anchor_repulsion: bool = True
    ) -> None:
        super(TotalLoss, self).__init__()
        self.lambda_attraction = float(lambda_attraction)
        self.lambda_anchor = float(lambda_anchor)
        self.lambda_anchor_repulsion = float(lambda_anchor_repulsion)
        self.lambda_embedding_reg = float(lambda_embedding_reg)
        self.eps = float(eps)
        self.balancing_enabled = balancing_enabled
        
        self.use_attraction = use_attraction
        self.use_anchor = use_anchor
        self.use_anchor_repulsion = use_anchor_repulsion
        
        self.attraction_fn = AttractionLoss(eps=self.eps)
        self.anchor_fn = AnchorCoverageLoss(eps=self.eps)
        self.anchor_repulsion_fn = AnchorRepulsionLoss(normalization=normalization, scale=scale, eps=self.eps)
        self.embedding_reg_fn = EmbeddingRegLoss()

        self.balance_factors: Dict[str, float] = {"attraction": 1.0, "anchor": 1.0, "repulsion": 1.0}
        self.balanced_initialized = False

    def forward(
        self,
        embeddings: torch.Tensor,
        edges: torch.Tensor,
        local_anchors: torch.Tensor,
        other_anchors: torch.Tensor,
        client_id: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes total loss and component breakdown with diagnostics ratios.
        """
        l_att = self.attraction_fn(embeddings, edges) if self.use_attraction else torch.tensor(0.0, device=embeddings.device)
        l_anc = self.anchor_fn(embeddings, local_anchors) if self.use_anchor else torch.tensor(0.0, device=embeddings.device)
        
        if self.use_anchor_repulsion:
            l_rep_scaled, rep_dict = self.anchor_repulsion_fn(embeddings, other_anchors)
            l_rep_raw_val = rep_dict["repulsion_raw"]
            l_rep_scaled_val = rep_dict["repulsion_scaled"]
        else:
            l_rep_scaled = torch.tensor(0.0, device=embeddings.device)
            l_rep_raw_val = 0.0
            l_rep_scaled_val = 0.0
            
        l_reg = self.embedding_reg_fn(embeddings) if self.lambda_embedding_reg > 0 else torch.tensor(0.0, device=embeddings.device)

        # Automatic initial loss balancing if enabled
        if self.balancing_enabled and not self.balanced_initialized:
            with torch.no_grad():
                att_mag = max(l_att.item(), 1e-5)
                anc_mag = max(l_anc.item(), 1e-5)
                rep_mag = max(l_rep_scaled.item(), 1e-5)
                
                self.balance_factors["attraction"] = 1.0
                self.balance_factors["anchor"] = att_mag / anc_mag
                self.balance_factors["repulsion"] = att_mag / rep_mag
                self.balanced_initialized = True

        b_att = self.balance_factors["attraction"]
        b_anc = self.balance_factors["anchor"]
        b_rep = self.balance_factors["repulsion"]

        total = (
            self.lambda_attraction * b_att * l_att +
            self.lambda_anchor * b_anc * l_anc +
            self.lambda_anchor_repulsion * b_rep * l_rep_scaled +
            self.lambda_embedding_reg * l_reg
        )
        
        if torch.isnan(total) or torch.isinf(total):
            msg = f"TotalLoss evaluated to NaN or Inf on Client {client_id} in Round {round_num}."
            raise ValueError(msg)

        # Diagnostic ratios
        rep_to_att = l_rep_scaled_val / (l_att.item() + self.eps)
        rep_to_anc = l_rep_scaled_val / (l_anc.item() + self.eps)

        breakdown = {
            "total_loss": total.item(),
            "attraction_loss": l_att.item(),
            "anchor_loss": l_anc.item(),
            "anchor_repulsion_raw": l_rep_raw_val,
            "anchor_repulsion_scaled": l_rep_scaled_val,
            "embedding_reg_loss": l_reg.item(),
            "repulsion_to_attraction_ratio": rep_to_att,
            "repulsion_to_anchor_ratio": rep_to_anc,
        }
        
        return total, breakdown
