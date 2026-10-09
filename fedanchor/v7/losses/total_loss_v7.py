"""
v7 Total Loss Function - TotalLossv7
=====================================

Preserves the full V5/V4 loss structure:

L_total = λ_att  * L_att
        + λ_anchor * L_soft
        + λ_rep  * L_rep
        + λ_spread * L_spread
        + λ_balance * L_balance
        + λ_sep  * L_sep
        + λ_reg  * L_reg
        + λ_embed_rep * L_embed_rep

The ONLY structural change from V6 is the addition of L_embed_rep.
to the CLIENT'S LOCAL COPY OF THE SHARED GLOBAL ANCHORS rather than
independently learned client-specific anchors.

The `other_anchors` argument is REMOVED from v7 because there are no
other-client-specific anchors — there is only one global set.
Instead, `lambda_anchor_repulsion` is set to 0.0 by default in v7.

The separation loss (L_sep) now encourages spread of the SHARED anchors,
which benefits all clients simultaneously since they share the same geometry.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

# Reuse V5/V4 loss components (not modified)
from fedanchor.losses.attraction import AttractionLoss
from fedanchor.v3.losses.anchor_soft_loss import AnchorSoftLoss
from fedanchor.v3.losses.anchor_spread_loss import AnchorSpreadLoss
from fedanchor.v4.losses.anchor_balance_loss import AnchorBalanceLoss
from fedanchor.v4.losses.anchor_separation_loss import AnchorSeparationLoss
from fedanchor.losses.embedding_reg import EmbeddingRegLoss
from fedanchor.v7.losses.embedding_repulsion_loss import EmbeddingRepulsionLoss


class TotalLossv7(nn.Module):
    """
    Combined objective for FEDANCHOR v7 (Shared Global Anchors).

    Interface is intentionally close to TotalLossV4 to allow
    side-by-side ablation, except:
    - `other_anchors` argument removed from forward() (no cross-client repulsion)
    - cross-client anchor repulsion weight default = 0.0
    """

    def __init__(
        self,
        lambda_attraction: float = 1.0,
        lambda_anchor: float = 1.0,
        lambda_spread: float = 1.0,
        lambda_balance: float = 1.0,
        lambda_sep: float = 0.1,
        lambda_embedding_reg: float = 0.0,
        lambda_embed_rep: float = 0.01,
        tau_embed_rep: float = 1.0,
        margin: float = 0.1,
        alpha_soft: float = 1.0,
        eps: float = 1e-7,
        use_attraction: bool = True,
        use_anchor: bool = True,
        use_spread: bool = True,
    ) -> None:
        super(TotalLossv7, self).__init__()

        self.lambda_attraction = float(lambda_attraction)
        self.lambda_anchor = float(lambda_anchor)
        self.lambda_spread = float(lambda_spread)
        self.lambda_balance = float(lambda_balance)
        self.lambda_sep = float(lambda_sep)
        self.lambda_embedding_reg = float(lambda_embedding_reg)
        self.lambda_embed_rep = float(lambda_embed_rep)
        self.eps = float(eps)

        self.use_attraction = use_attraction
        self.use_anchor = use_anchor
        self.use_spread = use_spread

        self.attraction_fn = AttractionLoss(eps=self.eps)
        self.anchor_fn = AnchorSoftLoss(alpha=alpha_soft, eps=self.eps)
        self.anchor_balance_fn = AnchorBalanceLoss(eps=self.eps)
        self.anchor_spread_fn = AnchorSpreadLoss(eps=self.eps)
        self.anchor_separation_fn = AnchorSeparationLoss(
            margin=margin, eps=self.eps
        )
        self.embedding_reg_fn = EmbeddingRegLoss()
        self.embedding_repulsion_fn = EmbeddingRepulsionLoss(tau=tau_embed_rep, eps=self.eps)

    def forward(
        self,
        embeddings: torch.Tensor,
        edges: torch.Tensor,
        shared_anchors: torch.Tensor,
        client_id: Optional[int] = None,
        round_num: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        """
        Compute total loss and component breakdown.

        Args:
            embeddings:     [N_batch, 2] — mini-batch of latent embeddings
            edges:          [E, 2]       — kNN graph edges (local indices)
            shared_anchors: [K, 2]       — current shared global anchor matrix
                                           (same tensor for every client)
            client_id:      int or None  — used for diagnostic messages
            round_num:      int or None  — used for diagnostic messages

        Returns:
            total:     scalar torch.Tensor (differentiable)
            breakdown: dict of float values for logging
            q:         [N_batch, K] soft assignment probabilities
        """
        # 1. Attraction loss (graph-based neighbourhood preservation)
        l_att = (
            self.attraction_fn(embeddings, edges)
            if self.use_attraction
            else torch.tensor(0.0, device=embeddings.device)
        )

        # 2. Soft anchor assignment loss
        q = torch.zeros(
            (embeddings.shape[0], shared_anchors.shape[0]),
            device=embeddings.device,
        )
        l_anc = torch.tensor(0.0, device=embeddings.device)
        if self.use_anchor and shared_anchors.shape[0] > 0:
            l_anc, q = self.anchor_fn(embeddings, shared_anchors)

        # 3. Anchor balance loss
        l_bal = torch.tensor(0.0, device=embeddings.device)
        if self.lambda_balance > 0 and q is not None and q.shape[1] > 0:
            l_bal, _ = self.anchor_balance_fn(q)

        # 4. Anchor spread loss (local repulsion between shared anchors)
        l_spread = (
            self.anchor_spread_fn(shared_anchors)
            if self.use_spread and shared_anchors.shape[0] > 1
            else torch.tensor(0.0, device=embeddings.device)
        )

        # 5. Anchor separation loss (explicit margin-based separation)
        l_sep = (
            self.anchor_separation_fn(shared_anchors)
            if self.lambda_sep > 0 and shared_anchors.shape[0] > 1
            else torch.tensor(0.0, device=embeddings.device)
        )

        # 6. Embedding regularisation
        l_reg = (
            self.embedding_reg_fn(embeddings)
            if self.lambda_embedding_reg > 0
            else torch.tensor(0.0, device=embeddings.device)
        )

        # 7. Embedding repulsion
        l_embed_rep = (
            self.embedding_repulsion_fn(embeddings, edges)
            if self.lambda_embed_rep > 0
            else torch.tensor(0.0, device=embeddings.device)
        )

        total = (
            self.lambda_attraction * l_att
            + self.lambda_anchor * l_anc
            + self.lambda_spread * l_spread
            + self.lambda_balance * l_bal
            + self.lambda_sep * l_sep
            + self.lambda_embedding_reg * l_reg
            + self.lambda_embed_rep * l_embed_rep
        )

        if torch.isnan(total) or torch.isinf(total):
            raise ValueError(
                f"TotalLossv7 NaN/Inf on client={client_id} round={round_num}."
            )

        breakdown = {
            "total_loss": total.item(),
            "attraction_loss": l_att.item(),
            "anchor_loss": l_anc.item(),
            "balance_loss": l_bal.item(),
            "anchor_spread_loss": l_spread.item(),
            "separation_loss": l_sep.item(),
            "embedding_reg_loss": l_reg.item(),
            "embedding_rep_loss": l_embed_rep.item(),
        }

        return total, breakdown, q
