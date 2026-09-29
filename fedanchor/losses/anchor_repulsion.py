import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class AnchorRepulsionLoss(nn.Module):
    """
    Anchor-based cross-client repulsion loss.

    Legacy (unweighted) form, used when no weights are given:
        L_rep_raw = (1 / (N * |A_other|)) * sum_i sum_{a in A_other} log(1 + 1 / (||z_i - a||^2 + eps))

    Weighted / smoothed form (v2, used when `weights` is given). Each other-client anchor a_k
    summarises a cluster of that client's 2D embeddings: its mass w_k (= cluster size / |D|,
    the FEDNE |D_m'|/|D| scaling) and its spread s_k^2 (mean squared radius of the cluster).
    The anchors then act as a mixture approximation of the other clients' data density, and
    the loss approximates the expected FEDNE negative-sample repulsion from those clients:

        L_rep_raw = (1/N) * sum_i  b * sum_k w_k * log(1 + 1 / (||z_i - a_k||^2 + s_k^2 + eps))

    Sampled form (v2, estimator="sampled"): instead of the closed-form smoothing above, draw
    b "virtual negatives" per point from the anchor mixture  x ~ sum_k w_k N(a_k, s_k^2/2 I)
    and apply exactly the FEDNE negative-sample kernel to them:

        L_rep_raw = (W / N) * sum_i sum_{s=1}^{b} log(1 + 1 / ||z_i - x_is||^2),   W = sum_k w_k = |D_other|/|D|

    This is an unbiased Monte-Carlo estimate of the repulsion the other clients' data would exert
    if their embeddings were distributed like the anchor mixture (it keeps the near-field
    part of the Cauchy kernel that the closed-form smoothing flattens out).

    If normalization == "scale":  L_rep_scaled = scale * L_rep_raw
    """

    def __init__(self, normalization: str = "none", scale: float = 1.0, eps: float = 1e-7,
                 num_negatives: int = 5, estimator: str = "closed_form") -> None:
        super().__init__()
        self.estimator = estimator
        self.normalization = normalization.lower()
        self.scale = float(scale)
        self.eps = float(eps)
        self.num_negatives = int(num_negatives)

    def forward(
        self,
        embeddings: torch.Tensor,
        other_anchors: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        spreads: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        N = embeddings.shape[0]
        K_other = other_anchors.shape[0]

        if N == 0 or K_other == 0:
            zero_tensor = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            return zero_tensor, {"repulsion_raw": 0.0, "repulsion_scaled": 0.0}

        sq_dists = torch.sum((embeddings.unsqueeze(1) - other_anchors.unsqueeze(0)) ** 2, dim=2)  # [N, K]

        if weights is not None and getattr(self, "estimator", "closed_form") == "sampled":
            B, b = N, self.num_negatives
            W = weights.sum()
            probs = weights / W
            k_idx = torch.multinomial(probs, B * b, replacement=True)
            std = torch.sqrt(torch.clamp(spreads[k_idx], min=0.0) / 2.0).unsqueeze(1) if spreads is not None else 0.0
            x = other_anchors[k_idx] + std * torch.randn(B * b, other_anchors.shape[1], device=embeddings.device)
            d2 = torch.sum((embeddings.unsqueeze(1) - x.view(B, b, -1)) ** 2, dim=-1)  # [B, b]
            phi = torch.clamp(1.0 / (1.0 + d2), max=1.0 - self.eps)
            raw_loss = W * torch.mean(-torch.sum(torch.log(1.0 - phi), dim=-1))
            scaled_loss = self.scale * raw_loss if self.normalization == "scale" else raw_loss
            return scaled_loss, {"repulsion_raw": raw_loss.item(), "repulsion_scaled": scaled_loss.item()}

        if weights is None:
            raw_loss = torch.mean(torch.log(1.0 + 1.0 / (sq_dists + self.eps)))
        else:
            s2 = spreads.view(1, -1) if spreads is not None else 0.0
            per_anchor = torch.log(1.0 + 1.0 / (sq_dists + s2 + self.eps))  # [N, K]
            raw_loss = self.num_negatives * torch.mean(per_anchor @ weights.view(-1))

        scaled_loss = self.scale * raw_loss if self.normalization == "scale" else raw_loss

        if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
            raise ValueError("AnchorRepulsionLoss evaluated to NaN or Inf.")

        return scaled_loss, {"repulsion_raw": raw_loss.item(), "repulsion_scaled": scaled_loss.item()}
