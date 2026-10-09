"""
GlobalAnchorSet - Shared Global Anchor Matrix for FEDANCHOR v7
==============================================================

In v7, a single anchor matrix A ∈ R^(K × 2) is maintained on the server
and broadcast to ALL clients at the beginning of every federated round.

Design principles:
- Anchor j has a CONSISTENT geometric meaning across all clients.
- Anchors are geometric reference points, NOT class centroids or prototypes.
- The anchor matrix is learnable (nn.Parameter) and optimized jointly with
  the encoder during local training.
- At the end of each round, the server collects local anchor updates,
  performs Hungarian alignment to resolve permutation ambiguity, then
  performs weighted aggregation to produce the next global anchor matrix.

Mathematical notation:
    A  ∈ R^(K × 2)   — global shared anchor matrix
    a_j ∈ R^2         — j-th anchor coordinate (j = 1..K)
    z_i ∈ R^2         — embedding of sample i
    q_ij              — soft assignment of sample i to anchor j (Student-t)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, Tuple


class GlobalAnchorSet(nn.Module):
    """
    Shared global anchor matrix used by ALL federated clients in v7.

    K anchors, each in R^2 (the embedding space). The anchors are
    trainable nn.Parameters so local gradients can update them during
    local training. The server aggregates these updates with Hungarian
    alignment before broadcasting to the next round.

    Args:
        num_anchors:   K, number of anchor points.
        embedding_dim: dimensionality of the embedding space (default 2).
    """

    def __init__(self, num_anchors: int = 5, embedding_dim: int = 2) -> None:
        super(GlobalAnchorSet, self).__init__()
        self.num_anchors = num_anchors
        self.embedding_dim = embedding_dim
        # Trainable anchor coordinates; initialised to zeros until
        # init_from_embeddings() is called.
        self.anchors = nn.Parameter(
            torch.zeros(num_anchors, embedding_dim, dtype=torch.float32)
        )
        self.initialized: bool = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_from_embeddings(
        self,
        embeddings: torch.Tensor,
        strategy: str = "kmeans",
        seed: int = 42,
    ) -> None:
        """
        Initialize anchor positions from a set of 2D embeddings.

        Called ONCE before federated training on the server side.
        Uses the combined initial embeddings from ALL clients (or a
        deterministic server-side initialization subset).

        Preferred strategy: 'kmeans'
            Fit KMeans(K) on the combined initial embeddings, use
            cluster centres as initial anchor positions.  Does NOT
            use labels — purely geometric.

        Fallback strategy: 'farthest_point'
            Farthest Point Sampling (FPS) — deterministic, label-free.

        Args:
            embeddings: torch.Tensor of shape [N, embedding_dim].
                        Can be combined embeddings from multiple clients.
            strategy:   'kmeans' | 'farthest_point' | 'linspace'
            seed:       random seed for KMeans reproducibility.
        """
        with torch.no_grad():
            N = embeddings.shape[0]
            if N == 0:
                raise ValueError(
                    "Cannot initialize anchors from empty embeddings."
                )

            emb_np = embeddings.detach().cpu().numpy().astype(np.float32)
            K = self.num_anchors

            if strategy == "kmeans":
                n_clusters = min(K, N)
                km = KMeans(
                    n_clusters=n_clusters,
                    random_state=seed,
                    n_init="auto",
                ).fit(emb_np)
                centres = km.cluster_centers_.astype(np.float32)
                # Tile if we have fewer embeddings than K
                if n_clusters < K:
                    repeats = (K // n_clusters) + 1
                    centres = np.tile(centres, (repeats, 1))[:K]
                chosen = torch.tensor(centres, dtype=torch.float32)

            elif strategy == "farthest_point":
                chosen_idx = [0]
                distances = np.full(N, np.inf)
                for _ in range(1, min(K, N)):
                    last = emb_np[chosen_idx[-1]]
                    d = np.sum((emb_np - last) ** 2, axis=1)
                    distances = np.minimum(distances, d)
                    chosen_idx.append(int(np.argmax(distances)))
                fps_pts = emb_np[chosen_idx].astype(np.float32)
                if len(chosen_idx) < K:
                    repeats = (K // len(chosen_idx)) + 1
                    fps_pts = np.tile(fps_pts, (repeats, 1))[:K]
                chosen = torch.tensor(fps_pts, dtype=torch.float32)

            elif strategy == "linspace":
                if N >= K:
                    indices = torch.linspace(0, N - 1, steps=K).long()
                    chosen = embeddings[indices].detach().cpu().clone()
                else:
                    repeats = (K // N) + 1
                    tiled = embeddings.repeat(repeats, 1).detach().cpu()
                    chosen = tiled[:K].clone()

            else:
                raise ValueError(
                    f"Unknown initialization strategy: '{strategy}'. "
                    "Use 'kmeans', 'farthest_point', or 'linspace'."
                )

            self.anchors.data.copy_(chosen)
            self.initialized = True

    # ------------------------------------------------------------------
    # Load / save helpers used by aggregation
    # ------------------------------------------------------------------

    def set_anchors(self, anchor_tensor: torch.Tensor) -> None:
        """
        Replace anchor coordinates with a new tensor (in-place).
        Used by the server after weighted aggregation.

        Args:
            anchor_tensor: shape [K, 2]
        """
        with torch.no_grad():
            self.anchors.data.copy_(
                anchor_tensor.to(self.anchors.device).to(torch.float32)
            )
        self.initialized = True

    def get_anchors(self) -> torch.Tensor:
        """
        Return a detached CPU copy of the current anchor coordinates.
        Shape: [K, 2]
        """
        return self.anchors.detach().cpu().clone()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self) -> torch.Tensor:
        """
        Returns the current anchor tensor of shape [K, embedding_dim].
        """
        return self.anchors
