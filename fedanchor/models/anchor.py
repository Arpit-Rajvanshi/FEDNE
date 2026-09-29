import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional

class AnchorSet(nn.Module):
    """
    AnchorSet represents K learnable 2D points A_m in the low-dimensional embedding space.
    A_m = {a_m1, a_m2, ..., a_mK} where a_mj in R^2.
    
    Supports 3 initialization strategies:
    1. 'kmeans': K-Means clustering on initial local embeddings Z_m^(0)
    2. 'farthest_point': Farthest Point Sampling (FPS) algorithm on Z_m^(0)
    3. 'representative_points': Linspace index sampling (for exact baseline comparison)
    """
    def __init__(self, num_anchors: int = 5, embedding_dim: int = 2) -> None:
        super(AnchorSet, self).__init__()
        self.num_anchors = num_anchors
        self.embedding_dim = embedding_dim
        # Initialize as trainable nn.Parameter
        self.anchors = nn.Parameter(torch.zeros(num_anchors, embedding_dim, dtype=torch.float32))
        self.initialized = False

    def init_from_embeddings(
        self,
        embeddings: torch.Tensor,
        strategy: str = "kmeans",
        seed: int = 42
    ) -> None:
        """
        Initialize anchor positions using client's initial local embeddings Z_m^(0).
        
        Args:
            embeddings: torch.Tensor of shape [N, 2]
            strategy: 'kmeans', 'farthest_point', or 'representative_points'
            seed: Random seed for deterministic K-Means / sampling
        """
        with torch.no_grad():
            N = embeddings.shape[0]
            if N == 0:
                raise ValueError("Cannot initialize anchors from empty embeddings.")
                
            emb_np = embeddings.detach().cpu().numpy()
            
            if strategy == "kmeans":
                n_clusters = min(self.num_anchors, N)
                kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto').fit(emb_np)
                centroids = kmeans.cluster_centers_.astype(np.float32)
                
                if n_clusters < self.num_anchors:
                    repeats = (self.num_anchors // n_clusters) + 1
                    centroids = np.tile(centroids, (repeats, 1))[:self.num_anchors]
                    
                chosen_points = torch.tensor(centroids, dtype=torch.float32, device=embeddings.device)

            elif strategy == "farthest_point":
                # Farthest Point Sampling (FPS) algorithm
                chosen_indices = [0]  # Start deterministically at first point
                distances = np.full(N, np.inf)
                
                for _ in range(1, min(self.num_anchors, N)):
                    last_point = emb_np[chosen_indices[-1]]
                    dist_to_last = np.sum((emb_np - last_point) ** 2, axis=1)
                    distances = np.minimum(distances, dist_to_last)
                    farthest_idx = int(np.argmax(distances))
                    chosen_indices.append(farthest_idx)
                    
                fps_points = emb_np[chosen_indices].astype(np.float32)
                if len(chosen_indices) < self.num_anchors:
                    repeats = (self.num_anchors // len(chosen_indices)) + 1
                    fps_points = np.tile(fps_points, (repeats, 1))[:self.num_anchors]
                    
                chosen_points = torch.tensor(fps_points, dtype=torch.float32, device=embeddings.device)

            elif strategy == "representative_points":
                if N >= self.num_anchors:
                    indices = torch.linspace(0, N - 1, steps=self.num_anchors).long()
                    chosen_points = embeddings[indices].clone()
                else:
                    repeats = (self.num_anchors // N) + 1
                    tiled = embeddings.repeat(repeats, 1)
                    chosen_points = tiled[:self.num_anchors].clone()
            else:
                raise ValueError(f"Unknown initialization strategy: {strategy}")

            self.anchors.copy_(chosen_points)
            self.initialized = True

    def forward(self) -> torch.Tensor:
        """
        Returns tensor of anchors of shape [K, 2].
        """
        return self.anchors


def summarize_embeddings_as_anchors(
    embeddings: np.ndarray,
    num_anchors: int,
    seed: int = 42,
    init_centers: Optional[np.ndarray] = None,
    max_points: int = 20000,
):
    """
    v2 anchors: summarise a client's 2D embedding distribution with K weighted, smoothed anchors.

    Runs K-Means on the client's current 2D embeddings and returns, per anchor:
      centers [K, 2]  - cluster centroid (the anchor position a_k)
      counts  [K]     - number of local points assigned to a_k (anchor mass)
      spreads [K]     - mean squared distance of assigned points to a_k (anchor smoothing s_k^2)

    Anchors are re-fitted every round instead of being learned by gradient descent, so they
    cannot collapse onto a single dominant anchor. Only these 4*K scalars are shared.
    """
    rng = np.random.RandomState(seed)
    Z = np.asarray(embeddings, dtype=np.float64)
    N = Z.shape[0]
    K = int(min(num_anchors, N))
    fit_Z = Z[rng.choice(N, max_points, replace=False)] if N > max_points else Z

    if init_centers is not None and init_centers.shape == (K, Z.shape[1]):
        km = KMeans(n_clusters=K, init=init_centers.astype(np.float64), n_init=1, max_iter=100, random_state=seed)
    else:
        km = KMeans(n_clusters=K, n_init=3, max_iter=100, random_state=seed)
    km.fit(fit_Z)
    centers = km.cluster_centers_

    labels = km.predict(Z)
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    sq = np.sum((Z - centers[labels]) ** 2, axis=1)
    spreads = np.bincount(labels, weights=sq, minlength=K) / np.maximum(counts, 1.0)

    if K < num_anchors:  # pad with zero-mass anchors
        pad = num_anchors - K
        centers = np.vstack([centers, np.tile(centers[:1], (pad, 1))])
        counts = np.concatenate([counts, np.zeros(pad)])
        spreads = np.concatenate([spreads, np.zeros(pad)])

    return centers.astype(np.float32), counts.astype(np.float32), spreads.astype(np.float32)
