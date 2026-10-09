"""
v7ServerWithSharedAnchors
=========================

The v7 server is fundamentally different from V5:

V5 server:
  - Stores separate anchor matrices per client (client_anchors dict)
  - Never aggregates anchors

v7 server:
  - Stores ONE global anchor matrix A ∈ R^(K × 2)
  - Broadcasts the SAME A to every client before each round
  - After each round, collects client anchor updates,
    performs Hungarian alignment, computes weighted average,
    stores the new A_global

Global anchor lifecycle per round:
  ┌─────────────────────────────────────────────────────┐
  │  server.get_global_anchor_tensor()  → A_t           │
  │     ↓ broadcast to all clients                      │
  │  client.local_train(..., A_t, ...)                  │
  │     ↓ returns A_t^m (locally updated copies)        │
  │  server.aggregate_anchors({cid: A_t^m, ...}, ...)  │
  │     ↓ Hungarian align + weighted average            │
  │  stores A_{t+1}                                     │
  └─────────────────────────────────────────────────────┘
"""

import copy
import torch
import numpy as np
from typing import Dict, List, Any, Tuple

from fedanchor.models.encoder import Encoder
from fedanchor.v7.models.global_anchor import GlobalAnchorSet
from fedanchor.v7.aggregation.hungarian_align import (
    aggregate_anchors_with_alignment,
)


class v7ServerWithSharedAnchors:
    """
    Federated server for FEDANCHOR v7.

    Maintains:
        global_encoder:       global encoder (aggregated each round via FedAvg)
        global_anchor_set:    single shared GlobalAnchorSet object

    Provides:
        get_global_encoder_state()    → state dict
        get_global_anchor_tensor()    → tensor [K, 2]
        aggregate_encoders(...)       → FedAvg encoder
        aggregate_anchors(...)        → Hungarian-aligned anchor aggregation
        compute_communication_bytes() → communication analysis
    """

    def __init__(
        self,
        input_dim: int = 784,
        embedding_dim: int = 2,
        num_anchors: int = 5,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.num_anchors = num_anchors
        self.embedding_dim = embedding_dim

        self.global_encoder = Encoder(
            input_dim=input_dim, embedding_dim=embedding_dim
        ).to(self.device)

        self.global_anchor_set = GlobalAnchorSet(
            num_anchors=num_anchors, embedding_dim=embedding_dim
        ).to(self.device)

        # Track anchor history for diagnostics (anchor trajectory)
        self.anchor_history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Global anchor initialisation
    # ------------------------------------------------------------------

    def initialize_global_anchors(
        self,
        combined_embeddings: torch.Tensor,
        strategy: str = "kmeans",
        seed: int = 42,
    ) -> None:
        """
        Initialize the shared global anchor matrix ONCE before training.

        Called on the server using the combined initial embeddings from
        ALL clients (or a deterministic subset).

        Args:
            combined_embeddings: shape [N_total, 2] — concatenated initial
                                 embeddings from all clients (or a subset).
            strategy:            'kmeans' | 'farthest_point' | 'linspace'
            seed:                random seed for reproducibility
        """
        self.global_anchor_set.init_from_embeddings(
            combined_embeddings, strategy=strategy, seed=seed
        )
        # Record initial state
        self.anchor_history.append(self.global_anchor_set.get_anchors().numpy())

    # ------------------------------------------------------------------
    # Getters for clients
    # ------------------------------------------------------------------

    def get_global_encoder_state(self) -> Dict[str, torch.Tensor]:
        """Return a CPU copy of the global encoder state dict."""
        return {
            k: v.cpu().clone()
            for k, v in self.global_encoder.state_dict().items()
        }

    def get_global_anchor_tensor(self) -> torch.Tensor:
        """
        Return a detached CPU tensor of the global anchors.
        Shape: [K, 2]

        All clients receive the SAME tensor before local training begins.
        """
        return self.global_anchor_set.get_anchors()

    # ------------------------------------------------------------------
    # Aggregation methods
    # ------------------------------------------------------------------

    def aggregate_encoders(
        self,
        client_encoder_states: List[Dict[str, torch.Tensor]],
        client_sample_counts: List[int],
    ) -> None:
        """FedAvg encoder aggregation (unchanged from V5)."""
        total_samples = sum(client_sample_counts)
        if total_samples == 0:
            return

        avg_state = copy.deepcopy(client_encoder_states[0])
        for key in avg_state:
            avg_state[key] = torch.zeros_like(
                avg_state[key], dtype=torch.float32
            )

        for state, n_m in zip(client_encoder_states, client_sample_counts):
            w = float(n_m) / float(total_samples)
            for key in avg_state:
                avg_state[key] += state[key].to(torch.float32) * w

        self.global_encoder.load_state_dict(avg_state)

    def aggregate_anchors(
        self,
        client_anchor_updates: Dict[int, torch.Tensor],
        client_sample_counts: Dict[int, int],
    ) -> Tuple[Dict[int, float], Dict[int, np.ndarray]]:
        """
        Aggregate client anchor updates into a new global anchor matrix.

        1. Convert tensors to numpy.
        2. Hungarian-align each client's anchors to the CURRENT global anchors.
        3. Weighted average → new global anchors.
        4. Update global_anchor_set.
        5. Record history.

        Args:
            client_anchor_updates: {cid → tensor [K, 2]}
            client_sample_counts:  {cid → int}

        Returns:
            alignment_costs: {cid → float} — mean squared alignment distance
            permutations:    {cid → array [K]} — applied permutations
        """
        current_global = self.global_anchor_set.get_anchors().numpy()

        np_updates = {
            cid: t.detach().cpu().numpy()
            for cid, t in client_anchor_updates.items()
        }

        new_anchors_np, alignment_costs, permutations = (
            aggregate_anchors_with_alignment(
                global_anchors=current_global,
                client_anchor_updates=np_updates,
                client_sample_counts=client_sample_counts,
            )
        )

        # Update global anchor set
        new_tensor = torch.tensor(new_anchors_np, dtype=torch.float32)
        self.global_anchor_set.set_anchors(new_tensor)

        # Record trajectory
        self.anchor_history.append(new_anchors_np.copy())

        return alignment_costs, permutations

    # ------------------------------------------------------------------
    # Communication analysis
    # ------------------------------------------------------------------

    def compute_communication_bytes(
        self,
        num_clients: int,
        precision_bytes: int = 4,
    ) -> Dict[str, Any]:
        """
        Compute per-round communication volume.

        In v7, each client receives:
          - global encoder parameters (download)
          - shared anchor matrix [K, 2] (download)

        Each client sends:
          - updated encoder parameters (upload)
          - locally updated anchor matrix (upload)

        Total anchor scalars: K * 2 (same for upload and download).
        """
        enc_params = sum(
            p.numel() for p in self.global_encoder.parameters()
        )
        enc_bytes = enc_params * precision_bytes

        anc_scalars = self.num_anchors * self.embedding_dim
        anc_bytes = anc_scalars * precision_bytes

        # Per-round totals
        upload_total = num_clients * (enc_bytes + anc_bytes)
        download_total = num_clients * (enc_bytes + anc_bytes)
        round_total = upload_total + download_total

        return {
            "encoder_parameter_count": enc_params,
            "encoder_bytes_per_client": enc_bytes,
            "anchor_scalar_count": anc_scalars,
            "anchor_bytes": anc_bytes,
            "upload_bytes_total": upload_total,
            "download_bytes_total": download_total,
            "total_round_bytes": round_total,
            "anchor_to_encoder_scalar_ratio": float(anc_scalars) / float(enc_params),
        }
