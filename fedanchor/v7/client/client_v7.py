"""
AnchorClientv7 - Federated Client for FEDANCHOR v7
====================================================

v7 Key Difference from V5
--------------------------
Each client receives a COPY of the SAME global anchor matrix at the
beginning of every federated round and optimises it locally.

At the end of local training, the client returns:
  - Updated encoder state dict
  - Updated anchor tensor  (to be aligned + averaged by server)
  - Loss metrics
  - Per-step diagnostics

The client does NOT maintain its own persistent anchor set across rounds.
At the start of each round, the anchor copy is always reset to whatever the
server broadcasts (i.e., the global anchor matrix after last round's
aggregation).

This is intentional: the server is the authoritative source of the global
anchor matrix.  The client only temporarily owns a local copy for gradient
computation.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, Any, Optional, List

from fedanchor.models.encoder import Encoder
from fedanchor.v7.models.global_anchor import GlobalAnchorSet
from fedanchor.graph.knn import construct_knn_graph
from fedanchor.v7.losses.total_loss_v7 import TotalLossv7


class AnchorClientv7:
    """
    Federated Anchor Client v7 (Shared Global Anchors).

    Each client holds:
      - local_data: private training samples (X_train, y_train)
      - encoder: local copy of the global encoder (loaded from server each round)
      - local_anchor_copy: a LOCAL COPY of the global anchor matrix, reset
        every round from the server's broadcast.
    """

    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k_neighbors: int = 5,
        num_anchors: int = 5,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(X_train)
        self.device = torch.device(device)
        self.k_neighbors = k_neighbors
        self.num_anchors = num_anchors
        self.seed = seed

        # Move training data to device tensors
        self.X_tensor = torch.tensor(
            self.X_train, dtype=torch.float32
        ).to(self.device)
        self.y_tensor = torch.tensor(
            self.y_train, dtype=torch.int64
        ).to(self.device)

        # Construct kNN graph (over raw feature space)
        self.edges_np = construct_knn_graph(self.X_train, k=self.k_neighbors)
        self.edges_tensor = torch.tensor(
            self.edges_np, dtype=torch.int64
        ).to(self.device)

        # Dataset for mini-batch sampling of edges
        self.dataset = TensorDataset(self.edges_tensor)

        # Local encoder (loaded fresh from global state at start of each round)
        self.encoder = Encoder(
            input_dim=self.X_train.shape[1], embedding_dim=2
        ).to(self.device)

        # Local COPY of global anchors — reset every round from server broadcast
        self.local_anchor_copy = GlobalAnchorSet(
            num_anchors=num_anchors, embedding_dim=2
        ).to(self.device)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _get_batch_embeddings(
        self, batch_edges: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a batch of edge indices (global), collect the unique
        node indices, extract their feature vectors, and compute
        embeddings.  Return embeddings and remapped local edge indices.
        """
        unique_nodes, inverse_indices = torch.unique(
            batch_edges, return_inverse=True
        )
        batch_x = self.X_tensor[unique_nodes]
        local_edges = inverse_indices.view(-1, 2)
        embeddings = self.encoder(batch_x)
        return embeddings, local_edges

    # ------------------------------------------------------------------
    # Per-epoch training
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loss_fn: TotalLossv7,
        optimizer: optim.Optimizer,
        dataloader: DataLoader,
        max_grad_norm: float = 1.0,
        round_num: int = 0,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        One full epoch of local optimisation over mini-batches.

        Both encoder parameters AND the local copy of the shared anchor
        matrix receive gradient updates.

        Returns:
            accumulated_metrics: dict of averaged loss components
            step_diagnostics:    list of per-batch diagnostic dicts
        """
        self.encoder.train()
        self.local_anchor_copy.train()

        accumulated_metrics: Dict[str, float] = {}
        step_diagnostics: List[Dict[str, float]] = []
        num_batches = len(dataloader)

        for batch_idx, (batch_edges,) in enumerate(dataloader):
            optimizer.zero_grad()

            # Record anchor positions before the update step
            anchors_before = self.local_anchor_copy().detach().clone()

            embeddings, local_edges = self._get_batch_embeddings(batch_edges)
            shared_anc = self.local_anchor_copy()

            total, breakdown, _ = loss_fn(
                embeddings=embeddings,
                edges=local_edges,
                shared_anchors=shared_anc,
                client_id=self.client_id,
                round_num=round_num,
            )

            total.backward()

            # Diagnostic: anchor gradient BEFORE clipping
            if self.local_anchor_copy.anchors.grad is not None:
                pre_clip_norm = torch.norm(
                    self.local_anchor_copy.anchors.grad
                ).item()
                max_abs_grad = torch.max(
                    torch.abs(self.local_anchor_copy.anchors.grad)
                ).item()
            else:
                pre_clip_norm = 0.0
                max_abs_grad = 0.0

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), max_norm=max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.local_anchor_copy.parameters(), max_norm=max_grad_norm
            )

            if self.local_anchor_copy.anchors.grad is not None:
                post_clip_norm = torch.norm(
                    self.local_anchor_copy.anchors.grad
                ).item()
            else:
                post_clip_norm = 0.0

            optimizer.step()

            anchors_after = self.local_anchor_copy().detach().clone()
            displacement = torch.norm(anchors_after - anchors_before).item()

            step_diagnostics.append(
                {
                    "batch_idx": batch_idx,
                    "pre_clip_grad_norm": pre_clip_norm,
                    "post_clip_grad_norm": post_clip_norm,
                    "max_abs_grad": max_abs_grad,
                    "displacement": displacement,
                    "nonzero_grad": 1.0 if pre_clip_norm > 0 else 0.0,
                    "nonzero_update": 1.0 if displacement > 0 else 0.0,
                    "l_sep": breakdown.get("separation_loss", 0.0),
                    "l_spread": breakdown.get("anchor_spread_loss", 0.0),
                    "l_balance": breakdown.get("balance_loss", 0.0),
                }
            )

            for k, v in breakdown.items():
                accumulated_metrics[k] = (
                    accumulated_metrics.get(k, 0.0) + v / num_batches
                )

        return accumulated_metrics, step_diagnostics

    # ------------------------------------------------------------------
    # Main local training entry point (called by trainer per round)
    # ------------------------------------------------------------------

    def local_train(
        self,
        global_encoder_state: Dict[str, torch.Tensor],
        global_anchor_tensor: torch.Tensor,
        config: Dict[str, Any],
        round_num: int = 0,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        Dict[str, float],
        List[Dict[str, float]],
        float,
    ]:
        """
        Perform local training for `local_epochs` rounds using mini-batches.

        Args:
            global_encoder_state: server's current global encoder state dict
            global_anchor_tensor: shape [K, 2] — server's current global anchors
                                  (ALL clients receive the SAME tensor)
            config:               training configuration dict
            round_num:            current federated round index (for diagnostics)

        Returns:
            updated_encoder_state: dict[str, Tensor] — updated local encoder
            updated_anchors:       shape [K, 2]      — locally updated anchors
            accumulated_metrics:   dict[str, float]  — averaged loss breakdown
            all_step_diagnostics:  list of per-step dicts
            init_consistency_max:  float — max |client_anchors - global_anchors|
                                   BEFORE training (should be ≈ 0 by design)
        """
        # ---- Step 1: Load global encoder into local copy ----
        self.encoder.load_state_dict(global_encoder_state)

        # ---- Step 2: Load global anchors into local copy ----
        # Verify that initialization was received correctly (diagnostic)
        self.local_anchor_copy.set_anchors(global_anchor_tensor)
        init_consistency_max = float(
            torch.max(
                torch.abs(
                    self.local_anchor_copy.get_anchors()
                    - global_anchor_tensor.cpu()
                )
            ).item()
        )
        # init_consistency_max should be ≈ 0; large values indicate a bug.

        # ---- Step 3: Build optimiser ----
        lr = config.get("training", {}).get("learning_rate", 0.001)
        max_grad_norm = config.get("training", {}).get("max_grad_norm", 1.0)
        local_epochs = config.get("training", {}).get("local_epochs", 1)
        batch_size = config.get("training", {}).get("batch_size", 256)

        optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.local_anchor_copy.parameters()),
            lr=lr,
        )

        # ---- Step 4: Build dataloader ----
        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True
        )

        # ---- Step 5: Build loss function ----
        loss_cfg = config.get("loss", {})
        loss_fn = TotalLossv7(
            lambda_attraction=loss_cfg.get("lambda_attraction", 1.0),
            lambda_anchor=loss_cfg.get("lambda_anchor", 1.0),
            lambda_spread=loss_cfg.get("lambda_spread", 1.0),
            lambda_balance=loss_cfg.get("lambda_balance", 1.0),
            lambda_sep=loss_cfg.get("lambda_sep", 0.1),
            lambda_embedding_reg=loss_cfg.get("lambda_embedding_reg", 0.0),
            lambda_embed_rep=loss_cfg.get("lambda_embed_rep", 0.01),
            tau_embed_rep=loss_cfg.get("tau_embed_rep", 1.0),
            margin=loss_cfg.get("margin", 0.1),
            alpha_soft=loss_cfg.get("alpha_soft", 1.0),
            eps=loss_cfg.get("eps", 1e-7),
            use_attraction=config.get("ablation", {}).get(
                "use_attraction", True
            ),
            use_anchor=config.get("ablation", {}).get("use_anchor", True),
            use_spread=config.get("ablation", {}).get("use_spread", True),
        )

        # ---- Step 6: Local training loop ----
        accumulated_metrics: Dict[str, float] = {}
        all_step_diagnostics: List[Dict[str, float]] = []

        for _epoch in range(local_epochs):
            metrics, step_diags = self._train_epoch(
                loss_fn=loss_fn,
                optimizer=optimizer,
                dataloader=dataloader,
                max_grad_norm=max_grad_norm,
                round_num=round_num,
            )
            for k, v in metrics.items():
                accumulated_metrics[k] = (
                    accumulated_metrics.get(k, 0.0) + v / local_epochs
                )
            all_step_diagnostics.extend(step_diags)

        # ---- Step 7: Return updated states ----
        updated_encoder_state = {
            k: v.cpu().clone() for k, v in self.encoder.state_dict().items()
        }
        updated_anchors = self.local_anchor_copy.get_anchors()  # [K, 2] CPU

        return (
            updated_encoder_state,
            updated_anchors,
            accumulated_metrics,
            all_step_diagnostics,
            init_consistency_max,
        )

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    def get_embeddings_eval(self) -> np.ndarray:
        """
        Compute 2D embeddings for all local training samples using the
        current encoder state.  Used for evaluation and visualisation.
        """
        self.encoder.eval()
        with torch.no_grad():
            emb = self.encoder(self.X_tensor).cpu().numpy()
        return emb
