import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, Any, Optional
from fedanchor.models.encoder import Encoder
from fedanchor.models.anchor import AnchorSet, summarize_embeddings_as_anchors
from fedanchor.graph.knn import construct_knn_graph
from fedanchor.losses.total_loss import TotalLoss
from fedanchor.losses.local_repulsion import LocalNegativeRepulsionLoss
from fedanchor.losses.anchor_repulsion import AnchorRepulsionLoss

class AnchorClient:
    """
    Federated Anchor Client.
    Holds local private dataset, constructs local kNN graph, maintains local encoder and learnable AnchorSet.
    """
    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k_neighbors: int = 5,
        num_anchors: int = 5,
        device: str = "cpu",
        seed: int = 42
    ) -> None:
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(X_train)
        self.device = torch.device(device)
        self.k_neighbors = k_neighbors
        self.num_anchors = num_anchors
        self.seed = seed
        
        # Convert local data to PyTorch tensor
        self.X_tensor = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        self.y_tensor = torch.tensor(self.y_train, dtype=torch.int64).to(self.device)
        
        # Construct local kNN graph (strictly private to client)
        self.edges_np = construct_knn_graph(self.X_train, k=self.k_neighbors)
        self.edges_tensor = torch.tensor(self.edges_np, dtype=torch.int64).to(self.device)
        
        # Initialize local encoder and AnchorSet
        self.encoder = Encoder(input_dim=self.X_train.shape[1], embedding_dim=2).to(self.device)
        self.anchor_set = AnchorSet(num_anchors=num_anchors, embedding_dim=2).to(self.device)

        # v2 ("neighbor_embedding" mode) state
        self.total_samples = self.num_samples  # |D|, set by the trainer to the federation-wide size
        self.anchor_meta: Optional[Dict[str, torch.Tensor]] = None  # counts / spreads of own anchors
        self._rng = torch.Generator().manual_seed(seed + 1000 * client_id)

    def initialize_anchors(self, strategy: str = "kmeans") -> None:
        """
        Initialize anchors using current local embeddings.
        """
        self.encoder.eval()
        with torch.no_grad():
            initial_z = self.encoder(self.X_tensor)
            self.anchor_set.init_from_embeddings(initial_z, strategy=strategy, seed=self.seed + self.client_id)

    def train_epoch(
        self,
        loss_fn: TotalLoss,
        optimizer: optim.Optimizer,
        other_anchors: torch.Tensor,
        max_grad_norm: float = 1.0,
        round_num: int = 0
    ) -> Dict[str, float]:
        """
        Runs one full epoch of local optimization over local embeddings and anchors.
        """
        self.encoder.train()
        self.anchor_set.train()
        
        optimizer.zero_grad()
        
        embeddings = self.encoder(self.X_tensor)
        local_anchors = self.anchor_set()
        
        loss, breakdown = loss_fn(
            embeddings=embeddings,
            edges=self.edges_tensor,
            local_anchors=local_anchors,
            other_anchors=other_anchors,
            client_id=self.client_id,
            round_num=round_num
        )
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.anchor_set.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        
        return breakdown

    def local_train(
        self,
        global_encoder_state: Dict[str, torch.Tensor],
        other_anchors_tensor: torch.Tensor,
        config: Dict[str, Any],
        round_num: int = 0,
        other_anchor_meta: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, float]]:
        """
        Perform local training for local_epochs rounds.

        training.mode == "neighbor_embedding" (v2, recommended) -> mini-batch neighbor embedding
        with local negative-sample repulsion + weighted anchor cross-client repulsion.
        Any other value (default "fullbatch") -> original Phase-2 prototype behaviour.
        """
        if config.get("training", {}).get("mode", "fullbatch") == "neighbor_embedding":
            return self._local_train_ne(global_encoder_state, other_anchors_tensor, config,
                                        round_num, other_anchor_meta)

        self.encoder.load_state_dict(global_encoder_state)
        
        if not self.anchor_set.initialized:
            init_strat = config.get("anchors", {}).get("initialization", "kmeans")
            self.initialize_anchors(strategy=init_strat)
            
        other_anchors_device = other_anchors_tensor.to(self.device)
        
        lr = config.get("training", {}).get("learning_rate", 0.001)
        max_grad_norm = config.get("training", {}).get("max_grad_norm", 1.0)
        local_epochs = config.get("training", {}).get("local_epochs", 1)
        
        loss_cfg = config.get("loss", {})
        rep_cfg = loss_cfg.get("anchor_repulsion", {})
        bal_cfg = loss_cfg.get("balancing", {})
        model_cfg = config.get("model", {})
        emb_reg_cfg = model_cfg.get("embedding_regularization", {})
        
        loss_fn = TotalLoss(
            lambda_attraction=loss_cfg.get("lambda_attraction", 1.0),
            lambda_anchor=loss_cfg.get("lambda_anchor", 1.0),
            lambda_anchor_repulsion=loss_cfg.get("lambda_anchor_repulsion", 1.0),
            lambda_embedding_reg=emb_reg_cfg.get("coefficient", 0.0) if emb_reg_cfg.get("enabled", False) else 0.0,
            normalization=rep_cfg.get("normalization", "none"),
            scale=rep_cfg.get("scale", 1.0),
            eps=loss_cfg.get("eps", 1e-7),
            balancing_enabled=bal_cfg.get("enabled", False),
            use_attraction=config.get("ablation", {}).get("use_attraction", True),
            use_anchor=config.get("ablation", {}).get("use_anchor", True),
            use_anchor_repulsion=config.get("ablation", {}).get("use_anchor_repulsion", True)
        )
        
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.anchor_set.parameters()),
            lr=lr
        )
        
        accumulated_metrics: Dict[str, float] = {}
        for epoch in range(local_epochs):
            metrics = self.train_epoch(
                loss_fn=loss_fn,
                optimizer=optimizer,
                other_anchors=other_anchors_device,
                max_grad_norm=max_grad_norm,
                round_num=round_num
            )
            for k, v in metrics.items():
                accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v / local_epochs
                
        updated_encoder_state = {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()}
        updated_anchors = self.anchor_set().detach().cpu().clone()
        
        return updated_encoder_state, updated_anchors, accumulated_metrics

    # ------------------------------------------------------------------
    # v2: mini-batch neighbor embedding with anchor-based cross-client repulsion
    # ------------------------------------------------------------------
    @torch.no_grad()
    def embed_local(self, batch_size: int = 4096) -> torch.Tensor:
        self.encoder.eval()
        out = [self.encoder(self.X_tensor[i:i + batch_size]) for i in range(0, self.num_samples, batch_size)]
        return torch.cat(out, dim=0)

    def refresh_anchors(self, num_anchors: int) -> torch.Tensor:
        """Re-fit this client's anchors (K-Means summary of its current 2D embeddings)."""
        Z = self.embed_local().cpu().numpy()
        prev = self.anchor_set.anchors.detach().cpu().numpy() if self.anchor_set.initialized else None
        centers, counts, spreads = summarize_embeddings_as_anchors(
            Z, num_anchors, seed=self.seed + self.client_id, init_centers=prev)
        with torch.no_grad():
            if self.anchor_set.anchors.shape != (num_anchors, 2):
                self.anchor_set.anchors = torch.nn.Parameter(torch.zeros(num_anchors, 2))
            self.anchor_set.anchors.copy_(torch.from_numpy(centers))
        self.anchor_set.initialized = True
        self.anchor_meta = {"counts": torch.from_numpy(counts), "spreads": torch.from_numpy(spreads)}
        return torch.from_numpy(centers).clone()

    def _local_train_ne(self, global_encoder_state, other_anchors_tensor, config, round_num, other_anchor_meta):
        self.encoder.load_state_dict(global_encoder_state)
        tr = config.get("training", {})
        loss_cfg = config.get("loss", {})
        rep_cfg = loss_cfg.get("anchor_repulsion", {})
        graph_cfg = config.get("graph", {})

        lr = tr.get("learning_rate", 0.001)
        for r_decay in tr.get("lr_decay_rounds", []) or []:
            if round_num >= r_decay:
                lr *= tr.get("lr_decay_factor", 0.1)
        batch_size = tr.get("batch_size", 512)
        local_epochs = tr.get("local_epochs", 1)
        max_grad_norm = tr.get("max_grad_norm", 1.0)
        b = graph_cfg.get("negative_samples", 5)
        num_anchors = config.get("anchors", {}).get("num_anchors", 32)
        start_round = rep_cfg.get("start_round", 1)

        lam_att = loss_cfg.get("lambda_attraction", 1.0)
        lam_local = loss_cfg.get("lambda_local_repulsion", 1.0)
        lam_anc_rep = loss_cfg.get("lambda_anchor_repulsion", 1.0)
        spread_scale = rep_cfg.get("spread_scale", 1.0)

        local_rep_fn = LocalNegativeRepulsionLoss(eps=loss_cfg.get("eps", 1e-7))
        anc_rep_fn = AnchorRepulsionLoss(normalization=rep_cfg.get("normalization", "none"),
                                         scale=rep_cfg.get("scale", 1.0), num_negatives=b,
                                         estimator=rep_cfg.get("estimator", "closed_form"))
        local_scale = self.num_samples / float(self.total_samples)  # FEDNE |D_m| / |D|

        use_anchor_rep = (lam_anc_rep > 0 and other_anchors_tensor is not None
                          and other_anchors_tensor.shape[0] > 0 and other_anchor_meta is not None
                          and round_num >= start_round)
        if use_anchor_rep:
            A = other_anchors_tensor.to(self.device).detach()
            W = other_anchor_meta["weights"].to(self.device).detach()
            S = (spread_scale * other_anchor_meta["spreads"]).to(self.device).detach()

        optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        E = self.edges_tensor.shape[0]
        sums = {"att": 0.0, "loc": 0.0, "anc_raw": 0.0, "anc": 0.0, "tot": 0.0}
        steps = 0
        self.encoder.train()
        for _ in range(local_epochs):
            perm = torch.randperm(E, generator=self._rng)
            for start in range(0, E, batch_size):
                idx = perm[start:start + batch_size].to(self.device)
                src = self.edges_tensor[idx, 0]
                dst = self.edges_tensor[idx, 1]
                B = src.shape[0]
                neg = torch.randint(0, self.num_samples, (B * b,), generator=self._rng).to(self.device)

                z = self.encoder(torch.cat([self.X_tensor[src], self.X_tensor[dst], self.X_tensor[neg]], 0))
                z_i, z_j, z_neg = z[:B], z[B:2 * B], z[2 * B:].view(B, b, -1)

                # attraction: -log phi(z_i, z_j) = log(1 + ||z_i - z_j||^2)
                l_att = torch.mean(torch.log1p(torch.sum((z_i - z_j) ** 2, dim=1)))
                l_loc = local_rep_fn(z_i, z_neg)
                loss = lam_att * l_att + lam_local * local_scale * l_loc
                if use_anchor_rep:
                    l_anc, rep_d = anc_rep_fn(z_i, A, weights=W, spreads=S)
                    loss = loss + lam_anc_rep * l_anc
                    sums["anc_raw"] += rep_d["repulsion_raw"]
                    sums["anc"] += rep_d["repulsion_scaled"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=max_grad_norm)
                optimizer.step()

                sums["att"] += l_att.item()
                sums["loc"] += l_loc.item()
                sums["tot"] += loss.item()
                steps += 1

        new_anchors = self.refresh_anchors(num_anchors)
        n = max(steps, 1)
        att = sums["att"] / n
        anc_scaled = sums["anc"] / n
        metrics = {
            "total_loss": sums["tot"] / n,
            "attraction_loss": att,
            "local_repulsion_loss": sums["loc"] / n,
            "anchor_loss": 0.0,
            "anchor_repulsion_raw": sums["anc_raw"] / n,
            "anchor_repulsion_scaled": anc_scaled,
            "embedding_reg_loss": 0.0,
            "repulsion_to_attraction_ratio": anc_scaled / (att + 1e-7),
            "repulsion_to_anchor_ratio": 0.0,
        }
        updated_encoder_state = {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()}
        return updated_encoder_state, new_anchors, metrics
