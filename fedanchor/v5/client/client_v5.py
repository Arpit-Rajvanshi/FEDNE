import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
from fedanchor.models.encoder import Encoder
from fedanchor.models.anchor import AnchorSet
from fedanchor.graph.knn import construct_knn_graph
from fedanchor.v4.losses.total_loss_v4 import TotalLossV4

class AnchorClientV5:
    """
    Federated Anchor Client V5 (Mini-Batch & Optimizer State Validation).
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
        
        self.X_tensor = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        self.y_tensor = torch.tensor(self.y_train, dtype=torch.int64).to(self.device)
        
        self.edges_np = construct_knn_graph(self.X_train, k=self.k_neighbors)
        self.edges_tensor = torch.tensor(self.edges_np, dtype=torch.int64).to(self.device)
        
        self.dataset = TensorDataset(self.edges_tensor)
        
        self.encoder = Encoder(input_dim=self.X_train.shape[1], embedding_dim=2).to(self.device)
        self.anchor_set = AnchorSet(num_anchors=num_anchors, embedding_dim=2).to(self.device)
        
        self.optimizer: Optional[optim.Optimizer] = None

    def initialize_anchors(self, strategy: str = "kmeans") -> None:
        self.encoder.eval()
        with torch.no_grad():
            initial_z = self.encoder(self.X_tensor)
            self.anchor_set.init_from_embeddings(initial_z, strategy=strategy, seed=self.seed + self.client_id)

    def train_epoch(
        self,
        loss_fn: TotalLossV4,
        optimizer: optim.Optimizer,
        dataloader: DataLoader,
        other_anchors: torch.Tensor,
        max_grad_norm: float = 1.0,
        round_num: int = 0
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Runs one full epoch of local optimization over mini-batches.
        Returns accumulated loss breakdown and per-step diagnostic records.
        """
        self.encoder.train()
        self.anchor_set.train()
        
        accumulated_metrics: Dict[str, float] = {}
        step_diagnostics: List[Dict[str, float]] = []
        
        num_batches = len(dataloader)
        
        for batch_idx, (batch_edges,) in enumerate(dataloader):
            optimizer.zero_grad()
            
            anchors_before = self.anchor_set().detach().clone()
            
            unique_nodes, inverse_indices = torch.unique(batch_edges, return_inverse=True)
            batch_x = self.X_tensor[unique_nodes]
            local_edges = inverse_indices.view(-1, 2)
            
            embeddings = self.encoder(batch_x)
            local_anchors = self.anchor_set()
            
            loss, breakdown = loss_fn(
                embeddings=embeddings,
                edges=local_edges,
                local_anchors=local_anchors,
                other_anchors=other_anchors,
                client_id=self.client_id,
                round_num=round_num
            )
            
            loss.backward()
            
            # Diagnostic tracking before clipping
            if self.anchor_set.anchors.grad is not None:
                pre_clip_grad_norm = torch.norm(self.anchor_set.anchors.grad).item()
                max_abs_grad = torch.max(torch.abs(self.anchor_set.anchors.grad)).item()
            else:
                pre_clip_grad_norm = 0.0
                max_abs_grad = 0.0
                
            enc_clip_coef = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=max_grad_norm)
            anc_clip_coef = torch.nn.utils.clip_grad_norm_(self.anchor_set.parameters(), max_norm=max_grad_norm)
            
            if self.anchor_set.anchors.grad is not None:
                post_clip_grad_norm = torch.norm(self.anchor_set.anchors.grad).item()
            else:
                post_clip_grad_norm = 0.0
                
            optimizer.step()
            
            anchors_after = self.anchor_set().detach().clone()
            displacement = torch.norm(anchors_after - anchors_before).item()
            
            step_diagnostics.append({
                "batch_idx": batch_idx,
                "pre_clip_grad_norm": pre_clip_grad_norm,
                "post_clip_grad_norm": post_clip_grad_norm,
                "anc_clip_coef": anc_clip_coef.item() if isinstance(anc_clip_coef, torch.Tensor) else float(anc_clip_coef),
                "max_abs_grad": max_abs_grad,
                "displacement": displacement,
                "nonzero_grad": 1.0 if pre_clip_grad_norm > 0 else 0.0,
                "nonzero_update": 1.0 if displacement > 0 else 0.0,
                "l_sep": breakdown.get("separation_loss", 0.0),
                "l_spread": breakdown.get("anchor_spread_loss", 0.0),
                "l_balance": breakdown.get("balance_loss", 0.0)
            })
            
            for k, v in breakdown.items():
                accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v / num_batches
                
        return accumulated_metrics, step_diagnostics

    def local_train(
        self,
        global_encoder_state: Dict[str, torch.Tensor],
        other_anchors_tensor: torch.Tensor,
        config: Dict[str, Any],
        round_num: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, float], List[Dict[str, float]]]:
        """
        Perform local training for local_epochs rounds using mini-batches.
        """
        self.encoder.load_state_dict(global_encoder_state)
        
        if not self.anchor_set.initialized:
            init_strat = config.get("anchors", {}).get("initialization", "kmeans")
            self.initialize_anchors(strategy=init_strat)
            
        other_anchors_device = other_anchors_tensor.to(self.device)
        
        lr = config.get("training", {}).get("learning_rate", 0.001)
        max_grad_norm = config.get("training", {}).get("max_grad_norm", 1.0)
        local_epochs = config.get("training", {}).get("local_epochs", 1)
        batch_size = config.get("training", {}).get("batch_size", 256)
        
        persistent_opt = config.get("training", {}).get("persistent_optimizer", False)
        
        if persistent_opt:
            if self.optimizer is None:
                self.optimizer = optim.Adam(
                    list(self.encoder.parameters()) + list(self.anchor_set.parameters()),
                    lr=lr
                )
            optimizer = self.optimizer
        else:
            optimizer = optim.Adam(
                list(self.encoder.parameters()) + list(self.anchor_set.parameters()),
                lr=lr
            )
            
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        loss_cfg = config.get("loss", {})
        rep_cfg = loss_cfg.get("anchor_repulsion", {})
        bal_cfg = loss_cfg.get("balancing", {})
        model_cfg = config.get("model", {})
        emb_reg_cfg = model_cfg.get("embedding_regularization", {})
        
        loss_fn = TotalLossV4(
            lambda_attraction=loss_cfg.get("lambda_attraction", 1.0),
            lambda_anchor=loss_cfg.get("lambda_anchor", 1.0),
            lambda_anchor_repulsion=loss_cfg.get("lambda_anchor_repulsion", 1.0),
            lambda_spread=loss_cfg.get("lambda_spread", 1.0),
            lambda_balance=loss_cfg.get("lambda_balance", 0.0),
            lambda_sep=loss_cfg.get("lambda_sep", 0.0),
            margin=loss_cfg.get("margin", 0.1),
            lambda_embedding_reg=emb_reg_cfg.get("coefficient", 0.0) if emb_reg_cfg.get("enabled", False) else 0.0,
            alpha_soft=loss_cfg.get("alpha_soft", 1.0),
            normalization=rep_cfg.get("normalization", "none"),
            scale=rep_cfg.get("scale", 1.0),
            eps=loss_cfg.get("eps", 1e-7),
            balancing_enabled=bal_cfg.get("enabled", False),
            use_attraction=config.get("ablation", {}).get("use_attraction", True),
            use_anchor=config.get("ablation", {}).get("use_anchor", True),
            use_anchor_repulsion=config.get("ablation", {}).get("use_anchor_repulsion", True),
            use_spread=config.get("ablation", {}).get("use_spread", True),
            anchor_assignment_type=config.get("ablation", {}).get("anchor_assignment_type", "soft")
        )
        
        accumulated_metrics: Dict[str, float] = {}
        all_step_diagnostics: List[Dict[str, float]] = []
        
        for epoch in range(local_epochs):
            metrics, step_diags = self.train_epoch(
                loss_fn=loss_fn,
                optimizer=optimizer,
                dataloader=dataloader,
                other_anchors=other_anchors_device,
                max_grad_norm=max_grad_norm,
                round_num=round_num
            )
            for k, v in metrics.items():
                accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v / local_epochs
                
            all_step_diagnostics.extend(step_diags)
                
        updated_encoder_state = {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()}
        updated_anchors = self.anchor_set().detach().cpu().clone()
        
        return updated_encoder_state, updated_anchors, accumulated_metrics, all_step_diagnostics
