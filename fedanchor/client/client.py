import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, Any, Optional
from fedanchor.models.encoder import Encoder
from fedanchor.models.anchor import AnchorSet
from fedanchor.graph.knn import construct_knn_graph
from fedanchor.losses.total_loss import TotalLoss

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
        round_num: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, float]]:
        """
        Perform local training for local_epochs rounds.
        """
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
