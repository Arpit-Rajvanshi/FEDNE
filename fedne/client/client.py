import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Optional

from fedne.graph.knn import construct_knn_graph
from fedne.augmentation.mixup import mixup_dataset
from fedne.models.surrogate import SurrogateRepulsion, generate_grid_query_points, compute_repulsion_targets
from fedne.losses.contrastive_loss import ContrastiveNeighborEmbeddingLoss

class EdgeDataset(Dataset):
    """
    Dataset of graph edges for neighbor embedding.
    """
    def __init__(self, edges: np.ndarray) -> None:
        self.edges = edges
        
    def __len__(self) -> int:
        return len(self.edges)
        
    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return int(self.edges[idx, 0]), int(self.edges[idx, 1])

class Client:
    """
    Federated client holding local data, local encoder copy, and local surrogate repulsion model.
    """
    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        client_sizes: List[int],
        config: Dict[str, Any],
        device: torch.device
    ) -> None:
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.client_sizes = client_sizes
        self.config = config
        self.device = device
        
        # Local model copies
        input_dim = X_train.shape[1]
        self.encoder = None  # Reference to local encoder, set by server broadcast
        self.surrogate = SurrogateRepulsion(hidden_dim=config["surrogate"]["hidden_dim"]).to(device)
        self.other_surrogates: List[nn.Module] = []
        
        # Instantiate combined loss
        self.loss_fn = ContrastiveNeighborEmbeddingLoss(
            client_id=client_id,
            client_sizes=client_sizes,
            attraction_coeff=config.get("coefficients", {}).get("attraction", 1.0),
            local_rep_coeff=config.get("coefficients", {}).get("local_repulsion", 1.0),
            surr_rep_coeff=config.get("coefficients", {}).get("surrogate_repulsion", 1.0)
        )
        
    def set_encoder(self, encoder: nn.Module) -> None:
        """
        Set local encoder copy.
        """
        self.encoder = encoder
        
    def set_other_surrogates(self, other_surrogates: List[nn.Module]) -> None:
        """
        Set reference to surrogates of other clients.
        """
        self.other_surrogates = other_surrogates
        
    def train_encoder(self, round_idx: int) -> Dict[str, float]:
        """
        Train local copy of the encoder network.
        """
        start_time = time.time()
        self.encoder.train()
        
        # Step 1: Augmentation (Mixup)
        aug_config = self.config.get("augmentation", {})
        if aug_config.get("enabled", True):
            X_aug, y_aug = mixup_dataset(
                self.X_train,
                self.y_train,
                k=self.config["graph"]["k"],
                alpha=aug_config.get("alpha", 0.2),
                mixup_ratio=aug_config.get("mixup_ratio", 1.0)
            )
        else:
            X_aug, y_aug = self.X_train, self.y_train
            
        # Step 2: Build local kNN graph
        k = self.config["graph"]["k"]
        edges = construct_knn_graph(X_aug, k)
        
        # Convert X_aug to torch tensor for fast loading
        X_aug_tensor = torch.from_numpy(X_aug).to(self.device)
        
        # Step 3: PyTorch DataLoader
        dataset = EdgeDataset(edges)
        batch_size = self.config.get("batch_size", 512)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer for encoder
        opt_config = self.config["optimizer"]
        optimizer = optim.Adam(
            self.encoder.parameters(),
            lr=opt_config["lr"],
            weight_decay=opt_config.get("weight_decay", 0.0)
        )
        
        # Adjust learning rate based on decay schedule
        decay_rounds = opt_config.get("lr_decay_rounds", [])
        decay_factor = opt_config.get("lr_decay_factor", 0.1)
        current_lr = opt_config["lr"]
        for r_decay in decay_rounds:
            if round_idx >= r_decay:
                current_lr *= decay_factor
        for g in optimizer.param_groups:
            g['lr'] = current_lr
            
        # Run training loop
        epoch_losses = []
        epoch_att = []
        epoch_rep_local = []
        epoch_rep_surr = []
        
        # Statistics collectors
        att_min_list, att_max_list, att_mean_list = [], [], []
        rep_min_list, rep_max_list, rep_mean_list = [], [], []
        surr_min_list, surr_max_list, surr_mean_list = [], [], []
        phi_min_list, phi_max_list = [], []
        dist_min_list, dist_max_list = [], []
        grad_norms = []
        emb_norms = []
        
        b = self.config["graph"]["negative_samples"]
        with_replacement = self.config["graph"].get("with_replacement", True)
        
        # Check if surrogate models should be included
        start_round = self.config["surrogate"].get("start_round", 0)
        enable_surrogate = round_idx >= start_round
        
        local_epochs = self.config["federated"].get("local_epochs", 1)
        
        # Enable anomaly detection if configured
        debug_anomaly = self.config.get("logging", {}).get("debug_anomaly", False)
        if debug_anomaly:
            torch.autograd.set_detect_anomaly(True)
            
        for epoch in range(local_epochs):
            for src_idx, dst_idx in dataloader:
                optimizer.zero_grad()
                
                # Fetch source and neighbor images
                x_i = X_aug_tensor[src_idx]
                x_j = X_aug_tensor[dst_idx]
                
                # Sample negative indices
                num_src = len(src_idx)
                if with_replacement:
                    neg_idx = np.random.choice(len(X_aug), size=(num_src, b), replace=True)
                else:
                    neg_idx = np.zeros((num_src, b), dtype=np.int64)
                    for i in range(num_src):
                        neg_idx[i] = np.random.choice(len(X_aug), size=b, replace=False)
                
                x_neg = X_aug_tensor[neg_idx] # [B, b, D]
                
                # Forward pass
                z_i = self.encoder(x_i)
                z_j = self.encoder(x_j)
                
                # Reshape x_neg to [B * b, D] for encoder forward pass, then reshape back
                z_neg_flat = self.encoder(x_neg.view(-1, x_neg.shape[-1]))
                z_neg = z_neg_flat.view(num_src, b, -1)
                
                # Compute loss
                loss_dict = self.loss_fn(
                    z_i=z_i,
                    z_j=z_j,
                    z_neg=z_neg,
                    surrogates=self.other_surrogates,
                    enable_surrogate=enable_surrogate
                )
                
                loss = loss_dict["loss"]
                
                # Check for NaNs or Infs
                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(
                        f"[Client {self.client_id} Round {round_idx}] Exploding loss encountered! "
                        f"Loss: {loss.item()}, Att: {loss_dict['attraction'].item()}, "
                        f"Local Rep: {loss_dict['repulsion_local'].item()}, "
                        f"Surr Rep: {loss_dict['repulsion_surrogate'].item()}"
                    )
                    
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                
                # Compute gradient norm
                total_grad_norm = 0.0
                for p in self.encoder.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.detach().data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                grad_norms.append(total_grad_norm)
                
                optimizer.step()
                
                # Log metrics for this step
                epoch_losses.append(loss.item())
                epoch_att.append(loss_dict["attraction"].item())
                epoch_rep_local.append(loss_dict["repulsion_local"].item())
                epoch_rep_surr.append(loss_dict["repulsion_surrogate"].item())
                
                # Compute distances and phi metrics
                with torch.no_grad():
                    d2_pos = torch.sum((z_i - z_j) ** 2, dim=-1)
                    d2_neg = torch.sum((z_i.unsqueeze(1) - z_neg) ** 2, dim=-1)
                    
                    dist_pos = torch.sqrt(d2_pos + 1e-12)
                    dist_neg = torch.sqrt(d2_neg + 1e-12)
                    
                    phi_pos = 1.0 / (1.0 + d2_pos)
                    phi_neg = 1.0 / (1.0 + d2_neg)
                    
                    # Embedding norm
                    emb_norm = torch.mean(torch.norm(z_i, p=2, dim=-1)).item()
                    emb_norms.append(emb_norm)
                    
                    dist_min_list.append(min(dist_pos.min().item(), dist_neg.min().item()))
                    dist_max_list.append(max(dist_pos.max().item(), dist_neg.max().item()))
                    phi_min_list.append(min(phi_pos.min().item(), phi_neg.min().item()))
                    phi_max_list.append(max(phi_pos.max().item(), phi_neg.max().item()))
                    
                    # Attraction loss details (per sample)
                    loss_att_samples = -torch.log(torch.clamp(phi_pos, min=1e-7))
                    att_min_list.append(loss_att_samples.min().item())
                    att_max_list.append(loss_att_samples.max().item())
                    att_mean_list.append(loss_att_samples.mean().item())
                    
                    # Local Repulsion loss details (per sample)
                    loss_rep_samples = -torch.sum(torch.log(torch.clamp(1.0 - phi_neg, min=1e-7)), dim=-1)
                    rep_min_list.append(loss_rep_samples.min().item())
                    rep_max_list.append(loss_rep_samples.max().item())
                    rep_mean_list.append(loss_rep_samples.mean().item())
                    
                    # Surrogate Repulsion details
                    if enable_surrogate and len(self.other_surrogates) > 1:
                        surr_preds = []
                        for m_prime, surrogate in enumerate(self.other_surrogates):
                            if m_prime == self.client_id:
                                continue
                            scale = self.loss_fn.surr_scales[m_prime]
                            # Clamp prediction to be non-negative in logs to match loss
                            surr_pred_clamped = torch.clamp(surrogate(z_i).squeeze(1), min=0.0)
                            surr_preds.append(scale * surr_pred_clamped)
                        surr_sum = torch.stack(surr_preds).sum(dim=0)  # [B]
                        surr_min_list.append(surr_sum.min().item())
                        surr_max_list.append(surr_sum.max().item())
                        surr_mean_list.append(surr_sum.mean().item())
                    else:
                        surr_min_list.append(0.0)
                        surr_max_list.append(0.0)
                        surr_mean_list.append(0.0)
                        
        training_time = time.time() - start_time
        
        # Disable anomaly detection
        if debug_anomaly:
            torch.autograd.set_detect_anomaly(False)
            
        return {
            "loss": np.mean(epoch_losses) if epoch_losses else 0.0,
            "attraction": np.mean(epoch_att) if epoch_att else 0.0,
            "repulsion_local": np.mean(epoch_rep_local) if epoch_rep_local else 0.0,
            "repulsion_surrogate": np.mean(epoch_rep_surr) if epoch_rep_surr else 0.0,
            "att_min": np.min(att_min_list) if att_min_list else 0.0,
            "att_max": np.max(att_max_list) if att_max_list else 0.0,
            "att_mean": np.mean(att_mean_list) if att_mean_list else 0.0,
            "rep_min": np.min(rep_min_list) if rep_min_list else 0.0,
            "rep_max": np.max(rep_max_list) if rep_max_list else 0.0,
            "rep_mean": np.mean(rep_mean_list) if rep_mean_list else 0.0,
            "surr_min": np.min(surr_min_list) if surr_min_list else 0.0,
            "surr_max": np.max(surr_max_list) if surr_max_list else 0.0,
            "surr_mean": np.mean(surr_mean_list) if surr_mean_list else 0.0,
            "phi_min": np.min(phi_min_list) if phi_min_list else 0.0,
            "phi_max": np.max(phi_max_list) if phi_max_list else 0.0,
            "distance_min": np.min(dist_min_list) if dist_min_list else 0.0,
            "distance_max": np.max(dist_max_list) if dist_max_list else 0.0,
            "grad_norm": np.mean(grad_norms) if grad_norms else 0.0,
            "emb_norm": np.mean(emb_norms) if emb_norms else 0.0,
            "lr": current_lr,
            "time": training_time
        }
        
    def train_surrogate(self) -> Dict[str, float]:
        """
        Train local surrogate repulsion model using current local embeddings.
        """
        start_time = time.time()
        self.encoder.eval()
        self.surrogate.train()
        
        # Step 1: Compute local embeddings
        with torch.no_grad():
            X_tensor = torch.from_numpy(self.X_train).to(self.device)
            # Embed in batches to save memory
            embeddings_list = []
            batch_size = 1024
            for i in range(0, len(X_tensor), batch_size):
                batch_x = X_tensor[i : i + batch_size]
                emb = self.encoder(batch_x)
                embeddings_list.append(emb.cpu().numpy())
            embeddings = np.concatenate(embeddings_list, axis=0)
            
        # Step 2: Generate grid query points
        surr_config = self.config["surrogate"]
        query_points = generate_grid_query_points(
            embeddings=embeddings,
            grid_step=surr_config.get("grid_step", 0.3),
            margin=surr_config.get("margin", 0.5),
            samples_per_axis=surr_config.get("samples_per_axis", "auto")
        )
        
        # Step 3: Compute true repulsion targets
        b = self.config["graph"]["negative_samples"]
        with_replacement = self.config["graph"].get("with_replacement", True)
        targets = compute_repulsion_targets(
            query_points=query_points,
            embeddings=embeddings,
            b=b,
            with_replacement=with_replacement
        )
        
        # Step 4: Create dataloader for surrogate dataset
        class SurrogateDataset(Dataset):
            def __init__(self, queries: np.ndarray, targs: np.ndarray) -> None:
                self.queries = torch.from_numpy(queries)
                self.targs = torch.from_numpy(targs).unsqueeze(1)
            def __len__(self) -> int:
                return len(self.queries)
            def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
                return self.queries[idx], self.targs[idx]
                
        surr_dataset = SurrogateDataset(query_points, targets)
        # Train on small batches for MSE optimization
        surr_dataloader = DataLoader(surr_dataset, batch_size=256, shuffle=True)
        
        # Optimizer for surrogate
        optimizer = optim.Adam(self.surrogate.parameters(), lr=surr_config["lr"])
        criterion = nn.MSELoss()
        
        # Fine-tune the surrogate model for a few epochs
        epochs = surr_config.get("epochs", 5)
        losses = []
        
        # Stats collections
        pred_mins, pred_maxs, pred_means = [], [], []
        grad_norms = []
        
        for epoch in range(epochs):
            for batch_q, batch_t in surr_dataloader:
                batch_q = batch_q.to(self.device)
                batch_t = batch_t.to(self.device)
                
                optimizer.zero_grad()
                pred = self.surrogate(batch_q)
                loss = criterion(pred, batch_t)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"[Client {self.client_id}] Surrogate MSE exploded! Loss={loss.item()}")
                    
                loss.backward()
                
                # Gradient clipping for surrogate
                torch.nn.utils.clip_grad_norm_(self.surrogate.parameters(), max_norm=1.0)
                
                # Compute gradient norm
                total_grad_norm = 0.0
                for p in self.surrogate.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.detach().data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                grad_norms.append(total_grad_norm)
                
                optimizer.step()
                
                losses.append(loss.item())
                
                with torch.no_grad():
                    pred_mins.append(pred.min().item())
                    pred_maxs.append(pred.max().item())
                    pred_means.append(pred.mean().item())
                    
        surrogate_time = time.time() - start_time
        
        return {
            "surrogate_loss": np.mean(losses) if losses else 0.0,
            "target_min": float(np.min(targets)) if len(targets) > 0 else 0.0,
            "target_max": float(np.max(targets)) if len(targets) > 0 else 0.0,
            "target_mean": float(np.mean(targets)) if len(targets) > 0 else 0.0,
            "pred_min": float(np.min(pred_mins)) if pred_mins else 0.0,
            "pred_max": float(np.max(pred_maxs)) if pred_maxs else 0.0,
            "pred_mean": float(np.mean(pred_means)) if pred_means else 0.0,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "time": surrogate_time
        }
