import os
import time
import json
import yaml
import copy
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Any, List, Tuple

from fedne.datasets.loader import load_dataset
from fedne.client.client import Client
from fedne.server.server import Server
from fedne.evaluation.metrics import (
    compute_trustworthiness_and_continuity,
    compute_knn_accuracy,
    compute_steadiness_cohesiveness
)
from fedne.evaluation.visualization import plot_embedding_by_class, plot_embedding_by_client

class Trainer:
    """
    Orchestrator for the Federated Neighbor Embedding training and evaluation loop.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = torch.device(config["federated"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
        
        # Output paths
        logging_config = config.get("logging", {})
        self.log_dir = logging_config.get("log_dir", "./runs")
        self.checkpoint_dir = logging_config.get("checkpoint_dir", "./checkpoints")
        self.output_dir = logging_config.get("output_dir", "./outputs")
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Tensorboard Writer
        self.tb_writer = None
        if logging_config.get("tensorboard", True):
            self.tb_writer = SummaryWriter(log_dir=self.log_dir)
            
        # Metric history
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Load and partition dataset
        print(f"Loading and partitioning dataset: {config['dataset']['name']}...")
        part_config = config.get("partition", {"type": "dirichlet", "alpha": 0.2})
        client_data, (self.X_test, self.y_test) = load_dataset(
            name=config["dataset"]["name"],
            data_dir=config["dataset"].get("data_dir", "./data"),
            partition_type=part_config["type"],
            num_clients=config["federated"]["clients"],
            alpha=part_config.get("alpha", 0.2),
            shards_per_client=part_config.get("shards_per_client", 2)
        )
        
        self.client_sizes = [len(x) for x, _ in client_data]
        self.input_dim = self.X_test.shape[1]
        
        # Initialize clients
        print(f"Initializing {len(client_data)} clients...")
        self.clients = []
        for i, (x_train, y_train) in enumerate(client_data):
            client = Client(
                client_id=i,
                X_train=x_train,
                y_train=y_train,
                client_sizes=self.client_sizes,
                config=config,
                device=self.device
            )
            self.clients.append(client)
            
        # Initialize server
        self.server = Server(
            config=config,
            input_dim=self.input_dim,
            client_sizes=self.client_sizes,
            device=self.device
        )
        
        self.best_knn_acc = 0.0
        self.start_round = 0
        
    def save_checkpoint(self, path: str, round_idx: int) -> None:
        """
        Save global encoder, optimizer state, surrogates, configs, and metrics history.
        """
        os.makedirs(path, exist_ok=True)
        
        # 1. Save global encoder
        torch.save(self.server.global_encoder.state_dict(), os.path.join(path, "global_encoder.pt"))
        
        # 2. Save client surrogates
        surr_dir = os.path.join(path, "client_surrogates")
        os.makedirs(surr_dir, exist_ok=True)
        for idx, surrogate in enumerate(self.server.surrogates):
            torch.save(surrogate.state_dict(), os.path.join(surr_dir, f"client{idx}.pt"))
            
        # 3. Save config
        with open(os.path.join(path, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)
            
        # 4. Save metrics history
        with open(os.path.join(path, "metrics.json"), "w") as f:
            json.dump(self.metrics_history, f, indent=2)
            
        # 5. Save metadata containing round index and best performance
        metadata = {
            "round": round_idx,
            "best_knn_acc": self.best_knn_acc
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
    def load_checkpoint(self, path: str) -> None:
        """
        Load weights, states, and history from checkpoint to resume training.
        """
        print(f"Resuming training from checkpoint: {path}...")
        
        # 1. Load global encoder
        self.server.global_encoder.load_state_dict(
            torch.load(os.path.join(path, "global_encoder.pt"), map_location=self.device)
        )
        
        # 2. Load client surrogates
        surr_dir = os.path.join(path, "client_surrogates")
        for idx in range(len(self.clients)):
            self.server.surrogates[idx].load_state_dict(
                torch.load(os.path.join(surr_dir, f"client{idx}.pt"), map_location=self.device)
            )
            self.clients[idx].surrogate.load_state_dict(
                torch.load(os.path.join(surr_dir, f"client{idx}.pt"), map_location=self.device)
            )
            
        # 3. Load metrics history
        with open(os.path.join(path, "metrics.json"), "r") as f:
            self.metrics_history = json.load(f)
            
        # 4. Load metadata
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
            self.start_round = metadata["round"] + 1
            self.best_knn_acc = metadata.get("best_knn_acc", 0.0)
            
        print(f"Resuming from round {self.start_round}. Best kNN Acc: {self.best_knn_acc:.4f}")

    def evaluate_global_embedding(self, round_idx: int) -> Dict[str, float]:
        """
        Evaluate global test set embedding.
        """
        self.server.global_encoder.eval()
        
        # Get test set embeddings
        with torch.no_grad():
            X_tensor = torch.from_numpy(self.X_test).to(self.device)
            embeddings_list = []
            batch_size = 1024
            for i in range(0, len(X_tensor), batch_size):
                batch_x = X_tensor[i : i + batch_size]
                emb = self.server.global_encoder(batch_x)
                embeddings_list.append(emb.cpu().numpy())
            embeddings = np.concatenate(embeddings_list, axis=0)
            
        # Since full metrics calculation can be extremely slow, we evaluate on a random subset
        eval_subset_size = self.config.get("logging", {}).get("eval_subset_size", 2000)
        n_test = len(embeddings)
        if n_test > eval_subset_size:
            subset_indices = np.random.choice(n_test, size=eval_subset_size, replace=False)
            X_test_eval = self.X_test[subset_indices]
            embeddings_eval = embeddings[subset_indices]
            y_test_eval = self.y_test[subset_indices]
        else:
            X_test_eval = self.X_test
            embeddings_eval = embeddings
            y_test_eval = self.y_test
            
        # Compute Trustworthiness & Continuity
        k_eval = self.config["graph"]["k"]
        trust, cont = compute_trustworthiness_and_continuity(X_test_eval, embeddings_eval, k=k_eval)
        
        # Compute Steadiness & Cohesiveness
        stead, cohes, using_snc = compute_steadiness_cohesiveness(
            X_high=X_test_eval,
            X_low=embeddings_eval,
            num_classes=len(np.unique(self.y_test)),
            k=k_eval
        )
        
        # For kNN classification accuracy, split test embeddings into a train/test split to compute accuracy
        # (Since we want to measure embedding's class separation)
        n_eval = len(embeddings_eval)
        split = int(0.8 * n_eval)
        train_emb, test_emb = embeddings_eval[:split], embeddings_eval[split:]
        train_labels, test_labels = y_test_eval[:split], y_test_eval[split:]
        knn_acc = compute_knn_accuracy(train_emb, train_labels, test_emb, test_labels, k=k_eval)
        
        # Save plots if visualization is scheduled
        visualize_freq = self.config.get("logging", {}).get("visualize_frequency", 5)
        is_final_round = round_idx == self.config["federated"]["rounds"] - 1
        
        if round_idx % visualize_freq == 0 or is_final_round:
            emb_dir = os.path.join(self.output_dir, "embeddings")
            
            # Map client ids to test samples for visualization by client
            # (To simulate client distributions in test space, we can assign clients based on closest local clusters)
            # Or simply assign clients to test samples using a 1NN classifier trained on client training samples
            client_id_labels = np.zeros_like(y_test_eval)
            X_train_list = []
            y_train_list = []
            for client in self.clients:
                X_train_list.append(client.X_train)
                y_train_list.append(np.full(len(client.X_train), client.client_id))
            X_train_all = np.concatenate(X_train_list, axis=0)
            y_train_all = np.concatenate(y_train_list, axis=0)
            
            from sklearn.neighbors import KNeighborsClassifier
            clf_client = KNeighborsClassifier(n_neighbors=1)
            clf_client.fit(X_train_all, y_train_all)
            client_id_labels = clf_client.predict(X_test_eval)
            
            class_plot_path = os.path.join(emb_dir, f"round_{round_idx:03d}_by_class.png")
            client_plot_path = os.path.join(emb_dir, f"round_{round_idx:03d}_by_client.png")
            
            plot_embedding_by_class(
                embeddings_eval,
                y_test_eval,
                class_plot_path,
                title=f"MNIST Embeddings by Class - Round {round_idx}"
            )
            plot_embedding_by_client(
                embeddings_eval,
                client_id_labels,
                client_plot_path,
                title=f"MNIST Embeddings by Client - Round {round_idx}"
            )
            
        return {
            "trustworthiness": trust,
            "continuity": cont,
            "knn_accuracy": knn_acc,
            "steadiness": stead,
            "cohesiveness": cohes,
            "using_snc": float(using_snc)
        }

    def fit(self) -> None:
        """
        Run the complete Federated Neighbor Embedding training process.
        """
        total_rounds = self.config["federated"]["rounds"]
        print(f"Starting FEDNE training loop: {total_rounds} communication rounds...")
        
        for r in range(self.start_round, total_rounds):
            round_start_time = time.time()
            
            # --- Server Broadcast ---
            comm_start = time.time()
            self.server.broadcast_encoder(self.clients)
            self.server.broadcast_surrogates(self.clients)
            comm_time = time.time() - comm_start
            
            # --- Client Local Training (Encoder and Surrogate) ---
            client_train_losses = []
            client_att_losses = []
            client_rep_local_losses = []
            client_rep_surr_losses = []
            client_surr_mses = []
            
            # Detailed stats lists
            client_att_min, client_att_max, client_att_mean = [], [], []
            client_rep_min, client_rep_max, client_rep_mean = [], [], []
            client_surr_min, client_surr_max, client_surr_mean = [], [], []
            client_phi_min, client_phi_max = [], []
            client_dist_min, client_dist_max = [], []
            client_grad_norms = []
            client_emb_norms = []
            
            client_target_min, client_target_max, client_target_mean = [], [], []
            client_pred_min, client_pred_max, client_pred_mean = [], [], []
            client_surr_grad_norms = []
            
            client_train_times = []
            client_surr_times = []
            
            for client in self.clients:
                # 1. Train local copy of the encoder
                enc_metrics = client.train_encoder(round_idx=r)
                client_train_losses.append(enc_metrics["loss"])
                client_att_losses.append(enc_metrics["attraction"])
                client_rep_local_losses.append(enc_metrics["repulsion_local"])
                client_rep_surr_losses.append(enc_metrics["repulsion_surrogate"])
                
                client_att_min.append(enc_metrics["att_min"])
                client_att_max.append(enc_metrics["att_max"])
                client_att_mean.append(enc_metrics["att_mean"])
                client_rep_min.append(enc_metrics["rep_min"])
                client_rep_max.append(enc_metrics["rep_max"])
                client_rep_mean.append(enc_metrics["rep_mean"])
                client_surr_min.append(enc_metrics["surr_min"])
                client_surr_max.append(enc_metrics["surr_max"])
                client_surr_mean.append(enc_metrics["surr_mean"])
                
                client_phi_min.append(enc_metrics["phi_min"])
                client_phi_max.append(enc_metrics["phi_max"])
                client_dist_min.append(enc_metrics["distance_min"])
                client_dist_max.append(enc_metrics["distance_max"])
                client_grad_norms.append(enc_metrics["grad_norm"])
                client_emb_norms.append(enc_metrics["emb_norm"])
                client_train_times.append(enc_metrics["time"])
                
                # 2. Train local surrogate model
                surr_metrics = client.train_surrogate()
                client_surr_mses.append(surr_metrics["surrogate_loss"])
                client_target_min.append(surr_metrics["target_min"])
                client_target_max.append(surr_metrics["target_max"])
                client_target_mean.append(surr_metrics["target_mean"])
                client_pred_min.append(surr_metrics["pred_min"])
                client_pred_max.append(surr_metrics["pred_max"])
                client_pred_mean.append(surr_metrics["pred_mean"])
                client_surr_grad_norms.append(surr_metrics["grad_norm"])
                client_surr_times.append(surr_metrics["time"])
                
            # --- Server Weight Aggregation ---
            comm_start = time.time()
            self.server.aggregate_encoder(self.clients)
            self.server.collect_surrogates(self.clients)
            comm_time += (time.time() - comm_start)
            
            # Record execution times
            local_train_time = sum(client_train_times) / len(self.clients)
            surr_train_time = sum(client_surr_times) / len(self.clients)
            round_elapsed = time.time() - round_start_time
            
            # GPU Memory usage
            gpu_mem = 0.0
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)  # MB
                torch.cuda.reset_peak_memory_stats(self.device)
                
            # Average losses across clients
            avg_loss = np.mean(client_train_losses)
            avg_att = np.mean(client_att_losses)
            avg_rep_local = np.mean(client_rep_local_losses)
            avg_rep_surr = np.mean(client_rep_surr_losses)
            avg_surr_mse = np.mean(client_surr_mses)
            
            # Aggregate detailed metrics
            att_min_val = np.min(client_att_min)
            att_max_val = np.max(client_att_max)
            att_mean_val = np.mean(client_att_mean)
            
            rep_min_val = np.min(client_rep_min)
            rep_max_val = np.max(client_rep_max)
            rep_mean_val = np.mean(client_rep_mean)
            
            surr_min_val = np.min(client_surr_min)
            surr_max_val = np.max(client_surr_max)
            surr_mean_val = np.mean(client_surr_mean)
            
            phi_min_val = np.min(client_phi_min)
            phi_max_val = np.max(client_phi_max)
            dist_min_val = np.min(client_dist_min)
            dist_max_val = np.max(client_dist_max)
            avg_grad_norm = np.mean(client_grad_norms)
            avg_emb_norm = np.mean(client_emb_norms)
            
            target_min_val = np.min(client_target_min)
            target_max_val = np.max(client_target_max)
            target_mean_val = np.mean(client_target_mean)
            
            pred_min_val = np.min(client_pred_min)
            pred_max_val = np.max(client_pred_max)
            pred_mean_val = np.mean(client_pred_mean)
            avg_surr_grad_norm = np.mean(client_surr_grad_norms)
            
            # --- Global Model Evaluation ---
            eval_metrics = self.evaluate_global_embedding(round_idx=r)
            
            # Combine all metrics
            round_metrics = {
                "round": r,
                "loss": float(avg_loss),
                "attraction_loss": float(avg_att),
                "local_repulsion": float(avg_rep_local),
                "surrogate_repulsion": float(avg_rep_surr),
                "surrogate_loss": float(avg_surr_mse),
                "trustworthiness": eval_metrics["trustworthiness"],
                "continuity": eval_metrics["continuity"],
                "knn_accuracy": eval_metrics["knn_accuracy"],
                "steadiness": eval_metrics["steadiness"],
                "cohesiveness": eval_metrics["cohesiveness"],
                "communication_time": float(comm_time),
                "local_training_time": float(local_train_time),
                "surrogate_training_time": float(surr_train_time),
                "gpu_memory_mb": float(gpu_mem),
                "round_time": float(round_elapsed),
                "att_min": float(att_min_val),
                "att_max": float(att_max_val),
                "att_mean": float(att_mean_val),
                "rep_min": float(rep_min_val),
                "rep_max": float(rep_max_val),
                "rep_mean": float(rep_mean_val),
                "surr_min": float(surr_min_val),
                "surr_max": float(surr_max_val),
                "surr_mean": float(surr_mean_val),
                "phi_min": float(phi_min_val),
                "phi_max": float(phi_max_val),
                "distance_min": float(dist_min_val),
                "distance_max": float(dist_max_val),
                "grad_norm": float(avg_grad_norm),
                "emb_norm": float(avg_emb_norm),
                "target_min": float(target_min_val),
                "target_max": float(target_max_val),
                "target_mean": float(target_mean_val),
                "pred_min": float(pred_min_val),
                "pred_max": float(pred_max_val),
                "pred_mean": float(pred_mean_val),
                "surr_grad_norm": float(avg_surr_grad_norm)
            }
            
            self.metrics_history.append(round_metrics)
            
            # Print round summary
            print(
                f"[Round {r:03d}/{total_rounds:03d}] "
                f"Loss: {avg_loss:.4f} | Att: {avg_att:.4f} | "
                f"Local Rep: {avg_rep_local:.4f} | Surr Rep: {avg_rep_surr:.4f} | "
                f"Surr MSE: {avg_surr_mse:.4e} | Trust: {eval_metrics['trustworthiness']:.4f} | "
                f"kNN Acc: {eval_metrics['knn_accuracy']:.4f}"
            )
            print(
                f"  [Surrogate validation] Target range: [{target_min_val:.2f}, {target_max_val:.2f}] (mean {target_mean_val:.2f}) | "
                f"Pred range: [{pred_min_val:.2f}, {pred_max_val:.2f}] (mean {pred_mean_val:.2f}) | "
                f"MSE: {avg_surr_mse:.2e} | Grad norm: {avg_surr_grad_norm:.2f}"
            )
            print(
                f"  [Encoder validation] Phi range: [{phi_min_val:.4f}, {phi_max_val:.4f}] | "
                f"Dist range: [{dist_min_val:.4f}, {dist_max_val:.4f}] | "
                f"Grad norm: {avg_grad_norm:.2f} | Emb norm: {avg_emb_norm:.2f}"
            )
            
            # --- Logging & Summary Writing ---
            if self.tb_writer:
                self.tb_writer.add_scalar("Loss/Total", avg_loss, r)
                self.tb_writer.add_scalar("Loss/Attraction", avg_att, r)
                self.tb_writer.add_scalar("Loss/Local_Repulsion", avg_rep_local, r)
                self.tb_writer.add_scalar("Loss/Surrogate_Repulsion", avg_rep_surr, r)
                self.tb_writer.add_scalar("Loss/Surrogate_MSE", avg_surr_mse, r)
                
                self.tb_writer.add_scalar("Metrics/Trustworthiness", eval_metrics["trustworthiness"], r)
                self.tb_writer.add_scalar("Metrics/Continuity", eval_metrics["continuity"], r)
                self.tb_writer.add_scalar("Metrics/kNN_Accuracy", eval_metrics["knn_accuracy"], r)
                self.tb_writer.add_scalar("Metrics/Steadiness", eval_metrics["steadiness"], r)
                self.tb_writer.add_scalar("Metrics/Cohesiveness", eval_metrics["cohesiveness"], r)
                
                self.tb_writer.add_scalar("Time/Communication", comm_time, r)
                self.tb_writer.add_scalar("Time/Local_Training", local_train_time, r)
                self.tb_writer.add_scalar("Time/Surrogate_Training", surr_train_time, r)
                self.tb_writer.add_scalar("Resource/GPU_Memory_MB", gpu_mem, r)
                
                # Detailed logs
                self.tb_writer.add_scalar("Validation/Attraction_Min", att_min_val, r)
                self.tb_writer.add_scalar("Validation/Attraction_Max", att_max_val, r)
                self.tb_writer.add_scalar("Validation/Local_Repulsion_Min", rep_min_val, r)
                self.tb_writer.add_scalar("Validation/Local_Repulsion_Max", rep_max_val, r)
                self.tb_writer.add_scalar("Validation/Surrogate_Repulsion_Min", surr_min_val, r)
                self.tb_writer.add_scalar("Validation/Surrogate_Repulsion_Max", surr_max_val, r)
                self.tb_writer.add_scalar("Validation/Phi_Min", phi_min_val, r)
                self.tb_writer.add_scalar("Validation/Phi_Max", phi_max_val, r)
                self.tb_writer.add_scalar("Validation/Distance_Min", dist_min_val, r)
                self.tb_writer.add_scalar("Validation/Distance_Max", dist_max_val, r)
                self.tb_writer.add_scalar("Validation/Encoder_Grad_Norm", avg_grad_norm, r)
                self.tb_writer.add_scalar("Validation/Embedding_Norm", avg_emb_norm, r)
                
                self.tb_writer.add_scalar("Validation/Surr_Target_Min", target_min_val, r)
                self.tb_writer.add_scalar("Validation/Surr_Target_Max", target_max_val, r)
                self.tb_writer.add_scalar("Validation/Surr_Prediction_Min", pred_min_val, r)
                self.tb_writer.add_scalar("Validation/Surr_Prediction_Max", pred_max_val, r)
                self.tb_writer.add_scalar("Validation/Surr_Grad_Norm", avg_surr_grad_norm, r)
                
            # --- Checkpoint Saving ---
            # Save latest checkpoint
            self.save_checkpoint(os.path.join(self.checkpoint_dir, "latest"), r)
            
            # Save best checkpoint
            if eval_metrics["knn_accuracy"] > self.best_knn_acc:
                self.best_knn_acc = eval_metrics["knn_accuracy"]
                self.save_checkpoint(os.path.join(self.checkpoint_dir, "best"), r)
                print(f"--> Saved new best checkpoint with kNN Acc: {self.best_knn_acc:.4f}")
                
        # Close tensorboard writer
        if self.tb_writer:
            self.tb_writer.close()
            
        print("Training completed successfully!")
