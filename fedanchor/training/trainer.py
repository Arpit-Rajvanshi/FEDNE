import os
import time
import yaml
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional
from fedanchor.utils.seed import set_seed
from fedanchor.datasets.loader import load_dataset
from fedanchor.server.server import AnchorServer
from fedanchor.client.client import AnchorClient
from fedanchor.evaluation.metrics import compute_metrics
from fedanchor.evaluation.visualization import save_all_visualizations

class FederatedAnchorTrainer:
    """
    Orchestrates Phase 2 Federated Anchor (FEDANCHOR) training loop, evaluation, diagnostics, checkpointing, and communication analysis.
    """
    def __init__(self, config_path: str, override_config: Optional[Dict[str, Any]] = None) -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        if override_config is not None:
            self._update_dict_recursive(self.config, override_config)
            
        seed_val = self.config.get("seed", {}).get("value", 42)
        set_seed(seed_val)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        ds_cfg = self.config.get("dataset", {})
        self.client_data, self.test_data = load_dataset(
            name=ds_cfg.get("name", "MNIST"),
            data_dir=ds_cfg.get("data_dir", "./data"),
            partition_type=ds_cfg.get("partition", "iid"),
            num_clients=ds_cfg.get("num_clients", 2),
            alpha=ds_cfg.get("alpha", 0.5),
            max_samples_per_client=ds_cfg.get("max_samples_per_client", None)
        )
        
        self.num_clients = len(self.client_data)
        
        model_cfg = self.config.get("model", {})
        self.server = AnchorServer(
            input_dim=model_cfg.get("input_dim", 784),
            embedding_dim=model_cfg.get("embedding_dim", 2),
            device=self.device
        )
        
        anc_cfg = self.config.get("anchors", {})
        graph_cfg = self.config.get("graph", {})
        self.clients: List[AnchorClient] = []
        for i, (X_c, y_c) in enumerate(self.client_data):
            client = AnchorClient(
                client_id=i,
                X_train=X_c,
                y_train=y_c,
                k_neighbors=graph_cfg.get("k", 5),
                num_anchors=anc_cfg.get("num_anchors", 5),
                device=self.device,
                seed=seed_val
            )
            self.clients.append(client)

        # v2: federation-wide size |D| for FEDNE-style |D_m|/|D| scaling
        total_n = sum(c.num_samples for c in self.clients)
        for c in self.clients:
            c.total_samples = total_n
            self.server.client_sizes[c.client_id] = c.num_samples
            
        eval_cfg = self.config.get("evaluation", {})
        self.output_dir = eval_cfg.get("output_dir", "./fedanchor/outputs")
        self.checkpoint_dir = eval_cfg.get("checkpoint_dir", "./fedanchor/checkpoints")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def scalars_per_anchor(self) -> int:
        # v2 shares (x, y, mass, spread) per anchor; legacy shares (x, y)
        return 4 if self.config.get("training", {}).get("mode", "fullbatch") == "neighbor_embedding" else 2

    def _update_dict_recursive(self, target: dict, source: dict) -> None:
        for k, v in source.items():
            if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                self._update_dict_recursive(target[k], v)
            else:
                target[k] = v

    def train(self) -> Dict[str, Any]:
        rounds = self.config.get("training", {}).get("rounds", 2)
        anc_num = self.config.get("anchors", {}).get("num_anchors", 5)
        
        comm_stats = self.server.compute_communication_bytes(
            num_clients=self.num_clients,
            num_anchors=anc_num,
            scalars_per_anchor=self.scalars_per_anchor()
        )
        
        history: List[Dict[str, Any]] = []
        start_time = time.time()
        
        for r in range(1, rounds + 1):
            round_start = time.time()
            global_state = self.server.get_global_encoder_state()
            
            client_states = []
            client_counts = []
            round_metrics_list = []
            
            for client in self.clients:
                other_anc = self.server.get_other_anchors(client.client_id)
                up_state, up_anc, metrics = client.local_train(
                    global_encoder_state=global_state,
                    other_anchors_tensor=other_anc,
                    config=self.config,
                    round_num=r,
                    other_anchor_meta=self.server.get_other_anchor_meta(client.client_id)
                )
                
                self.server.update_client_anchors(client.client_id, up_anc, meta=client.anchor_meta)
                
                client_states.append(up_state)
                client_counts.append(client.num_samples)
                round_metrics_list.append(metrics)
                
            self.server.aggregate_encoders(client_states, client_counts)
            
            avg_loss = np.mean([m["total_loss"] for m in round_metrics_list])
            avg_att = np.mean([m["attraction_loss"] for m in round_metrics_list])
            avg_anc = np.mean([m["anchor_loss"] for m in round_metrics_list])
            avg_rep_raw = np.mean([m["anchor_repulsion_raw"] for m in round_metrics_list])
            avg_rep_scaled = np.mean([m["anchor_repulsion_scaled"] for m in round_metrics_list])
            avg_rep_to_att = np.mean([m["repulsion_to_attraction_ratio"] for m in round_metrics_list])
            avg_rep_to_anc = np.mean([m["repulsion_to_anchor_ratio"] for m in round_metrics_list])
            
            # Evaluate global model
            X_test, y_test = self.test_data
            self.server.global_encoder.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
                X_test_low = self.server.global_encoder(X_test_tensor).cpu().numpy()
                
                client_embs_low = []
                X_train_low_list = []
                y_train_list = []
                for client in self.clients:
                    emb = client.embed_local().cpu().numpy()
                    client_embs_low.append(emb)
                    X_train_low_list.append(emb)
                    y_train_list.append(client.y_train)
                    
                X_train_low = np.concatenate(X_train_low_list, axis=0)
                y_train_cat = np.concatenate(y_train_list, axis=0)
                X_high_train = np.concatenate([c.X_train for c in self.clients], axis=0)
                
            eval_metrics = compute_metrics(
                X_high_train=X_high_train,
                X_low_train=X_train_low,
                y_train=y_train_cat,
                X_high_test=X_test,
                X_low_test=X_test_low,
                y_test=y_test,
                client_anchors=self.server.client_anchors,
                k=self.config.get("graph", {}).get("k", 5)
            )
            
            round_elapsed = time.time() - round_start
            
            utilization_str = ", ".join([f"{u:.1f}%" for u in eval_metrics["anchor_utilization"]])
            
            print(f"[Round {r:03d}/{rounds:03d}]")
            print(f"Loss: {avg_loss:.4f} | Attraction: {avg_att:.4f} | Anchor Loss: {avg_anc:.4f}")
            print(f"Repulsion Raw: {avg_rep_raw:.4f} | Repulsion Scaled: {avg_rep_scaled:.4f}")
            print(f"Diagnostics Ratios -> Rep/Att: {avg_rep_to_att:.2f} | Rep/Anc: {avg_rep_to_anc:.2f}")
            print(f"Trustworthiness: {eval_metrics['trustworthiness']:.4f} | Continuity: {eval_metrics['continuity']:.4f} | kNN Acc: {eval_metrics['knn_accuracy']:.4f}")
            print(f"Anchor Coverage (Mean Min Dist): {eval_metrics['anchor_coverage']:.4f} | Max Min Dist: {eval_metrics['anchor_max_distance']:.4f}")
            print(f"Anchor Utilization: [{utilization_str}]")
            print(f"Time: {round_elapsed:.2f}s\n")
            
            round_info = {
                "round": r,
                "loss": avg_loss,
                "attraction_loss": avg_att,
                "anchor_loss": avg_anc,
                "anchor_repulsion_raw": avg_rep_raw,
                "anchor_repulsion_scaled": avg_rep_scaled,
                "repulsion_to_attraction_ratio": avg_rep_to_att,
                "repulsion_to_anchor_ratio": avg_rep_to_anc,
                "eval_metrics": eval_metrics,
                "round_time": round_elapsed
            }
            history.append(round_info)
            
        total_training_time = time.time() - start_time
        
        viz_paths = save_all_visualizations(
            client_embeddings=client_embs_low,
            global_test_embeddings=X_test_low,
            y_test=y_test,
            client_anchors=self.server.client_anchors,
            output_dir=self.output_dir
        )
        
        ckpt_path = os.path.join(self.checkpoint_dir, "fedanchor_checkpoint.pt")
        torch.save({
            "global_encoder": self.server.global_encoder.state_dict(),
            "client_anchors": self.server.client_anchors,
            "config": self.config,
            "communication": comm_stats,
            "history": history
        }, ckpt_path)
        
        return {
            "history": history,
            "communication": comm_stats,
            "total_time": total_training_time,
            "visualizations": viz_paths,
            "checkpoint_path": ckpt_path,
            "final_metrics": history[-1]["eval_metrics"] if len(history) > 0 else {}
        }
