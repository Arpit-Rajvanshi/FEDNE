import os
import sys
import time
import json
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedne.training.trainer import Trainer as FEDNE_Trainer
from fedanchor.training.trainer import FederatedAnchorTrainer as FEDANCHOR_Trainer
from fedanchor.evaluation.metrics import compute_trustworthiness_and_continuity

OUTPUT_DIR = os.path.abspath("fedanchor/outputs/controlled_benchmark")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "fedne"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "fedanchor"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

# -----------------------------------------------------------------------------
# UNIFIED EVALUATOR
# -----------------------------------------------------------------------------
def unified_evaluate_global_embedding(global_encoder, X_train, y_train, X_test, y_test, device):
    """
    Unified evaluation: 
    - 2000 random train samples for kNN training
    - 2000 random test samples for kNN testing and T/C
    - k=5 for both kNN and T/C
    """
    # Fix seed for subset selection to ensure identical subsets every time
    np.random.seed(42)
    
    train_idx = np.random.choice(len(X_train), size=2000, replace=False)
    test_idx = np.random.choice(len(X_test), size=2000, replace=False)
    
    X_train_sub, y_train_sub = X_train[train_idx], y_train[train_idx]
    X_test_sub, y_test_sub = X_test[test_idx], y_test[test_idx]
    
    global_encoder.eval()
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train_sub, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test_sub, dtype=torch.float32).to(device)
        
        Z_train = global_encoder(X_train_tensor).cpu().numpy()
        Z_test = global_encoder(X_test_tensor).cpu().numpy()
        
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(Z_train, y_train_sub)
    knn_acc = clf.score(Z_test, y_test_sub)
    
    t, c = compute_trustworthiness_and_continuity(X_test_sub, Z_test, k=5)
    
    return float(knn_acc), float(t), float(c)

# -----------------------------------------------------------------------------
# MONKEY-PATCH FEDNE
# -----------------------------------------------------------------------------
class ControlledFEDNETrainer(FEDNE_Trainer):
    def evaluate_global_embedding(self, round_idx: int):
        X_train_all = np.concatenate([c.X_train for c in self.clients], axis=0)
        y_train_all = np.concatenate([c.y_train for c in self.clients], axis=0)
        
        knn_acc, t, c = unified_evaluate_global_embedding(
            self.server.global_encoder,
            X_train_all, y_train_all,
            self.X_test, self.y_test,
            self.device
        )
        return {
            "trustworthiness": t,
            "continuity": c,
            "knn_accuracy": knn_acc,
            "steadiness": 0.0,
            "cohesiveness": 0.0,
            "using_snc": 0.0
        }

# -----------------------------------------------------------------------------
# MONKEY-PATCH FEDANCHOR
# -----------------------------------------------------------------------------
import fedanchor.training.trainer
original_compute_metrics = fedanchor.training.trainer.compute_metrics

def patched_compute_metrics(X_high_train, X_low_train, y_train, X_high_test, X_low_test, y_test, client_anchors, k):
    # Call the original to get anchor diagnostics
    res = original_compute_metrics(X_high_train, X_low_train, y_train, X_high_test, X_low_test, y_test, client_anchors, k)
    
    # But overwrite T, C, and kNN with the unified evaluator
    # Since we need the global encoder, we get it from the trainer instance. 
    # This is a bit hacky, so let's just do the unified eval inside the trainer loop or here if we have it.
    pass

# We will just rewrite the evaluation part inside a subclass of FEDANCHOR_Trainer to be clean
class ControlledFEDANCHORTrainer(FEDANCHOR_Trainer):
    def train(self):
        # Mostly identical to original train, but overrides the eval metrics calculation
        rounds = self.config.get("training", {}).get("rounds", 2)
        anc_num = self.config.get("anchors", {}).get("num_anchors", 5)
        comm_stats = self.server.compute_communication_bytes(num_clients=self.num_clients, num_anchors=anc_num,
                                                             scalars_per_anchor=self.scalars_per_anchor())
        history = []
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
            
            # Use unified evaluator
            X_test, y_test = self.test_data
            X_high_train = np.concatenate([c.X_train for c in self.clients], axis=0)
            y_train_all = np.concatenate([c.y_train for c in self.clients], axis=0)
            
            knn_acc, t, c = unified_evaluate_global_embedding(
                self.server.global_encoder,
                X_high_train, y_train_all,
                X_test, y_test,
                self.device
            )
            
            # Compute anchor diagnostics (needs full train embeddings)
            self.server.global_encoder.eval()
            with torch.no_grad():
                X_train_low_list = []
                for client in self.clients:
                    emb = client.embed_local().cpu().numpy()
                    X_train_low_list.append(emb)
                X_train_low = np.concatenate(X_train_low_list, axis=0)
            
            from fedanchor.evaluation.metrics import compute_anchor_diagnostics
            all_anc_np = [anc.numpy() for anc in self.server.client_anchors.values()]
            if len(all_anc_np) > 0:
                concat_anc = np.concatenate(all_anc_np, axis=0)
                anc_diag = compute_anchor_diagnostics(X_train_low, concat_anc)
            else:
                anc_diag = {"mean_min_distance": 0.0, "max_min_distance": 0.0, "anchor_utilization": []}
                
            eval_metrics = {
                "trustworthiness": t,
                "continuity": c,
                "knn_accuracy": knn_acc,
                "anchor_coverage": anc_diag["mean_min_distance"],
                "anchor_max_distance": anc_diag["max_min_distance"],
                "anchor_utilization": anc_diag["anchor_utilization"]
            }
            
            round_elapsed = time.time() - round_start
            print(f"[FEDANCHOR round {r:02d}] loss={avg_loss:.4f} att={avg_att:.4f} "
                  f"anc_rep={avg_rep_scaled:.4f} | kNN={knn_acc:.4f} T={t:.4f} C={c:.4f} | {round_elapsed:.1f}s", flush=True)
            
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
        return {
            "history": history,
            "communication": comm_stats,
            "total_time": total_training_time,
            "final_metrics": history[-1]["eval_metrics"] if len(history) > 0 else {}
        }


# -----------------------------------------------------------------------------
# MAIN BENCHMARK RUNNER
# -----------------------------------------------------------------------------
def run_fedne():
    fedne_cfg = {
        "dataset": {"name": "MNIST", "data_dir": "./data"},
        "partition": {"type": "iid", "alpha": 0.5},
        "federated": {"clients": 2, "rounds": 20, "local_epochs": 1, "device": "cpu"},
        "model": {"input_dim": 784, "embedding_dim": 2, "hidden_dims": [512, 256, 128]},
        "graph": {"k": 5, "negative_samples": 5, "with_replacement": True},
        "augmentation": {"enabled": False},
        "surrogate": {"hidden_dim": 64, "grid_step": 0.3, "margin": 0.5, "lr": 0.001, "epochs": 2, "start_round": 0},
        "optimizer": {"lr": 0.001, "weight_decay": 0.0, "lr_decay_rounds": [], "lr_decay_factor": 1.0},
        "logging": {"tensorboard": False, "log_dir": "./runs", "checkpoint_dir": "./checkpoints", "output_dir": "./outputs", "seed": 42}
    }
    
    trainer = ControlledFEDNETrainer(fedne_cfg)
    trainer.fit()
    
    csv_path = os.path.join(OUTPUT_DIR, "fedne", "fedne_results.csv")
    enc_bytes = 2265608
    surr_bytes = 10000 
    upload_b = 2 * (enc_bytes + surr_bytes)
    
    rows = []
    for h in trainer.metrics_history:
        rows.append({
            "round": h["round"],
            "loss": h.get("loss", 0.0),
            "attraction": h.get("attraction_loss", 0.0),
            "repulsion_local": h.get("local_repulsion", 0.0),
            "repulsion_surrogate": h.get("surrogate_repulsion", 0.0),
            "trustworthiness": h.get("trustworthiness", 0.0),
            "continuity": h.get("continuity", 0.0),
            "knn_accuracy": h.get("knn_accuracy", 0.0),
            "runtime_seconds": h.get("round_time", 0.0),
            "upload_bytes": upload_b,
            "download_bytes": upload_b
        })
        
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        
    with open(os.path.join(OUTPUT_DIR, "fedne", "experiment_config.json"), "w") as f:
        json.dump(fedne_cfg, f, indent=2)
        
    return rows

def run_fedanchor():
    override = {
        "dataset": {"name": "MNIST", "data_dir": "./data", "num_clients": 2, "partition": "iid"},
        "anchors": {"num_anchors": 32},
        "training": {"mode": "neighbor_embedding", "rounds": 20, "local_epochs": 1,
                     "learning_rate": 0.001, "batch_size": 512},
        "graph": {"k": 5, "negative_samples": 5},
        "loss": {
            "lambda_attraction": 1.0,
            "lambda_local_repulsion": 1.0,
            "lambda_anchor_repulsion": 1.0,
            "anchor_repulsion": {"normalization": "none", "scale": 1.0, "spread_scale": 1.0}
        },
        "seed": {"value": 42},
        "evaluation": {"output_dir": os.path.join(OUTPUT_DIR, "fedanchor"), "checkpoint_dir": os.path.join(OUTPUT_DIR, "fedanchor")}
    }
    
    trainer = ControlledFEDANCHORTrainer(config_path="fedanchor/configs/mnist_default.yaml", override_config=override)
    res = trainer.train()
    
    csv_path = os.path.join(OUTPUT_DIR, "fedanchor", "fedanchor_results.csv")
    comm = res["communication"]
    rows = []
    
    for h in res["history"]:
        em = h["eval_metrics"]
        util_str = "[" + ", ".join([f"{u:.1f}%" for u in em.get("anchor_utilization", [])]) + "]"
        rows.append({
            "round": h["round"],
            "loss": h["loss"],
            "attraction": h["attraction_loss"],
            "anchor_loss": h["anchor_loss"],
            "repulsion_raw": h["anchor_repulsion_raw"],
            "repulsion_scaled": h["anchor_repulsion_scaled"],
            "trustworthiness": em["trustworthiness"],
            "continuity": em["continuity"],
            "knn_accuracy": em["knn_accuracy"],
            "anchor_utilization": util_str,
            "mean_nearest_anchor_distance": em["anchor_coverage"],
            "max_nearest_anchor_distance": em["anchor_max_distance"],
            "runtime_seconds": h["round_time"],
            "total_upload_bytes": comm["total_upload_bytes"],
            "total_download_bytes": comm["total_download_bytes"]
        })
        
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        
    with open(os.path.join(OUTPUT_DIR, "fedanchor", "experiment_config.json"), "w") as f:
        json.dump(override, f, indent=2)
        
    return rows

def generate_graphs(fedne_data, fedanchor_data):
    rounds = [d["round"] for d in fedne_data]
    
    # 1. kNN Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [d["knn_accuracy"] * 100 for d in fedne_data], 'o-', label='FEDNE', linewidth=2)
    plt.plot(rounds, [d["knn_accuracy"] * 100 for d in fedanchor_data], 's-', label='FEDANCHOR', linewidth=2)
    plt.title("kNN Accuracy vs Round (Controlled Protocol)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "1_accuracy_vs_round.png"), dpi=150)
    plt.close()

    # 2. Trustworthiness
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [d["trustworthiness"] for d in fedne_data], 'o-', label='FEDNE', linewidth=2)
    plt.plot(rounds, [d["trustworthiness"] for d in fedanchor_data], 's-', label='FEDANCHOR', linewidth=2)
    plt.title("Trustworthiness vs Round (Controlled Protocol)")
    plt.xlabel("Round")
    plt.ylabel("Trustworthiness")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "2_trustworthiness_vs_round.png"), dpi=150)
    plt.close()

    # 3. Continuity
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [d["continuity"] for d in fedne_data], 'o-', label='FEDNE', linewidth=2)
    plt.plot(rounds, [d["continuity"] for d in fedanchor_data], 's-', label='FEDANCHOR', linewidth=2)
    plt.title("Continuity vs Round (Controlled Protocol)")
    plt.xlabel("Round")
    plt.ylabel("Continuity")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "3_continuity_vs_round.png"), dpi=150)
    plt.close()

    # 4. Anchor Nearest Dist
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [d["mean_nearest_anchor_distance"] for d in fedanchor_data], 's-', color='orange', linewidth=2)
    plt.title("Mean Nearest Anchor Distance vs Round")
    plt.xlabel("Round")
    plt.ylabel("Euclidean Distance")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "4_nearest_anchor_distance_vs_round.png"), dpi=150)
    plt.close()
    
    # 5. Loss Comp
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [d["loss"] for d in fedanchor_data], label='Total Loss', linewidth=2)
    plt.plot(rounds, [d["attraction"] for d in fedanchor_data], label='Attraction', linewidth=2)
    plt.plot(rounds, [d["repulsion_scaled"] for d in fedanchor_data], label='Repulsion (Scaled)', linewidth=2)
    plt.plot(rounds, [d["anchor_loss"] for d in fedanchor_data], label='Anchor Coverage', linewidth=2)
    plt.title("FEDANCHOR Loss Components vs Round")
    plt.xlabel("Round")
    plt.yscale('log')
    plt.ylabel("Loss Value (Log Scale)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "5_loss_components_vs_round.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    print("RUNNING FEDNE CONTROLLED BENCHMARK...")
    fedne_rows = run_fedne()
    print("\nRUNNING FEDANCHOR CONTROLLED BENCHMARK...")
    fedanchor_rows = run_fedanchor()
    print("\nGENERATING GRAPHS...")
    generate_graphs(fedne_rows, fedanchor_rows)
    print("DONE.")
