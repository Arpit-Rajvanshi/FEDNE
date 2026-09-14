import os
import sys
import time
import json
import yaml
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch

if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits = lambda maxdigits: None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedne.training.trainer import Trainer as FEDNE_Trainer
from fedanchor.training.trainer import FederatedAnchorTrainer as FEDANCHOR_Trainer
from fedanchor.models.anchor import AnchorSet

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs/fedne_vs_fedanchor"))
os.makedirs(OUT_DIR, exist_ok=True)

def run_all():
    print("==================================================")
    print(" GENERATING CONTROLLED VALIDATION RUNS & REPORTS ")
    print("==================================================")

    # 1. Run FEDNE Baseline (20 Rounds)
    print("\n---> Running FEDNE Baseline (20 Rounds) <---")
    fedne_cfg = {
        "dataset": {"name": "MNIST", "data_dir": "./data"},
        "partition": {"type": "iid", "alpha": 0.5},
        "federated": {"clients": 2, "rounds": 20, "local_epochs": 1, "device": "cpu"},
        "model": {"input_dim": 784, "embedding_dim": 2, "hidden_dims": [512, 256, 128]},
        "graph": {"k": 5, "negative_samples": 5, "with_replacement": True},
        "augmentation": {"enabled": False},
        "surrogate": {"hidden_dim": 64, "grid_step": 0.3, "margin": 0.5, "lr": 0.001, "epochs": 2, "start_round": 0},
        "optimizer": {"lr": 0.001, "weight_decay": 0.0, "lr_decay_rounds": [], "lr_decay_factor": 1.0},
        "logging": {"tensorboard": False, "log_dir": "./runs", "checkpoint_dir": "./checkpoints", "output_dir": "./outputs", "seed": 42, "eval_subset_size": 2000}
    }
    
    fedne_trainer = FEDNE_Trainer(fedne_cfg)
    fedne_trainer.fit()
    
    fedne_csv = os.path.join(OUT_DIR, "fedne_results.csv")
    fedne_fields = [
        "round", "loss", "attraction", "repulsion_local", "repulsion_surrogate",
        "trustworthiness", "continuity", "knn_accuracy", "runtime_seconds",
        "upload_bytes", "download_bytes"
    ]
    
    enc_bytes = 2265608
    surr_bytes = 10000
    upload_b = 2 * (enc_bytes + surr_bytes)
    download_b = 2 * (enc_bytes + surr_bytes)
    
    fedne_rows = []
    for h in fedne_trainer.metrics_history:
        r = h["round"] + 1
        fedne_rows.append({
            "round": r,
            "loss": float(h.get("loss", 0.0)),
            "attraction": float(h.get("attraction_loss", 0.0)),
            "repulsion_local": float(h.get("local_repulsion", 0.0)),
            "repulsion_surrogate": float(h.get("surrogate_repulsion", 0.0)),
            "trustworthiness": float(h.get("trustworthiness", 0.0)),
            "continuity": float(h.get("continuity", 0.0)),
            "knn_accuracy": float(h.get("knn_accuracy", 0.0)),
            "runtime_seconds": float(h.get("round_time", 0.0)),
            "upload_bytes": upload_b,
            "download_bytes": download_b
        })
        
    with open(fedne_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fedne_fields)
        writer.writeheader()
        writer.writerows(fedne_rows)
    print("Saved:", fedne_csv)

    # 2. Run FEDANCHOR Prototype (20 Rounds)
    print("\n---> Running FEDANCHOR Prototype (20 Rounds) <---")
    fedanchor_override = {
        "dataset": {"name": "MNIST", "data_dir": "./data", "num_clients": 2, "partition": "iid"},
        "anchors": {"num_anchors": 5, "initialization": "kmeans"},
        "training": {"rounds": 20, "local_epochs": 1, "learning_rate": 0.001},
        "loss": {
            "lambda_attraction": 1.0,
            "lambda_anchor": 1.0,
            "lambda_anchor_repulsion": 1.0,
            "anchor_repulsion": {"normalization": "scale", "scale": 0.01}
        },
        "seed": {"value": 42},
        "evaluation": {
            "output_dir": os.path.join(OUT_DIR, "fedanchor_run"),
            "checkpoint_dir": os.path.join(OUT_DIR, "fedanchor_ckpt")
        }
    }
    
    fedanchor_trainer = FEDANCHOR_Trainer(config_path="fedanchor/configs/mnist_default.yaml", override_config=fedanchor_override)
    res = fedanchor_trainer.train()
    
    fedanchor_csv = os.path.join(OUT_DIR, "fedanchor_results.csv")
    fedanchor_fields = [
        "round", "loss", "attraction", "anchor_loss", "repulsion_raw", "repulsion_scaled",
        "repulsion_attraction_ratio", "repulsion_anchor_ratio", "trustworthiness",
        "continuity", "knn_accuracy", "anchor_utilization", "mean_nearest_anchor_distance",
        "max_nearest_anchor_distance", "runtime_seconds", "encoder_upload_bytes",
        "anchor_upload_bytes", "encoder_download_bytes", "anchor_download_bytes",
        "total_upload_bytes", "total_download_bytes"
    ]
    
    comm = res["communication"]
    fedanchor_rows = []
    for h in res["history"]:
        r = h["round"]
        eval_m = h["eval_metrics"]
        util_str = "[" + ", ".join([f"{u:.1f}%" for u in eval_m.get("anchor_utilization", [])]) + "]"
        
        fedanchor_rows.append({
            "round": r,
            "loss": float(h["loss"]),
            "attraction": float(h["attraction_loss"]),
            "anchor_loss": float(h["anchor_loss"]),
            "repulsion_raw": float(h["anchor_repulsion_raw"]),
            "repulsion_scaled": float(h["anchor_repulsion_scaled"]),
            "repulsion_attraction_ratio": float(h["repulsion_to_attraction_ratio"]),
            "repulsion_anchor_ratio": float(h["repulsion_to_anchor_ratio"]),
            "trustworthiness": float(eval_m["trustworthiness"]),
            "continuity": float(eval_m["continuity"]),
            "knn_accuracy": float(eval_m["knn_accuracy"]),
            "anchor_utilization": util_str,
            "mean_nearest_anchor_distance": float(eval_m["anchor_coverage"]),
            "max_nearest_anchor_distance": float(eval_m["anchor_max_distance"]),
            "runtime_seconds": float(h["round_time"]),
            "encoder_upload_bytes": comm["encoder_upload_bytes_total"],
            "anchor_upload_bytes": comm["anchor_upload_bytes_total"],
            "encoder_download_bytes": comm["encoder_download_bytes_total"],
            "anchor_download_bytes": comm["anchor_download_bytes_total"],
            "total_upload_bytes": comm["total_upload_bytes"],
            "total_download_bytes": comm["total_download_bytes"]
        })
        
    with open(fedanchor_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fedanchor_fields)
        writer.writeheader()
        writer.writerows(fedanchor_rows)
    print("Saved:", fedanchor_csv)

    # 3. Ablation Study
    print("\n---> Running Ablation Study <---")
    ablation_modes = {
        "A_attraction_only": {"use_attraction": True, "use_anchor": False, "use_anchor_repulsion": False},
        "B_attraction_anchor": {"use_attraction": True, "use_anchor": True, "use_anchor_repulsion": False},
        "C_attraction_repulsion": {"use_attraction": True, "use_anchor": False, "use_anchor_repulsion": True},
        "D_full_fedanchor": {"use_attraction": True, "use_anchor": True, "use_anchor_repulsion": True}
    }
    
    ablation_res = {}
    for mode_name, ab_cfg in ablation_modes.items():
        override = {
            "ablation": ab_cfg,
            "dataset": {"name": "MNIST", "data_dir": "./data", "num_clients": 2, "partition": "iid"},
            "anchors": {"num_anchors": 5, "initialization": "kmeans"},
            "training": {"rounds": 2},
            "loss": {"lambda_attraction": 1.0, "lambda_anchor": 1.0, "lambda_anchor_repulsion": 1.0, "anchor_repulsion": {"normalization": "scale", "scale": 0.01}},
            "seed": {"value": 42},
            "evaluation": {"output_dir": os.path.join(OUT_DIR, f"ablation_{mode_name}"), "checkpoint_dir": os.path.join(OUT_DIR, f"ablation_ckpt_{mode_name}")}
        }
        tr = FEDANCHOR_Trainer(config_path="fedanchor/configs/mnist_default.yaml", override_config=override)
        out = tr.train()
        
        hist_last = out["history"][-1]
        final_eval = out["final_metrics"]
        
        ablation_res[mode_name] = {
            "total_loss": float(hist_last["loss"]),
            "attraction_loss": float(hist_last["attraction_loss"]),
            "anchor_loss": float(hist_last["anchor_loss"]),
            "repulsion_scaled": float(hist_last["anchor_repulsion_scaled"]),
            "trustworthiness": float(final_eval["trustworthiness"]),
            "continuity": float(final_eval["continuity"]),
            "knn_accuracy": float(final_eval["knn_accuracy"]),
            "anchor_coverage": float(final_eval["anchor_coverage"])
        }
        
    ablation_json = os.path.join(OUT_DIR, "ablation_results.json")
    with open(ablation_json, "w") as f:
        json.dump(ablation_res, f, indent=2)
    print("Saved:", ablation_json)

    # 4. Generate 11 Real Graphs directly from loaded CSV data
    print("\n---> Generating 11 Real Graphs <---")
    rounds = [r["round"] for r in fedne_rows]

    # Graph 1: kNN Accuracy vs Round
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [r["knn_accuracy"] * 100 for r in fedne_rows], 'o-', label='FEDNE', linewidth=2)
    plt.plot(rounds, [r["knn_accuracy"] * 100 for r in fedanchor_rows], 's-', label='FEDANCHOR', linewidth=2)
    plt.title("kNN Accuracy vs Federated Round")
    plt.xlabel("Round")
    plt.ylabel("kNN Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "1_accuracy_vs_round.png"), dpi=150)
    plt.close()

    # Graph 2: Trustworthiness vs Round
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [r["trustworthiness"] for r in fedne_rows], 'o-', label='FEDNE', linewidth=2)
    plt.plot(rounds, [r["trustworthiness"] for r in fedanchor_rows], 's-', label='FEDANCHOR', linewidth=2)
    plt.title("Trustworthiness vs Federated Round")
    plt.xlabel("Round")
    plt.ylabel("Trustworthiness")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "2_trustworthiness_vs_round.png"), dpi=150)
    plt.close()

    # Graph 3: Continuity vs Round
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [r["continuity"] for r in fedne_rows], 'o-', label='FEDNE', linewidth=2)
    plt.plot(rounds, [r["continuity"] for r in fedanchor_rows], 's-', label='FEDANCHOR', linewidth=2)
    plt.title("Continuity vs Federated Round")
    plt.xlabel("Round")
    plt.ylabel("Continuity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "3_continuity_vs_round.png"), dpi=150)
    plt.close()

    # Graph 4: Loss vs Round
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [r["loss"] for r in fedne_rows], 'o-', label='FEDNE Loss', linewidth=2)
    plt.plot(rounds, [r["loss"] for r in fedanchor_rows], 's-', label='FEDANCHOR Loss', linewidth=2)
    plt.title("Training Loss vs Federated Round")
    plt.xlabel("Round")
    plt.ylabel("Total Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "4_loss_vs_round.png"), dpi=150)
    plt.close()

    # Graph 5: Communication vs Round
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [(r["upload_bytes"] + r["download_bytes"]) / (1024*1024) for r in fedne_rows], 'o-', label='FEDNE Comm (MB)', linewidth=2)
    plt.plot(rounds, [(r["total_upload_bytes"] + r["total_download_bytes"]) / (1024*1024) for r in fedanchor_rows], 's-', label='FEDANCHOR Comm (MB)', linewidth=2)
    plt.title("Total Network Transfer per Round")
    plt.xlabel("Round")
    plt.ylabel("Communication Volume (MB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "5_communication_vs_round.png"), dpi=150)
    plt.close()

    # Graph 6: Final Metrics Comparison Bar Chart
    plt.figure(figsize=(8, 5))
    metrics_names = ['kNN Acc (%)', 'Trustworthiness', 'Continuity']
    fedne_final = [fedne_rows[-1]["knn_accuracy"] * 100, fedne_rows[-1]["trustworthiness"], fedne_rows[-1]["continuity"]]
    fedanc_final = [fedanchor_rows[-1]["knn_accuracy"] * 100, fedanchor_rows[-1]["trustworthiness"], fedanchor_rows[-1]["continuity"]]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    plt.bar(x - width/2, fedne_final, width, label='FEDNE Baseline')
    plt.bar(x + width/2, fedanc_final, width, label='FEDANCHOR Prototype')
    plt.xticks(x, metrics_names)
    plt.title("Final Performance Metrics Comparison")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "6_final_metrics_comparison.png"), dpi=150)
    plt.close()

    # Get test embeddings for FEDNE & FEDANCHOR
    X_test = fedne_trainer.X_test[:2000]
    y_test = fedne_trainer.y_test[:2000]
    
    # FEDNE final embeddings
    fedne_trainer.server.global_encoder.eval()
    with torch.no_grad():
        z_fedne = fedne_trainer.server.global_encoder(torch.from_numpy(X_test)).cpu().numpy()
        
    # FEDANCHOR final embeddings
    fedanchor_trainer.server.global_encoder.eval()
    with torch.no_grad():
        z_fedanchor = fedanchor_trainer.server.global_encoder(torch.from_numpy(X_test)).cpu().numpy()

    # Graph 7: FEDNE final embeddings
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_fedne[:, 0], z_fedne[:, 1], c=y_test, cmap='tab10', s=10, alpha=0.8)
    plt.colorbar(scatter, label='Digit Class')
    plt.title("FEDNE Final 2D Test Embeddings (Round 20)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "7_fedne_final_embeddings.png"), dpi=150)
    plt.close()

    # Graph 8: FEDANCHOR final embeddings
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_fedanchor[:, 0], z_fedanchor[:, 1], c=y_test, cmap='tab10', s=10, alpha=0.8)
    plt.colorbar(scatter, label='Digit Class')
    plt.title("FEDANCHOR Final 2D Test Embeddings (Round 20)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "8_fedanchor_final_embeddings.png"), dpi=150)
    plt.close()

    # Graph 9: FEDANCHOR embeddings + anchors
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_fedanchor[:, 0], z_fedanchor[:, 1], c=y_test, cmap='tab10', s=10, alpha=0.6)
    plt.colorbar(scatter, label='Digit Class')
    
    # Plot client anchors
    client0_anc = fedanchor_trainer.server.client_anchors[0].numpy()
    client1_anc = fedanchor_trainer.server.client_anchors[1].numpy()
    plt.scatter(client0_anc[:, 0], client0_anc[:, 1], c='red', marker='X', s=150, linewidths=2, edgecolor='black', label='Client 0 Anchors')
    plt.scatter(client1_anc[:, 0], client1_anc[:, 1], c='cyan', marker='^', s=150, linewidths=2, edgecolor='black', label='Client 1 Anchors')
    
    plt.title("FEDANCHOR 2D Embeddings with Learned Client Anchors")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "9_fedanchor_embeddings_with_anchors.png"), dpi=150)
    plt.close()

    # Graph 10: FEDANCHOR anchor utilization vs round
    # Extract utilization of dominant anchor from rows
    plt.figure(figsize=(8, 5))
    dominant_util = []
    for r in fedanchor_rows:
        u_str = r["anchor_utilization"].strip("[]").split(", ")
        vals = [float(v.rstrip("%")) for v in u_str]
        dominant_util.append(max(vals))
    plt.plot(rounds, dominant_util, 'o-', color='purple', linewidth=2)
    plt.title("FEDANCHOR Maximum Single-Anchor Utilization % vs Round")
    plt.xlabel("Round")
    plt.ylabel("Max Utilization (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "10_fedanchor_utilization_vs_round.png"), dpi=150)
    plt.close()

    # Graph 11: FEDANCHOR nearest anchor distance vs round
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [r["mean_nearest_anchor_distance"] for r in fedanchor_rows], 'o-', label='Mean Min Dist', linewidth=2)
    plt.plot(rounds, [r["max_nearest_anchor_distance"] for r in fedanchor_rows], 's-', label='Max Min Dist', linewidth=2)
    plt.title("FEDANCHOR Nearest-Anchor Distance vs Round")
    plt.xlabel("Round")
    plt.ylabel("Distance (2D Embedding Space)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "11_fedanchor_nearest_anchor_distance_vs_round.png"), dpi=150)
    plt.close()

    print("All 11 graphs generated cleanly in:", OUT_DIR)

if __name__ == "__main__":
    run_all()
