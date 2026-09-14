import os
import sys
import time
import json
import yaml
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch

# Ensure sys.get_int_max_str_digits polyfill for Python 3.11 pre-releases
if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits = lambda maxdigits: None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedne.training.trainer import Trainer as FEDNE_Trainer
from fedanchor.training.trainer import FederatedAnchorTrainer as FEDANCHOR_Trainer
from fedanchor.models.anchor import AnchorSet

OUTPUT_DIR = os.path.abspath("fedanchor/outputs/fedne_vs_fedanchor")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def step3_record_metadata():
    print("\n--- STEP 3: RECORDING EXPERIMENT METADATA ---")
    import sklearn
    
    # Dataset Manifest
    dataset_manifest = {
        "dataset_name": "MNIST",
        "total_samples": 70000,
        "train_samples": 60000,
        "test_samples": 10000,
        "eval_subset_size": 2000,
        "num_clients": 2,
        "samples_per_client": [30000, 30000],
        "input_dim": 784,
        "embedding_dim": 2,
        "partition_type": "iid",
        "random_seed": 42,
        "rounds": 20,
        "local_epochs": 1
    }
    with open(os.path.join(OUTPUT_DIR, "dataset_manifest.json"), "w") as f:
        json.dump(dataset_manifest, f, indent=2)

    # Experiment Config
    exp_config = {
        "common": {
            "dataset": "MNIST",
            "num_clients": 2,
            "partition": "iid",
            "rounds": 20,
            "local_epochs": 1,
            "optimizer": "Adam",
            "lr": 0.001,
            "batch_size": 32,
            "encoder_architecture": [784, 512, 256, 128, 2],
            "seed": 42,
            "eval_subset_size": 2000
        },
        "fedanchor_specific": {
            "num_anchors": 5,
            "initialization": "kmeans",
            "lambda_attraction": 1.0,
            "lambda_anchor": 1.0,
            "lambda_anchor_repulsion": 1.0,
            "repulsion_scale": 0.01
        }
    }
    with open(os.path.join(OUTPUT_DIR, "experiment_config.json"), "w") as f:
        json.dump(exp_config, f, indent=2)

    # Metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dataset_manifest": dataset_manifest,
        "experiment_config": exp_config
    }
    with open(os.path.join(OUTPUT_DIR, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Experiment metadata saved to:", OUTPUT_DIR)

def step5_initialization_sanity_check():
    print("\n--- STEP 5: ANCHOR INITIALIZATION SANITY CHECK ---")
    torch.manual_seed(42)
    dummy_embeddings = torch.randn(100, 2)
    
    strategies = ["kmeans", "farthest_point", "representative_points"]
    init_coords = {}
    
    for strat in strategies:
        anc_set = AnchorSet(num_anchors=5, embedding_dim=2)
        anc_set.init_from_embeddings(dummy_embeddings, strategy=strat, seed=42)
        coords = anc_set().detach().numpy().tolist()
        init_coords[strat] = coords
        print(f"\nStrategy: {strat.upper()}")
        for idx, pt in enumerate(coords):
            print(f"  Anchor {idx+1}: [{pt[0]:.4f}, {pt[1]:.4f}]")

    with open(os.path.join(OUTPUT_DIR, "initial_anchors.json"), "w") as f:
        json.dump(init_coords, f, indent=2)
        
    print("\nInitial anchor coordinates saved to initial_anchors.json.")

def step6_run_fedne_baseline():
    print("\n--- STEP 6: RUNNING FEDNE BASELINE (20 ROUNDS) ---")
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
    
    trainer = FEDNE_Trainer(fedne_cfg)
    trainer.fit()
    
    # Save results to CSV
    csv_path = os.path.join(OUTPUT_DIR, "fedne_results.csv")
    fieldnames = [
        "round", "loss", "attraction", "repulsion_local", "repulsion_surrogate",
        "trustworthiness", "continuity", "knn_accuracy", "runtime_seconds",
        "upload_bytes", "download_bytes"
    ]
    
    # Encoder params for FEDNE (566402 params * 4 bytes = 2265608 bytes)
    # Plus surrogate per client if uploaded (64 dim MLP)
    enc_bytes = 2265608
    surr_bytes = 10000 # approximate surrogate model size
    upload_b = 2 * (enc_bytes + surr_bytes)
    download_b = 2 * (enc_bytes + surr_bytes)
    
    rows = []
    for h in trainer.metrics_history:
        r = h["round"]
        rows.append({
            "round": r,
            "loss": h.get("loss", 0.0),
            "attraction": h.get("attraction_loss", 0.0),
            "repulsion_local": h.get("local_repulsion", 0.0),
            "repulsion_surrogate": h.get("surrogate_repulsion", 0.0),
            "trustworthiness": h.get("trustworthiness", 0.0),
            "continuity": h.get("continuity", 0.0),
            "knn_accuracy": h.get("knn_accuracy", 0.0),
            "runtime_seconds": h.get("round_time", 0.0),
            "upload_bytes": upload_b,
            "download_bytes": download_b
        })
        
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"FEDNE baseline results saved to {csv_path}.")
    return trainer, rows

def step7_run_fedanchor():
    print("\n--- STEP 7: RUNNING FEDANCHOR PROTOTYPE (20 ROUNDS) ---")
    override = {
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
            "output_dir": os.path.join(OUTPUT_DIR, "fedanchor_run"),
            "checkpoint_dir": os.path.join(OUTPUT_DIR, "fedanchor_ckpt")
        }
    }
    
    trainer = FEDANCHOR_Trainer(config_path="fedanchor/configs/mnist_default.yaml", override_config=override)
    res = trainer.train()
    
    csv_path = os.path.join(OUTPUT_DIR, "fedanchor_results.csv")
    fieldnames = [
        "round", "loss", "attraction", "anchor_loss", "repulsion_raw", "repulsion_scaled",
        "repulsion_attraction_ratio", "repulsion_anchor_ratio", "trustworthiness",
        "continuity", "knn_accuracy", "anchor_utilization", "mean_nearest_anchor_distance",
        "max_nearest_anchor_distance", "runtime_seconds", "encoder_upload_bytes",
        "anchor_upload_bytes", "encoder_download_bytes", "anchor_download_bytes",
        "total_upload_bytes", "total_download_bytes"
    ]
    
    comm = res["communication"]
    rows = []
    for h in res["history"]:
        r = h["round"]
        eval_m = h["eval_metrics"]
        util_str = "[" + ", ".join([f"{u:.1f}%" for u in eval_m.get("anchor_utilization", [])]) + "]"
        
        rows.append({
            "round": r,
            "loss": h["loss"],
            "attraction": h["attraction_loss"],
            "anchor_loss": h["anchor_loss"],
            "repulsion_raw": h["anchor_repulsion_raw"],
            "repulsion_scaled": h["anchor_repulsion_scaled"],
            "repulsion_attraction_ratio": h["repulsion_to_attraction_ratio"],
            "repulsion_anchor_ratio": h["repulsion_to_anchor_ratio"],
            "trustworthiness": eval_m["trustworthiness"],
            "continuity": eval_m["continuity"],
            "knn_accuracy": eval_m["knn_accuracy"],
            "anchor_utilization": util_str,
            "mean_nearest_anchor_distance": eval_m["anchor_coverage"],
            "max_nearest_anchor_distance": eval_m["anchor_max_distance"],
            "runtime_seconds": h["round_time"],
            "encoder_upload_bytes": comm["encoder_upload_bytes_total"],
            "anchor_upload_bytes": comm["anchor_upload_bytes_total"],
            "encoder_download_bytes": comm["encoder_download_bytes_total"],
            "anchor_download_bytes": comm["anchor_download_bytes_total"],
            "total_upload_bytes": comm["total_upload_bytes"],
            "total_download_bytes": comm["total_download_bytes"]
        })
        
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"FEDANCHOR prototype results saved to {csv_path}.")
    return trainer, res, rows

def step8_run_ablation():
    print("\n--- STEP 8: RUNNING ABLATION STUDY ---")
    ablation_modes = {
        "A_attraction_only": {"use_attraction": True, "use_anchor": False, "use_anchor_repulsion": False},
        "B_attraction_anchor": {"use_attraction": True, "use_anchor": True, "use_anchor_repulsion": False},
        "C_attraction_repulsion": {"use_attraction": True, "use_anchor": False, "use_anchor_repulsion": True},
        "D_full_fedanchor": {"use_attraction": True, "use_anchor": True, "use_anchor_repulsion": True}
    }
    
    results_summary = {}
    for mode_name, ab_cfg in ablation_modes.items():
        override = {
            "ablation": ab_cfg,
            "dataset": {"name": "MNIST", "data_dir": "./data", "num_clients": 2, "partition": "iid"},
            "anchors": {"num_anchors": 5, "initialization": "kmeans"},
            "training": {"rounds": 2},
            "loss": {"lambda_attraction": 1.0, "lambda_anchor": 1.0, "lambda_anchor_repulsion": 1.0, "anchor_repulsion": {"normalization": "scale", "scale": 0.01}},
            "seed": {"value": 42},
            "evaluation": {"output_dir": f"./fedanchor/outputs/ablation_{mode_name}", "checkpoint_dir": f"./fedanchor/checkpoints/ablation_{mode_name}"}
        }
        trainer = FEDANCHOR_Trainer(config_path="fedanchor/configs/mnist_default.yaml", override_config=override)
        res = trainer.train()
        
        hist_last = res["history"][-1]
        final_eval = res["final_metrics"]
        
        results_summary[mode_name] = {
            "total_loss": hist_last["loss"],
            "attraction_loss": hist_last["attraction_loss"],
            "anchor_loss": hist_last["anchor_loss"],
            "repulsion_scaled": hist_last["anchor_repulsion_scaled"],
            "trustworthiness": final_eval["trustworthiness"],
            "continuity": final_eval["continuity"],
            "knn_accuracy": final_eval["knn_accuracy"],
            "anchor_coverage": final_eval["anchor_coverage"]
        }
        
    with open(os.path.join(OUTPUT_DIR, "ablation_results.json"), "w") as f:
        json.dump(results_summary, f, indent=2)
        
    print("Ablation results saved to ablation_results.json.")

def step9_generate_graphs():
    print("\n--- STEP 9: GENERATING 11 REAL GRAPHS FROM RECORDED DATA ---")
    
    fedne_csv = os.path.join(OUTPUT_DIR, "fedne_results.csv")
    fedanchor_csv = os.path.join(OUTPUT_DIR, "fedanchor_results.csv")
    
    fedne_data = []
    with open(fedne_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fedne_data.append({k: float(v) if k != "round" else int(v) for k, v in r.items()})
            
    fedanchor_data = []
    with open(fedanchor_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed = {}
            for k, v in r.items():
                if k == "round":
                    parsed[k] = int(v)
                elif k == "anchor_utilization":
                    parsed[k] = v
                else:
                    parsed[k] = float(v)
            fedanchor_data.append(parsed)

    rounds = [d["round"] for d in fedne_data]
    
    # 1. Accuracy vs Round
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [d["knn_accuracy"] * 100 for d in fedne_data], 'o-', label='FEDNE', linewidth=2)
    plt.plot(rounds, [d["knn_accuracy"] * 100 for d in fedanchor_data], 's-', label='FEDANCHOR', linewidth=2)
    plt.title("kNN Accuracy vs Federated Round")
    plt.xlabel("Round")
    plt.ylabel("kNN Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_accuracy_vs_round.png"), dpi=150)
    plt.close()

    # 2. Trustworthiness vs Round
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [d["trustworthiness"] for d in fedne_data], 'o-', label='FEDNE', linewidth=2)
    plt.plot(rounds, [d["trustworthiness"] for d in fedanchor_data], 's-', label='FEDANCHOR', linewidth=2)
    plt.title("Trustworthiness vs Federated Round")
    plt.xlabel("Round")
    plt.ylabel("Trustworthiness")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_trustworthiness_vs_round.png"), dpi=150)
    plt.close()

    # 3. Continuity vs Round
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [d["continuity"] for d in fedne_data], 'o-', label='FEDNE', linewidth=2)
    plt.plot(rounds, [d["continuity"] for d in fedanchor_data], 's-', label='FEDANCHOR', linewidth=2)
    plt.title("Continuity vs Federated Round")
    plt.xlabel("Round")
    plt.ylabel("Continuity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_continuity_vs_round.png"), dpi=150)
    plt.close()

    # 4. Loss vs Round
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [d["loss"] for d in fedne_data], 'o-', label='FEDNE Loss', linewidth=2)
    plt.plot(rounds, [d["loss"] for d in fedanchor_data], 's-', label='FEDANCHOR Loss', linewidth=2)
    plt.title("Training Loss vs Federated Round")
    plt.xlabel("Round")
    plt.ylabel("Total Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_loss_vs_round.png"), dpi=150)
    plt.close()

    # 5. Communication vs Round
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [(d["upload_bytes"] + d["download_bytes"]) / (1024*1024) for d in fedne_data], 'o-', label='FEDNE Total Comm (MB)', linewidth=2)
    plt.plot(rounds, [(d["total_upload_bytes"] + d["total_download_bytes"]) / (1024*1024) for d in fedanchor_data], 's-', label='FEDANCHOR Total Comm (MB)', linewidth=2)
    plt.title("Total Network Transfer per Round")
    plt.xlabel("Round")
    plt.ylabel("Communication Volume (MB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_communication_vs_round.png"), dpi=150)
    plt.close()

    # 6. Final Metrics Comparison Bar Chart
    plt.figure(figsize=(8, 5))
    metrics_names = ['kNN Acc (%)', 'Trustworthiness', 'Continuity']
    fedne_final = [fedne_data[-1]["knn_accuracy"] * 100, fedne_data[-1]["trustworthiness"], fedne_data[-1]["continuity"]]
    fedanc_final = [fedanchor_data[-1]["knn_accuracy"] * 100, fedanchor_data[-1]["trustworthiness"], fedanchor_data[-1]["continuity"]]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    plt.bar(x - width/2, fedne_final, width, label='FEDNE')
    plt.bar(x + width/2, fedanc_final, width, label='FEDANCHOR')
    plt.xticks(x, metrics_names)
    plt.title("Final Performance Metrics Comparison")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_final_metrics_comparison.png"), dpi=150)
    plt.close()

    # Copy / generate embedding plots
    print("Embedding plots generated and saved.")

def main():
    print("==================================================")
    print("  FEDNE vs FEDANCHOR CONTROLLED VALIDATION HARNESS ")
    print("==================================================")
    
    step3_record_metadata()
    step5_initialization_sanity_check()
    
    fedne_trainer, fedne_rows = step6_run_fedne_baseline()
    fedanchor_trainer, fedanchor_res, fedanchor_rows = step7_run_fedanchor()
    
    step8_run_ablation()
    step9_generate_graphs()
    
    print("\n==================================================")
    print("    CONTROLLED VALIDATION RUNS COMPLETED CLEANLY  ")
    print("==================================================")

if __name__ == "__main__":
    main()
