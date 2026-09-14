import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fedanchor.training.trainer import FederatedAnchorTrainer

OUTPUT_DIR = os.path.abspath("fedanchor/outputs/controlled_ablation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_ablation():
    ablation_modes = {
        "A_attraction_only": {"use_attraction": True, "use_anchor": False, "use_anchor_repulsion": False},
        "B_attraction_anchor": {"use_attraction": True, "use_anchor": True, "use_anchor_repulsion": False},
        "C_attraction_repulsion": {"use_attraction": True, "use_anchor": False, "use_anchor_repulsion": True},
        "D_full_fedanchor": {"use_attraction": True, "use_anchor": True, "use_anchor_repulsion": True}
    }
    
    results = []
    
    for mode_name, ab_cfg in ablation_modes.items():
        print(f"Running Ablation Mode: {mode_name}")
        
        override = {
            "ablation": ab_cfg,
            "dataset": {"name": "MNIST", "data_dir": "./data", "num_clients": 2, "partition": "iid"},
            "anchors": {"num_anchors": 5, "initialization": "kmeans"},
            "training": {"rounds": 10, "local_epochs": 1},
            "loss": {
                "lambda_attraction": 1.0,
                "lambda_anchor": 1.0,
                "lambda_anchor_repulsion": 1.0,
                "anchor_repulsion": {"normalization": "scale", "scale": 0.01}
            },
            "seed": {"value": 42},
            "evaluation": {
                "output_dir": f"./fedanchor/outputs/controlled_ablation/{mode_name}",
                "checkpoint_dir": f"./fedanchor/checkpoints/controlled_ablation/{mode_name}"
            }
        }
        
        trainer = FederatedAnchorTrainer(config_path="fedanchor/configs/mnist_default.yaml", override_config=override)
        res = trainer.train()
        
        hist = res["history"]
        
        for h in hist:
            em = h["eval_metrics"]
            util_str = "[" + ", ".join([f"{u:.1f}%" for u in em.get("anchor_utilization", [])]) + "]"
            results.append({
                "mode": mode_name,
                "round": h["round"],
                "loss": h["loss"],
                "attraction": h["attraction_loss"],
                "anchor_loss": h["anchor_loss"],
                "repulsion_raw": h["anchor_repulsion_raw"],
                "repulsion_scaled": h["anchor_repulsion_scaled"],
                "trustworthiness": em["trustworthiness"],
                "continuity": em["continuity"],
                "knn_accuracy": em["knn_accuracy"],
                "anchor_utilization": util_str
            })
            
    csv_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Ablation results saved to {csv_path}")

if __name__ == "__main__":
    run_ablation()
