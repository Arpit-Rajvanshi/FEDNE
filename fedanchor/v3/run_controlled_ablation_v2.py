import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fedanchor.v2.training.trainer_v2 import FederatedAnchorTrainerV2

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs/controlled_ablation_v2"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def entropy(utilization_percentages):
    # Convert percentages to probabilities
    probs = np.array(utilization_percentages) / 100.0
    # Avoid log(0)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    h = -np.sum(probs * np.log(probs))
    return h

def run_ablation():
    ablation_modes = {
        "A_V1_Hard": {
            "use_attraction": True, 
            "use_anchor": True, 
            "use_anchor_repulsion": True,
            "use_spread": False,
            "anchor_assignment_type": "hard"
        },
        "B_Soft_Only": {
            "use_attraction": True, 
            "use_anchor": True, 
            "use_anchor_repulsion": True,
            "use_spread": False,
            "anchor_assignment_type": "soft"
        },
        "C_Spread_Only": {
            "use_attraction": True, 
            "use_anchor": True, 
            "use_anchor_repulsion": True,
            "use_spread": True,
            "anchor_assignment_type": "hard"
        },
        "D_Soft_And_Spread": {
            "use_attraction": True, 
            "use_anchor": True, 
            "use_anchor_repulsion": True,
            "use_spread": True,
            "anchor_assignment_type": "soft"
        }
    }
    
    results = []
    
    for mode_name, ab_cfg in ablation_modes.items():
        print(f"==================================================")
        print(f"Running Ablation Mode: {mode_name}")
        print(f"==================================================")
        
        override = {
            "ablation": ab_cfg,
            "dataset": {"name": "MNIST", "data_dir": "./data", "num_clients": 2, "partition": "iid"},
            "anchors": {"num_anchors": 5, "initialization": "kmeans"},
            "training": {"rounds": 20, "local_epochs": 1},
            "loss": {
                "lambda_attraction": 1.0,
                "lambda_anchor": 1.0,
                "lambda_anchor_repulsion": 1.0,
                "lambda_spread": 1.0,  # Default for spread when enabled
                "anchor_repulsion": {"normalization": "scale", "scale": 0.01}
            },
            "seed": {"value": 42},
            "evaluation": {
                "output_dir": os.path.join(OUTPUT_DIR, mode_name),
                "checkpoint_dir": os.path.join(OUTPUT_DIR, f"{mode_name}_ckpt")
            }
        }
        
        trainer = FederatedAnchorTrainerV2(config_path="fedanchor/configs/mnist_default.yaml", override_config=override)
        res = trainer.train()
        
        hist = res["history"]
        
        for h in hist:
            em = h["eval_metrics"]
            util_list = em.get("anchor_utilization", [0.0]*5)
            util_str = "[" + ", ".join([f"{u:.1f}%" for u in util_list]) + "]"
            
            # Anchor pairwise distances (mean and min)
            pairwise_dists = em.get("anchor_pairwise_distances", [])
            if pairwise_dists:
                mean_dist = np.mean(pairwise_dists)
                min_dist = np.min(pairwise_dists)
            else:
                mean_dist = 0.0
                min_dist = 0.0
                
            nearest_dists = em.get("nearest_anchor_distances", [])
            if nearest_dists:
                mean_nearest = np.mean(nearest_dists)
            else:
                mean_nearest = 0.0
                
            K = 5
            ent = entropy(util_list)
            ent_norm = ent / np.log(K) if K > 1 else 0.0
            
            # How many anchors have > 10% utilization?
            active_anchors = sum(1 for u in util_list if u > 10.0)
            
            results.append({
                "mode": mode_name,
                "round": h["round"],
                "total_loss": h["loss"],
                "attraction_loss": h["attraction_loss"],
                "anchor_loss": h.get("anchor_loss", 0.0),
                "repulsion_scaled": h.get("anchor_repulsion_scaled", 0.0),
                "spread_loss": h.get("anchor_spread_loss", 0.0),
                "knn_accuracy": em["knn_accuracy"],
                "trustworthiness": em["trustworthiness"],
                "continuity": em["continuity"],
                "anchor_utilization": util_str,
                "entropy_norm": ent_norm,
                "active_anchors": active_anchors,
                "mean_pairwise_dist": mean_dist,
                "min_pairwise_dist": min_dist,
                "mean_nearest_dist": mean_nearest
            })
            
    csv_path = os.path.join(OUTPUT_DIR, "v2_ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    print(f"V2 Ablation results saved to {csv_path}")

if __name__ == "__main__":
    run_ablation()
