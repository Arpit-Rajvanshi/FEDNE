import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fedanchor.v3.training.trainer_v3 import AnchorTrainerV3

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs/controlled_ablation_v3"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_ablation():
    ablation_modes = {
        "A_V1_Baseline": {
            "use_attraction": True, 
            "use_anchor": True, 
            "use_anchor_repulsion": True,
            "use_spread": False,
            "anchor_assignment_type": "hard",
            "lambda_balance": 0.0
        },
        "B_V2_Soft_Spread": {
            "use_attraction": True, 
            "use_anchor": True, 
            "use_anchor_repulsion": True,
            "use_spread": True,
            "anchor_assignment_type": "soft",
            "lambda_balance": 0.0
        },
        "C_V3_Balance_0.001": {
            "use_attraction": True, 
            "use_anchor": True, 
            "use_anchor_repulsion": True,
            "use_spread": True,
            "anchor_assignment_type": "soft",
            "lambda_balance": 0.001
        },
        "D_V3_Balance_0.01": {
            "use_attraction": True, 
            "use_anchor": True, 
            "use_anchor_repulsion": True,
            "use_spread": True,
            "anchor_assignment_type": "soft",
            "lambda_balance": 0.01
        },
        "E_V3_Balance_0.1": {
            "use_attraction": True, 
            "use_anchor": True, 
            "use_anchor_repulsion": True,
            "use_spread": True,
            "anchor_assignment_type": "soft",
            "lambda_balance": 0.1
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
            "anchors": {"num_anchors": 5, "initialization": "kmeans", "num_anchors_per_client": 5},
            "training": {"rounds": 20, "local_epochs": 1},
            "loss": {
                "lambda_attraction": 1.0,
                "lambda_anchor": 1.0,
                "lambda_anchor_repulsion": 1.0,
                "lambda_spread": 1.0,  
                "lambda_balance": ab_cfg["lambda_balance"],
                "anchor_repulsion": {"normalization": "scale", "scale": 0.01}
            },
            "seed": {"value": 42},
            "evaluation": {
                "output_dir": os.path.join(OUTPUT_DIR, mode_name),
                "checkpoint_dir": os.path.join(OUTPUT_DIR, f"{mode_name}_ckpt")
            }
        }
        
        # Merge ablation flags logically handled in trainer_v3
        # V3 implementation natively reads these flags via loss config
        override["loss"].update({
            "use_attraction": ab_cfg["use_attraction"],
            "use_anchor": ab_cfg["use_anchor"],
            "use_anchor_repulsion": ab_cfg["use_anchor_repulsion"],
            "use_spread": ab_cfg["use_spread"],
            "anchor_assignment_type": ab_cfg["anchor_assignment_type"]
        })
        
        trainer = AnchorTrainerV3(config_path="fedanchor/configs/mnist_default.yaml", override_config=override)
        res = trainer.train()
        
        hist = res["history"]
        
        for h in hist:
            # We only record specific rounds as requested
            r = h["round"]
            if r not in [1, 5, 10, 15, 20]:
                continue
                
            em = h["eval_metrics"]
            util_list = em.get("anchor_utilization", [0.0]*5)
            util_str = "[" + ", ".join([f"{u:.1f}%" for u in util_list]) + "]"
            
            # Anchor pairwise distances (mean and min)
            pairwise_dists = em.get("anchor_pairwise_distances", [])
            if pairwise_dists:
                mean_dist = np.mean(pairwise_dists)
                min_dist = np.min(pairwise_dists)
                max_dist = np.max(pairwise_dists)
            else:
                mean_dist = 0.0
                min_dist = 0.0
                max_dist = 0.0
                
            nearest_dists = em.get("nearest_anchor_distances", [])
            if nearest_dists:
                mean_nearest = np.mean(nearest_dists)
                max_nearest = np.max(nearest_dists)
            else:
                mean_nearest = 0.0
                max_nearest = 0.0
            
            results.append({
                "mode": mode_name,
                "round": r,
                "total_loss": h["loss"],
                "attraction_loss": h["attraction_loss"],
                "anchor_loss": h.get("anchor_loss", 0.0),
                "repulsion_scaled": h.get("anchor_repulsion_scaled", 0.0),
                "spread_loss": h.get("anchor_spread_loss", 0.0),
                "balance_loss": h.get("balance_loss", 0.0),
                "knn_accuracy": em["knn_accuracy"],
                "trustworthiness": em["trustworthiness"],
                "continuity": em["continuity"],
                "anchor_utilization": util_str,
                "entropy_norm": em.get("normalized_entropy", 0.0),
                "max_utilization": em.get("max_utilization", 0.0),
                "min_utilization": em.get("min_utilization", 0.0),
                "utilization_imbalance": em.get("utilization_imbalance", 0.0),
                "effective_anchors": em.get("effective_anchors", 0),
                "collapse_state": em.get("collapse_state", "UNKNOWN"),
                "mean_pairwise_dist": mean_dist,
                "min_pairwise_dist": min_dist,
                "max_pairwise_dist": max_dist,
                "mean_nearest_dist": mean_nearest,
                "max_nearest_dist": max_nearest
            })
            
    csv_path = os.path.join(OUTPUT_DIR, "v3_ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    print(f"V3 Ablation results saved to {csv_path}")

if __name__ == "__main__":
    run_ablation()
