import os
import sys
import json
import csv
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fedanchor.v4.training.trainer_v4 import AnchorTrainerV4

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs/controlled_ablation_v4"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_ablation():
    ablation_modes = {
        "A_V2_Baseline": {
            "use_attraction": True, 
            "use_anchor": True, 
            "use_anchor_repulsion": True,
            "use_spread": True,
            "anchor_assignment_type": "soft",
            "lambda_balance": 0.0,
            "lambda_sep": 0.0
        },
        "B_V3_Balance": {
            "use_attraction": True, 
            "use_anchor": True, 
            "use_anchor_repulsion": True,
            "use_spread": True,
            "anchor_assignment_type": "soft",
            "lambda_balance": 0.1,
            "lambda_sep": 0.0
        },
        "C_V4_Sep_Low": {
            "use_attraction": True, 
            "use_anchor": True, 
            "use_anchor_repulsion": True,
            "use_spread": True,
            "anchor_assignment_type": "soft",
            "lambda_balance": 0.1,
            "lambda_sep": 0.001
        },
        "D_V4_Sep_Med": {
            "use_attraction": True, 
            "use_anchor": True, 
            "use_anchor_repulsion": True,
            "use_spread": True,
            "anchor_assignment_type": "soft",
            "lambda_balance": 0.1,
            "lambda_sep": 0.01
        },
        "E_V4_Sep_High": {
            "use_attraction": True, 
            "use_anchor": True, 
            "use_anchor_repulsion": True,
            "use_spread": True,
            "anchor_assignment_type": "soft",
            "lambda_balance": 0.1,
            "lambda_sep": 0.1
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
                "lambda_sep": ab_cfg["lambda_sep"],
                "margin": 0.10,
                "anchor_repulsion": {"normalization": "scale", "scale": 0.01}
            },
            "seed": {"value": 42},
            "evaluation": {
                "output_dir": os.path.join(OUTPUT_DIR, mode_name),
                "checkpoint_dir": os.path.join(OUTPUT_DIR, f"{mode_name}_ckpt")
            }
        }
        
        override["loss"].update({
            "use_attraction": ab_cfg["use_attraction"],
            "use_anchor": ab_cfg["use_anchor"],
            "use_anchor_repulsion": ab_cfg["use_anchor_repulsion"],
            "use_spread": ab_cfg["use_spread"],
            "anchor_assignment_type": ab_cfg["anchor_assignment_type"]
        })
        
        trainer = AnchorTrainerV4(config_path="fedanchor/configs/mnist_default.yaml", override_config=override)
        res = trainer.train()
        
        hist = res["history"]
        
        for h in hist:
            r = h["round"]
            if r not in [1, 5, 10, 15, 20]:
                continue
                
            em = h["eval_metrics"]
            
            soft_u = em.get("soft_utilization_frac", [0.0]*5)
            hard_u = em.get("hard_utilization_frac", [0.0]*5)
            
            # Anchor pairwise distances (mean and min)
            results.append({
                "mode": mode_name,
                "round": r,
                "total_loss": h["loss"],
                "attraction_loss": h["attraction_loss"],
                "soft_anchor_loss": h.get("anchor_loss", 0.0),
                "repulsion_loss": h.get("anchor_repulsion_scaled", 0.0),
                "separation_loss": h.get("separation_loss", 0.0),
                "balance_loss": h.get("balance_loss", 0.0),
                "regularization_loss": h.get("embedding_reg_loss", 0.0),
                "knn_accuracy": em["knn_accuracy"],
                "trustworthiness": em["trustworthiness"],
                "continuity": em["continuity"],
                "soft_utilization_a1": soft_u[0] if len(soft_u) > 0 else 0.0,
                "soft_utilization_a2": soft_u[1] if len(soft_u) > 1 else 0.0,
                "soft_utilization_a3": soft_u[2] if len(soft_u) > 2 else 0.0,
                "soft_utilization_a4": soft_u[3] if len(soft_u) > 3 else 0.0,
                "soft_utilization_a5": soft_u[4] if len(soft_u) > 4 else 0.0,
                "hard_utilization_a1": hard_u[0] if len(hard_u) > 0 else 0.0,
                "hard_utilization_a2": hard_u[1] if len(hard_u) > 1 else 0.0,
                "hard_utilization_a3": hard_u[2] if len(hard_u) > 2 else 0.0,
                "hard_utilization_a4": hard_u[3] if len(hard_u) > 3 else 0.0,
                "hard_utilization_a5": hard_u[4] if len(hard_u) > 4 else 0.0,
                "normalized_entropy": em.get("normalized_entropy", 0.0),
                "effective_anchors": em.get("effective_anchors", 0),
                "min_pairwise_anchor_distance": em.get("min_pairwise_anchor_distance", 0.0),
                "mean_pairwise_anchor_distance": em.get("mean_pairwise_anchor_distance", 0.0),
                "max_pairwise_anchor_distance": em.get("max_pairwise_anchor_distance", 0.0),
                "mean_nearest_anchor_distance": em.get("anchor_coverage", 0.0),
                "max_nearest_anchor_distance": em.get("anchor_max_distance", 0.0),
                "collapse_state": em.get("collapse_state", "UNKNOWN")
            })
            
    csv_path = os.path.join(OUTPUT_DIR, "v4_ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    print(f"V4 Ablation results saved to {csv_path}")

if __name__ == "__main__":
    run_ablation()
