import os
import sys
import yaml
import csv

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fedanchor.v5.training.trainer_v5 import AnchorTrainerV5

def run_ablation():
    print("="*80)
    print("FEDANCHOR V5 - OPTIMIZATION PROTOCOL ABLATION")
    print("="*80)
    
    out_dir = os.path.join(os.path.dirname(__file__), "outputs", "controlled_ablation_v5")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "v5_ablation_results.csv")
    
    # Modes exactly as specified by user
    # A_V4_Baseline: Full batch + reset
    # B_V5_MiniBatch_Reset: Mini-batch + reset
    # C_V5_MiniBatch_Persistent: Mini-batch + persistent
    modes = [
        {"name": "A_V4_Baseline", "batch_size": 30000, "persistent": False},
        {"name": "B_V5_MiniBatch_Reset", "batch_size": 1280, "persistent": False}, # 256 samples * 5 edges
        {"name": "C_V5_MiniBatch_Persistent", "batch_size": 1280, "persistent": True}
    ]
    
    results = []
    
    for mode in modes:
        mode_name = mode["name"]
        print(f"\n\n{'*'*50}")
        print(f"RUNNING MODE: {mode_name}")
        print(f"{'*'*50}\n")
        
        cfg = {
            "dataset": {
                "name": "MNIST",
                "num_clients": 2,
                "partition": "iid",
                "alpha": 0.5,
                "max_samples_per_client": None # Use full 30,000 per client
            },
            "training": {
                "rounds": 20,
                "local_epochs": 1,
                "learning_rate": 0.001,
                "batch_size": mode["batch_size"],
                "persistent_optimizer": mode["persistent"]
            },
            "loss": {
                "lambda_attraction": 1.0,
                "lambda_anchor": 1.0,
                "lambda_anchor_repulsion": 1.0,
                "lambda_spread": 1.0,
                "lambda_balance": 1.0,
                "lambda_sep": 0.1,
                "margin": 0.1,
                "eps": 1e-7,
                "balancing": {"enabled": False}
            },
            "ablation": {
                "use_attraction": True,
                "use_anchor": True,
                "use_anchor_repulsion": True,
                "use_spread": True,
                "anchor_assignment_type": "soft"
            },
            "evaluation": {
                "output_dir": os.path.join(out_dir, mode_name)
            }
        }
        
        trainer = AnchorTrainerV5(config_path=r"D:\FEDNE\fedanchor\configs\mnist_default.yaml", override_config=cfg)
        res = trainer.train()
        
        for r_info in res["history"]:
            eval_m = r_info["eval_metrics"]
            row = {
                "mode": mode_name,
                "round": r_info["round"],
                "loss": r_info["loss"],
                "attraction_loss": r_info["attraction_loss"],
                "anchor_loss": r_info["anchor_loss"],
                "anchor_repulsion_scaled": r_info["anchor_repulsion_scaled"],
                "anchor_spread_loss": r_info["anchor_spread_loss"],
                "separation_loss": r_info["separation_loss"],
                "balance_loss": r_info["balance_loss"],
                "knn_accuracy": eval_m["knn_accuracy"],
                "trustworthiness": eval_m["trustworthiness"],
                "continuity": eval_m["continuity"],
                "soft_util_0": eval_m["soft_utilization_frac"][0],
                "soft_util_1": eval_m["soft_utilization_frac"][1],
                "soft_util_2": eval_m["soft_utilization_frac"][2],
                "soft_util_3": eval_m["soft_utilization_frac"][3],
                "soft_util_4": eval_m["soft_utilization_frac"][4],
                "hard_util_max": max(eval_m["hard_utilization_frac"]),
                "normalized_entropy": eval_m["normalized_entropy"],
                "effective_anchors": eval_m["effective_anchors"],
                "min_pairwise_anchor_distance": eval_m["min_pairwise_anchor_distance"],
                "mean_pairwise_anchor_distance": eval_m["mean_pairwise_anchor_distance"],
                "max_pairwise_anchor_distance": eval_m["max_pairwise_anchor_distance"],
                "mean_nearest_anchor_distance": sum(eval_m.get("nearest_anchor_distances", [0]))/max(1, len(eval_m.get("nearest_anchor_distances", [0]))),
                "max_nearest_anchor_distance": max(eval_m.get("nearest_anchor_distances", [0]) if eval_m.get("nearest_anchor_distances") else [0]),
                "collapse_state": eval_m["collapse_state"],
                "optimizer_steps_per_round": r_info.get("optimizer_steps_per_round", 0),
                "anchor_gradient_norm": r_info.get("anchor_gradient_norm", 0.0),
                "anchor_displacement": r_info.get("anchor_displacement", 0.0),
                "fraction_nonzero_update": r_info.get("fraction_nonzero_update", 0.0)
            }
            results.append(row)
            
    # Write CSV
    if len(results) > 0:
        keys = list(results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
            
    print(f"\n\nV5 Ablation results saved to {csv_path}")

if __name__ == "__main__":
    run_ablation()
