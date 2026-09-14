import os
import sys
import yaml
import numpy as np
import torch

if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits = lambda maxdigits: None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedanchor.training.trainer import FederatedAnchorTrainer

def run_anchor_initialization_experiments():
    print("\n==================================================")
    print("  EXPERIMENT A: ANCHOR INITIALIZATION COMPARISON  ")
    print("==================================================")
    
    strategies = ["kmeans", "farthest_point", "representative_points"]
    results_summary = {}
    
    for strat in strategies:
        print(f"\n---> Running Strategy: {strat.upper()} <---")
        override = {
            "anchors": {"initialization": strat},
            "training": {"rounds": 2},
            "loss": {
                "lambda_attraction": 1.0,
                "lambda_anchor": 1.0,
                "lambda_anchor_repulsion": 1.0,
                "anchor_repulsion": {"normalization": "scale", "scale": 0.01}
            },
            "evaluation": {
                "output_dir": f"./fedanchor/outputs/exp_init_{strat}",
                "checkpoint_dir": f"./fedanchor/checkpoints/exp_init_{strat}"
            }
        }
        
        trainer = FederatedAnchorTrainer(config_path="fedanchor/configs/mnist_default.yaml", override_config=override)
        res = trainer.train()
        
        hist = res["history"]
        r1_anc_cov = hist[0]["eval_metrics"]["anchor_coverage"]
        rf_anc_cov = hist[-1]["eval_metrics"]["anchor_coverage"]
        final_eval = res["final_metrics"]
        
        results_summary[strat] = {
            "initial_anchor_coverage": r1_anc_cov,
            "final_anchor_coverage": rf_anc_cov,
            "trustworthiness": final_eval["trustworthiness"],
            "continuity": final_eval["continuity"],
            "knn_accuracy": final_eval["knn_accuracy"]
        }
        
    print("\n--- ANCHOR INITIALIZATION COMPARISON SUMMARY ---")
    print(f"{'Strategy':<22} | {'Init Cov':<8} | {'Final Cov':<9} | {'Trust':<6} | {'Cont':<6} | {'kNN Acc':<7}")
    print("-" * 75)
    for strat, m in results_summary.items():
        print(f"{strat:<22} | {m['initial_anchor_coverage']:<8.4f} | {m['final_anchor_coverage']:<9.4f} | {m['trustworthiness']:<6.4f} | {m['continuity']:<6.4f} | {m['knn_accuracy']:<7.4f}")
    return results_summary

def run_ablation_experiments():
    print("\n==================================================")
    print("       EXPERIMENT B: ABLATION STUDY               ")
    print("==================================================")
    
    ablation_modes = {
        "A_attraction_only": {"use_attraction": True, "use_anchor": False, "use_anchor_repulsion": False},
        "B_attraction_anchor": {"use_attraction": True, "use_anchor": True, "use_anchor_repulsion": False},
        "C_attraction_repulsion": {"use_attraction": True, "use_anchor": False, "use_anchor_repulsion": True},
        "D_full_fedanchor": {"use_attraction": True, "use_anchor": True, "use_anchor_repulsion": True}
    }
    
    results_summary = {}
    
    for mode_name, ab_cfg in ablation_modes.items():
        print(f"\n---> Running Ablation Mode: {mode_name} <---")
        override = {
            "ablation": ab_cfg,
            "anchors": {"initialization": "kmeans"},
            "training": {"rounds": 2},
            "loss": {
                "lambda_attraction": 1.0,
                "lambda_anchor": 1.0,
                "lambda_anchor_repulsion": 1.0,
                "anchor_repulsion": {"normalization": "scale", "scale": 0.01}
            },
            "evaluation": {
                "output_dir": f"./fedanchor/outputs/exp_ablation_{mode_name}",
                "checkpoint_dir": f"./fedanchor/checkpoints/exp_ablation_{mode_name}"
            }
        }
        
        trainer = FederatedAnchorTrainer(config_path="fedanchor/configs/mnist_default.yaml", override_config=override)
        res = trainer.train()
        
        hist = res["history"]
        last_hist = hist[-1]
        final_eval = res["final_metrics"]
        
        results_summary[mode_name] = {
            "total_loss": last_hist["loss"],
            "attraction_loss": last_hist["attraction_loss"],
            "anchor_loss": last_hist["anchor_loss"],
            "repulsion_scaled": last_hist["anchor_repulsion_scaled"],
            "trustworthiness": final_eval["trustworthiness"],
            "continuity": final_eval["continuity"],
            "knn_accuracy": final_eval["knn_accuracy"],
            "anchor_coverage": final_eval["anchor_coverage"]
        }
        
    print("\n--- ABLATION STUDY SUMMARY ---")
    print(f"{'Mode':<24} | {'Loss':<6} | {'L_att':<6} | {'L_anc':<6} | {'L_rep':<6} | {'Trust':<6} | {'Cont':<6} | {'kNN Acc':<7}")
    print("-" * 88)
    for mode_name, m in results_summary.items():
        print(f"{mode_name:<24} | {m['total_loss']:<6.4f} | {m['attraction_loss']:<6.4f} | {m['anchor_loss']:<6.4f} | {m['repulsion_scaled']:<6.4f} | {m['trustworthiness']:<6.4f} | {m['continuity']:<6.4f} | {m['knn_accuracy']:<7.4f}")
    return results_summary

def run_extended_20_round_experiment():
    print("\n==================================================")
    print("  EXPERIMENT C: EXTENDED 20-ROUND FEDERATED RUN   ")
    print("==================================================")
    
    override = {
        "anchors": {"initialization": "kmeans"},
        "training": {"rounds": 20},
        "loss": {
            "lambda_attraction": 1.0,
            "lambda_anchor": 1.0,
            "lambda_anchor_repulsion": 1.0,
            "anchor_repulsion": {"normalization": "scale", "scale": 0.01}
        },
        "evaluation": {
            "output_dir": "./fedanchor/outputs/exp_20_rounds",
            "checkpoint_dir": "./fedanchor/checkpoints/exp_20_rounds"
        }
    }
    
    trainer = FederatedAnchorTrainer(config_path="fedanchor/configs/mnist_default.yaml", override_config=override)
    res = trainer.train()
    
    print(f"\n20-Round Experiment Complete! Final kNN Accuracy: {res['final_metrics']['knn_accuracy']:.4f}")
    return res

def main():
    print("==================================================")
    print("      FEDANCHOR PHASE 2 EXPERIMENT SUITE          ")
    print("==================================================")
    
    init_res = run_anchor_initialization_experiments()
    ablation_res = run_ablation_experiments()
    ext_res = run_extended_20_round_experiment()
    
    print("\n==================================================")
    print("    ALL PHASE 2 EXPERIMENTS EXECUTED SUCCESSFULLY ")
    print("==================================================")

if __name__ == "__main__":
    main()
