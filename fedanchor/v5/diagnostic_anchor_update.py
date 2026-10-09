import os
import sys
import torch
import numpy as np

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fedanchor.v5.training.trainer_v5 import AnchorTrainerV5

def run_diagnostic():
    print("="*60)
    print("FEDANCHOR V5 - ANCHOR UPDATE DIAGNOSTIC")
    print("="*60)
    
    base_config = {
        "dataset": {
            "name": "MNIST",
            "num_clients": 2,
            "max_samples_per_client": 1000 # Small dataset for fast diagnostic
        },
        "training": {
            "rounds": 1,
            "local_epochs": 1,
            "learning_rate": 0.001,
            "batch_size": 256,
            "persistent_optimizer": False
        },
        "loss": {
            "lambda_attraction": 1.0,
            "lambda_anchor": 1.0,
            "lambda_anchor_repulsion": 1.0,
            "lambda_spread": 1.0,
            "lambda_balance": 1.0,
            "lambda_sep": 0.1,
            "margin": 0.1,
            "eps": 1e-7
        }
    }
    
    print("\n--- Diagnostic 1: Full-Batch (V4 emulation) ---")
    cfg_full = dict(base_config)
    cfg_full["training"]["batch_size"] = 30000
    trainer_full = AnchorTrainerV5(config_path=r"D:\FEDNE\fedanchor\configs\mnist_default.yaml", override_config=cfg_full)
    res_full = trainer_full.train()
    
    print("\n--- Diagnostic 2: Mini-Batch + Reset (V5-A) ---")
    cfg_mini = dict(base_config)
    cfg_mini["training"]["batch_size"] = 256
    trainer_mini = AnchorTrainerV5(config_path=r"D:\FEDNE\fedanchor\configs\mnist_default.yaml", override_config=cfg_mini)
    res_mini = trainer_mini.train()
    
    print("\n--- Diagnostic 3: Mini-Batch + Persistent (V5-B) ---")
    cfg_pers = dict(base_config)
    cfg_pers["training"]["batch_size"] = 256
    cfg_pers["training"]["persistent_optimizer"] = True
    trainer_pers = AnchorTrainerV5(config_path=r"D:\FEDNE\fedanchor\configs\mnist_default.yaml", override_config=cfg_pers)
    res_pers = trainer_pers.train()

if __name__ == "__main__":
    run_diagnostic()
