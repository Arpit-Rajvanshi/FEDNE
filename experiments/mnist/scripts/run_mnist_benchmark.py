import os
import copy
import yaml
import json
from typing import Dict, Any
from fedne.training.trainer import Trainer
from fedne.utils.seed import set_seed

def run_experiment(name: str, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Run a single benchmark experiment and return its final metrics.
    """
    print("=" * 60)
    print(f"RUNNING EXPERIMENT: {name}")
    print("=" * 60)
    
    # Enforce reproducibility
    set_seed(seed=config.get("logging", {}).get("seed", 42), deterministic=True)
    
    trainer = Trainer(config)
    trainer.fit()
    
    # Retrieve final round's metrics
    final_metrics = trainer.metrics_history[-1]
    return {
        "trustworthiness": final_metrics["trustworthiness"],
        "continuity": final_metrics["continuity"],
        "knn_accuracy": final_metrics["knn_accuracy"],
        "steadiness": final_metrics["steadiness"],
        "cohesiveness": final_metrics["cohesiveness"]
    }

def main() -> None:
    # Base configuration template
    base_config = {
        "dataset": {
            "name": "MNIST",
            "data_dir": "./data"
        },
        "federated": {
            "clients": 20,
            "rounds": 50,             # Reduced from 200 for faster benchmark run, configurable
            "local_epochs": 1,
            "device": "cuda"
        },
        "model": {
            "embedding_dim": 2,
            "hidden_dims": [512, 256, 128]
        },
        "graph": {
            "k": 7,                  # Fix k=7 as in metrics evaluation
            "negative_samples": 5,
            "with_replacement": True
        },
        "augmentation": {
            "enabled": True,
            "alpha": 0.2,
            "mixup_ratio": 1.0
        },
        "surrogate": {
            "hidden_dim": 64,
            "grid_step": 0.3,
            "margin": 0.5,
            "lr": 0.001,
            "epochs": 5,
            "start_round": 0
        },
        "optimizer": {
            "lr": 0.001,
            "weight_decay": 0.0,
            "lr_decay_rounds": [15, 30],  # Adjusted for 50 rounds
            "lr_decay_factor": 0.1
        },
        "logging": {
            "tensorboard": True,
            "log_dir": "./runs",
            "checkpoint_dir": "./checkpoints",
            "output_dir": "./outputs",
            "log_frequency": 1,
            "visualize_frequency": 10,
            "seed": 42,
            "eval_subset_size": 2000
        }
    }
    
    # Define experiment configurations
    experiments = {
        "IID": {
            "partition": {"type": "iid"}
        },
        "Dirichlet(0.1)": {
            "partition": {"type": "dirichlet", "alpha": 0.1}
        },
        "Dirichlet(0.5)": {
            "partition": {"type": "dirichlet", "alpha": 0.5}
        },
        "Shards(C=2)": {
            "partition": {"type": "shards", "shards_per_client": 2}
        },
        "Shards(C=3)": {
            "partition": {"type": "shards", "shards_per_client": 3}
        }
    }
    
    results = {}
    
    for exp_name, exp_diff in experiments.items():
        # Build configuration for this run
        config = copy.deepcopy(base_config)
        config["partition"] = exp_diff["partition"]
        
        # Point logs and outputs to experiment-specific subfolders
        safe_name = exp_name.replace("(", "_").replace(")", "_").replace("=", "_")
        config["logging"]["log_dir"] = f"./experiments/mnist/results/logs/{safe_name}"
        config["logging"]["checkpoint_dir"] = f"./experiments/mnist/results/checkpoints/{safe_name}"
        config["logging"]["output_dir"] = f"./experiments/mnist/results/outputs/{safe_name}"
        
        metrics = run_experiment(exp_name, config)
        results[exp_name] = metrics
        
    # Write consolidated markdown comparison table
    results_dir = "./experiments/mnist/results"
    os.makedirs(results_dir, exist_ok=True)
    
    table_path = os.path.join(results_dir, "benchmark_table.md")
    
    markdown_content = "# MNIST Partitioning Benchmark Results (FEDNE Reproduction)\n\n"
    markdown_content += "| Partition Strategy | Trustworthiness | Continuity | kNN Accuracy | Steadiness | Cohesiveness |\n"
    markdown_content += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    
    for exp_name, metrics in results.items():
        markdown_content += (
            f"| **{exp_name}** | "
            f"{metrics['trustworthiness']:.4f} | "
            f"{metrics['continuity']:.4f} | "
            f"{metrics['knn_accuracy']:.4f} | "
            f"{metrics['steadiness']:.4f} | "
            f"{metrics['cohesiveness']:.4f} |\n"
        )
        
    with open(table_path, "w") as f:
        f.write(markdown_content)
        
    print("\nBenchmark experiments completed!")
    print(f"Results table written to: {table_path}")
    print(markdown_content)

if __name__ == "__main__":
    main()
