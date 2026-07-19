import fedne
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import copy
import sys
import os
import matplotlib.pyplot as plt
from fedne.training.trainer import Trainer
from fedne.utils.seed import set_seed
from fedne.datasets.loader import partition_data

def main() -> None:
    print("=" * 60)
    # 1. Load MNIST and take 1000 samples (500 samples per client)
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Bypassing raw directory issues, download if needed
    os.makedirs("./data", exist_ok=True)
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    X = mnist_train.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y = mnist_train.targets.numpy().astype(np.int64)
    
    # Subset to 1000 samples for the benchmark
    X_subset = X[:1000]
    y_subset = y[:1000]
    
    # 2. Configure a 2-client, 10-round MNIST FL benchmark
    config = {
        "dataset": {
            "name": "MNIST",
            "data_dir": "./data"
        },
        "federated": {
            "clients": 2,     # 2 clients
            "rounds": 10,     # 10 rounds
            "local_epochs": 1,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "model": {
            "embedding_dim": 2,
            "hidden_dims": [128, 64]  # Compressed hidden dimensions for speed
        },
        "graph": {
            "k": 7,           # k = 7 neighbors
            "negative_samples": 5,
            "with_replacement": True
        },
        "augmentation": {
            "enabled": True,
            "alpha": 0.2,
            "mixup_ratio": 1.0
        },
        "surrogate": {
            "hidden_dim": 32,
            "grid_step": 0.5,
            "margin": 0.5,
            "lr": 0.01,
            "epochs": 5,
            "start_round": 0
        },
        "optimizer": {
            "lr": 0.005,
            "weight_decay": 1e-4
        },
        "logging": {
            "tensorboard": False,
            "log_dir": "./experiments/mnist/benchmark/logs",
            "checkpoint_dir": "./experiments/mnist/benchmark/checkpoints",
            "output_dir": "./experiments/mnist/benchmark/outputs",
            "log_frequency": 1,
            "visualize_frequency": 10,
            "seed": 42,
            "eval_subset_size": 200
        }
    }
    
    # Set seed for reproducibility
    set_seed(42, deterministic=True)
    
    # 3. Instantiate Trainer and split client data
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.device = torch.device(config["federated"]["device"])
    trainer.log_dir = config["logging"]["log_dir"]
    trainer.checkpoint_dir = config["logging"]["checkpoint_dir"]
    trainer.output_dir = config["logging"]["output_dir"]
    os.makedirs(trainer.checkpoint_dir, exist_ok=True)
    os.makedirs(trainer.output_dir, exist_ok=True)
    trainer.tb_writer = None
    trainer.metrics_history = []
    
    # Split IID: 500 samples per client
    client_data = partition_data(X_subset, y_subset, partition_type="iid", num_clients=2)
    
    # Global test set = the same 1000 sample subset
    trainer.X_test = X_subset
    trainer.y_test = y_subset
    trainer.client_sizes = [len(x) for x, _ in client_data]
    trainer.input_dim = X_subset.shape[1]
    
    from fedne.client.client import Client
    from fedne.server.server import Server
    
    trainer.clients = []
    for i, (c_x, c_y) in enumerate(client_data):
        client = Client(
            client_id=i,
            X_train=c_x,
            y_train=c_y,
            client_sizes=trainer.client_sizes,
            config=config,
            device=trainer.device
        )
        trainer.clients.append(client)
        
    trainer.server = Server(
        config=config,
        input_dim=trainer.input_dim,
        client_sizes=trainer.client_sizes,
        device=trainer.device
    )
    trainer.best_knn_acc = 0.0
    trainer.start_round = 0
    
    # 4. Run the FEDNE fit training loop
    trainer.fit()
    
    # 5. Extract metrics history for plotting
    rounds = range(len(trainer.metrics_history))
    losses = [m["loss"] for m in trainer.metrics_history]
    attractions = [m["attraction_loss"] for m in trainer.metrics_history]
    local_reps = [m["local_repulsion"] for m in trainer.metrics_history]
    surr_reps = [m["surrogate_repulsion"] for m in trainer.metrics_history]
    surr_mses = [m["surrogate_loss"] for m in trainer.metrics_history]
    
    trusts = [m["trustworthiness"] for m in trainer.metrics_history]
    conts = [m["continuity"] for m in trainer.metrics_history]
    knn_accs = [m["knn_accuracy"] for m in trainer.metrics_history]
    
    # 6. Plotting and saving validation figures
    os.makedirs("./experiments/mnist/benchmark/plots", exist_ok=True)
    
    # Plot 1: Contrastive Loss Breakdown
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, losses, label="Total Loss", marker="o", color="black", linewidth=2)
    plt.plot(rounds, attractions, label="Attraction Loss", marker="s", color="blue")
    plt.plot(rounds, local_reps, label="Local Repulsion", marker="^", color="green")
    plt.plot(rounds, surr_reps, label="Surrogate Repulsion", marker="d", color="red")
    plt.xlabel("Communication Round")
    plt.ylabel("Loss value")
    plt.title("FEDNE Loss Component Breakdown (MNIST 2-Client Benchmark)")
    plt.grid(True)
    plt.legend()
    plt.savefig("./experiments/mnist/benchmark/plots/loss_breakdown.png", dpi=150)
    plt.close()
    
    # Plot 2: Low-Dimensional Embedding Quality Metrics
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, trusts, label="Trustworthiness", marker="o", color="blue")
    plt.plot(rounds, conts, label="Continuity", marker="s", color="green")
    plt.plot(rounds, knn_accs, label="kNN Classification Accuracy", marker="^", color="orange")
    plt.xlabel("Communication Round")
    plt.ylabel("Metric value")
    plt.title("FEDNE Embedding Quality Metrics (MNIST 2-Client Benchmark)")
    plt.grid(True)
    plt.legend()
    plt.savefig("./experiments/mnist/benchmark/plots/quality_metrics.png", dpi=150)
    plt.close()
    
    # Plot 3: Surrogate Training MSE Loss
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, surr_mses, label="Surrogate MSE Loss", marker="o", color="purple")
    plt.xlabel("Communication Round")
    plt.ylabel("MSE Loss")
    plt.title("FEDNE Surrogate Model Approximation MSE (MNIST 2-Client Benchmark)")
    plt.grid(True)
    plt.legend()
    plt.savefig("./experiments/mnist/benchmark/plots/surrogate_mse.png", dpi=150)
    plt.close()
    
    print("\n" + "=" * 60)
    print("MNIST BENCHMARK VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Final Trustworthiness: {trusts[-1]:.4f}")
    print(f"Final Continuity: {conts[-1]:.4f}")
    print(f"Final kNN Accuracy: {knn_accs[-1]:.4f}")
    print(f"Final Surrogate MSE: {surr_mses[-1]:.4e}")
    print("Benchmark plots saved in: ./experiments/mnist/benchmark/plots/")
    
    # Copy plots to the brain artifact directory so the user can easily view them
    artifact_plots_dir = "C:/Users/arpit/.gemini/antigravity-ide/brain/e3034730-7976-4759-86d5-b4ceab9b6a1f/plots"
    os.makedirs(artifact_plots_dir, exist_ok=True)
    import shutil
    shutil.copy("./experiments/mnist/benchmark/plots/loss_breakdown.png", os.path.join(artifact_plots_dir, "loss_breakdown.png"))
    shutil.copy("./experiments/mnist/benchmark/plots/quality_metrics.png", os.path.join(artifact_plots_dir, "quality_metrics.png"))
    shutil.copy("./experiments/mnist/benchmark/plots/surrogate_mse.png", os.path.join(artifact_plots_dir, "surrogate_mse.png"))
    print(f"Artifact plots saved to: {artifact_plots_dir}")

if __name__ == "__main__":
    main()
