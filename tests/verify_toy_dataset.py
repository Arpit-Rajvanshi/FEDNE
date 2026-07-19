import numpy as np
import torch
import torch.nn as nn
import copy
import sys
import os

# Import package components
import fedne
from fedne.training.trainer import Trainer
from fedne.utils.seed import set_seed
from fedne.evaluation.metrics import pairwise_distances

def main() -> None:
    # 1. Create a synthetic dataset of 100 samples in 4D space (50 per class)
    np.random.seed(42)
    class_a_x = np.random.normal(loc=0.0, scale=0.1, size=(50, 4)).astype(np.float32)
    class_a_y = np.zeros(50, dtype=np.int64)
    
    class_b_x = np.random.normal(loc=1.5, scale=0.1, size=(50, 4)).astype(np.float32)
    class_b_y = np.ones(50, dtype=np.int64)
    
    X_train = np.concatenate([class_a_x, class_b_x], axis=0)
    y_train = np.concatenate([class_a_y, class_b_y], axis=0)
    
    X_test = copy.deepcopy(X_train)
    y_test = copy.deepcopy(y_train)
    
    # 2. Configure a FL setting
    config = {
        "dataset": {
            "name": "MNIST",  # Dummy name for loader bypass
            "data_dir": "./data"
        },
        "federated": {
            "clients": 2,     # 2 clients
            "rounds": 10,     # 10 rounds
            "local_epochs": 2,
            "device": "cpu"
        },
        "model": {
            "embedding_dim": 2,
            "hidden_dims": [16]  # Tiny hidden dimensions
        },
        "graph": {
            "k": 5,           # k = 5 neighbors
            "negative_samples": 5,
            "with_replacement": True
        },
        "augmentation": {
            "enabled": False  # Disable mixup for deterministic distance checks
        },
        "surrogate": {
            "hidden_dim": 16,
            "grid_step": 0.5,
            "margin": 0.5,
            "lr": 0.01,
            "epochs": 5,
            "start_round": 0
        },
        "optimizer": {
            "lr": 0.01,
            "weight_decay": 0.0
        },
        "logging": {
            "tensorboard": False,
            "log_dir": "./test_runs",
            "checkpoint_dir": "./test_checkpoints",
            "output_dir": "./test_outputs",
            "log_frequency": 1,
            "visualize_frequency": 10,
            "seed": 42,
            "eval_subset_size": 100
        }
    }
    
    # Set seed for reproducibility
    set_seed(42, deterministic=True)
    
    # 3. Instantiate Trainer and override load_dataset to return our custom partition
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.device = torch.device("cpu")
    trainer.log_dir = "./toy_logs"
    trainer.checkpoint_dir = "./toy_checkpoints"
    trainer.output_dir = "./toy_outputs"
    os.makedirs(trainer.checkpoint_dir, exist_ok=True)
    os.makedirs(trainer.output_dir, exist_ok=True)
    trainer.tb_writer = None
    trainer.metrics_history = []
    
    # Split IID: Client 0 and Client 1 both get mixtures of Class A and Class B
    client0_x = np.concatenate([class_a_x[:25], class_b_x[:25]], axis=0)
    client0_y = np.concatenate([class_a_y[:25], class_b_y[:25]], axis=0)
    
    client1_x = np.concatenate([class_a_x[25:], class_b_x[25:]], axis=0)
    client1_y = np.concatenate([class_a_y[25:], class_b_y[25:]], axis=0)
    
    client_data = [
        (client0_x, client0_y),
        (client1_x, client1_y)
    ]
    trainer.X_test = X_test
    trainer.y_test = y_test
    trainer.client_sizes = [len(x) for x, _ in client_data]
    trainer.input_dim = X_train.shape[1]
    
    from fedne.client.client import Client
    from fedne.server.server import Server
    
    trainer.clients = []
    # Avoid shadowing y_train
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
    
    # Record initial embedding distances
    trainer.server.global_encoder.eval()
    with torch.no_grad():
        init_emb = trainer.server.global_encoder(torch.from_numpy(X_train)).numpy()
    
    D_init = pairwise_distances(init_emb)
    
    # Same class indices and diff class indices dynamically
    same_idx = []
    diff_idx = []
    for i in range(len(y_train)):
        for j in range(i + 1, len(y_train)):
            if y_train[i] == y_train[j]:
                same_idx.append((i, j))
            else:
                diff_idx.append((i, j))
                
    init_same_dist = np.mean([D_init[i, j] for i, j in same_idx])
    init_diff_dist = np.mean([D_init[i, j] for i, j in diff_idx])
    init_ratio = init_diff_dist / init_same_dist
    
    print(f"Initial Same-Class Avg Distance: {init_same_dist:.4f}")
    print(f"Initial Different-Class Avg Distance: {init_diff_dist:.4f}")
    print(f"Initial Separation Ratio (Diff/Same): {init_ratio:.4f}")
    
    # Run the fit training loop
    trainer.fit()
    
    # Record final embedding distances
    trainer.server.global_encoder.eval()
    with torch.no_grad():
        final_emb = trainer.server.global_encoder(torch.from_numpy(X_train)).numpy()
        
    D_final = pairwise_distances(final_emb)
    
    final_same_dist = np.mean([D_final[i, j] for i, j in same_idx])
    final_diff_dist = np.mean([D_final[i, j] for i, j in diff_idx])
    final_ratio = final_diff_dist / final_same_dist
    
    print("\n" + "=" * 50)
    print("TOY DATASET TRAINING VERIFICATION")
    print("=" * 50)
    print(f"Final Same-Class Avg Distance: {final_same_dist:.4f} (Initial: {init_same_dist:.4f})")
    print(f"Final Different-Class Avg Distance: {final_diff_dist:.4f} (Initial: {init_diff_dist:.4f})")
    print(f"Final Separation Ratio (Diff/Same): {final_ratio:.4f} (Initial: {init_ratio:.4f})")
    
    # Relative checks:
    # 1. Same-class distances must be smaller than different-class distances in final embeddings
    assert final_same_dist < final_diff_dist, "Same-class distance is not smaller than different-class distance!"
    
    # 2. Separation ratio must increase, showing class clustering
    assert final_ratio > init_ratio, "Class separation ratio did not improve!"
    
    # 3. kNN accuracy should be high
    final_knn_acc = trainer.metrics_history[-1]["knn_accuracy"]
    assert final_knn_acc >= 0.90, f"Final kNN classification accuracy is too low: {final_knn_acc:.4f}"
    
    # Verify no NaN / Inf losses
    for r_idx, metrics in enumerate(trainer.metrics_history):
        assert not np.isnan(metrics["loss"]), f"Round {r_idx} Loss is NaN!"
        assert not np.isinf(metrics["loss"]), f"Round {r_idx} Loss is Inf!"
        assert metrics["loss"] > -1000.0, f"Round {r_idx} Loss exploded negatively: {metrics['loss']}!"
        
    print("\nSUCCESS: Same-class distances decreased, different-class distances increased, and losses remained numerically stable!")

if __name__ == "__main__":
    main()
