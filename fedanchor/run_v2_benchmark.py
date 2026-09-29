"""
FEDNE vs FEDANCHOR v2 benchmark (controlled protocol, same evaluator as run_controlled_benchmark.py).

Usage (from repo root):
  python fedanchor/run_v2_benchmark.py <method> <rounds> <partition> ['<json override>']
    method    : fedne | anchor | norep     (norep = FEDANCHOR v2 with lambda_anchor_repulsion = 0)
    partition : iid | dirichlet0.1          (dirichlet0.1 uses one fixed, cached split for every method)
  e.g.  python fedanchor/run_v2_benchmark.py anchor 20 dirichlet0.1 '{"loss":{"anchor_repulsion":{"estimator":"sampled"}}}'
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
torch.set_num_threads(int(os.environ.get("NT", "2")))

import fedanchor.run_controlled_benchmark as rcb
import fedne.datasets.loader as fedne_loader
import fedanchor.datasets.loader as anchor_loader

method, R, part = sys.argv[1], int(sys.argv[2]), sys.argv[3]
extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}


def fixed_dirichlet_split(alpha=0.1, num_clients=2, path="data/partition_dir0.1_2c.npz"):
    if not os.path.exists(path):
        from torchvision import datasets
        y = datasets.MNIST("./data", train=True, download=True).targets.numpy()
        rng = np.random.RandomState(42)
        while True:
            parts = [[] for _ in range(num_clients)]
            for c in range(10):
                idx = np.where(y == c)[0]; rng.shuffle(idx)
                cuts = (np.cumsum(rng.dirichlet([alpha] * num_clients)) * len(idx)).astype(int)[:-1]
                for m, s in enumerate(np.split(idx, cuts)):
                    parts[m] += s.tolist()
            if min(len(p) for p in parts) >= 10000:
                break
        np.savez(path, *[rng.permutation(p) for p in parts])
    P = np.load(path)
    return [P["arr_%d" % i] for i in range(len(P.files))]


if part.startswith("dirichlet"):
    SPLIT = fixed_dirichlet_split()
    def _fixed_partition(X, y, partition_type, num_clients, **kw):
        return [(X[p], y[p]) for p in SPLIT]
    fedne_loader.partition_data = _fixed_partition
    anchor_loader.partition_data = _fixed_partition
    part_cfg = {"type": "dirichlet", "alpha": 0.1}
else:
    part_cfg = {"type": "iid", "alpha": 0.5}


def upd(t, s):
    for k, v in s.items():
        if isinstance(v, dict) and isinstance(t.get(k), dict):
            upd(t[k], v)
        else:
            t[k] = v


if method == "fedne":
    cfg = {
        "dataset": {"name": "MNIST", "data_dir": "./data"},
        "partition": part_cfg,
        "federated": {"clients": 2, "rounds": R, "local_epochs": 1, "device": "cpu"},
        "model": {"input_dim": 784, "embedding_dim": 2, "hidden_dims": [512, 256, 128]},
        "graph": {"k": 5, "negative_samples": 5, "with_replacement": True},
        "augmentation": {"enabled": False},
        "surrogate": {"hidden_dim": 64, "grid_step": 0.3, "margin": 0.5, "lr": 0.001, "epochs": 2, "start_round": 0},
        "optimizer": {"lr": 0.001, "weight_decay": 0.0, "lr_decay_rounds": [], "lr_decay_factor": 1.0},
        "logging": {"tensorboard": False, "log_dir": "./runs", "checkpoint_dir": "./checkpoints", "output_dir": "./outputs", "seed": 42}}
    upd(cfg, extra)
    tr = rcb.ControlledFEDNETrainer(cfg)
    for c in tr.clients:
        print("labels", c.client_id, np.bincount(c.y_train, minlength=10).tolist(), flush=True)
    tr.fit()
    for h in tr.metrics_history:
        print("[FEDNE round %02d] kNN=%.4f T=%.4f C=%.4f" % (h["round"] + 1, h["knn_accuracy"], h["trustworthiness"], h["continuity"]), flush=True)
else:
    ov = {"dataset": {"name": "MNIST", "data_dir": "./data", "num_clients": 2, "partition": part_cfg["type"], "alpha": part_cfg["alpha"]},
          "anchors": {"num_anchors": 32},
          "training": {"mode": "neighbor_embedding", "rounds": R, "local_epochs": 1, "learning_rate": 0.001, "batch_size": 512},
          "graph": {"k": 5, "negative_samples": 5},
          "loss": {"lambda_attraction": 1.0, "lambda_local_repulsion": 1.0,
                   "lambda_anchor_repulsion": 0.0 if method == "norep" else 1.0,
                   "anchor_repulsion": {"normalization": "none", "scale": 1.0, "spread_scale": 1.0}},
          "seed": {"value": 42},
          "evaluation": {"output_dir": "fedanchor/outputs/v2_benchmark/run", "checkpoint_dir": "fedanchor/outputs/v2_benchmark/run"}}
    upd(ov, extra)
    tr = rcb.ControlledFEDANCHORTrainer(config_path="fedanchor/configs/mnist_default.yaml", override_config=ov)
    for c in tr.clients:
        print("labels", c.client_id, np.bincount(c.y_train, minlength=10).tolist(), flush=True)
    tr.train()
