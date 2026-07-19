# FEDNE: Surrogate-Assisted Federated Neighbor Embedding

> **NeurIPS 2024** — PyTorch implementation of *Surrogate-Assisted Federated Neighbor Embedding for Dimensionality Reduction*

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Overview

FEDNE enables **collaborative dimensionality reduction** across distributed clients without sharing raw data. It maps high-dimensional datasets to 2D embeddings for visualization by resolving the lack of inter-client repulsion through a lightweight **surrogate repulsion network** architecture:

1. Each client trains a local surrogate on 2D grid query points
2. Surrogates are shared via the central server
3. Aggregated surrogates are incorporated into the global embedding model's training objective

This produces faithful 2D visualizations that respect both intra-client and inter-client neighborhood structure, all under **federated privacy constraints**.

---

## Key Features

| Feature | Description |
|---|---|
| **Paper-Faithful Loss** | Exact Equation (10) contrastive objective with attraction + local repulsion + surrogate repulsion |
| **Client Augmentation** | Intra-client Mixup with local kNN neighbors to simulate unseen neighbors on other clients |
| **Lightweight Surrogates** | Single-hidden-layer MLP repulsion estimators trained on configurable grid query points |
| **Flexible Partitioning** | IID, Dirichlet (α-imbalanced), and Shards (class-grouped) dataset splits |
| **Reproducibility** | Deterministic runs via random seed forcing across PyTorch, NumPy, and Python |
| **Rich Logging** | TensorBoard + JSON logs for losses, communication/computation times, and GPU memory |
| **Checkpointing** | Full state: global encoder, optimizer, all client surrogates, configs, and metrics history |

---

## Project Structure

```
FEDNE/
├── main.py                          # CLI entrypoint
├── requirements.txt                 # Python dependencies
├── fedne/
│   ├── configs/
│   │   └── mnist_default.yaml       # Default hyperparameters
│   ├── datasets/
│   │   └── loader.py                # MNIST/FashionMNIST loader & partitions
│   ├── models/
│   │   ├── encoder.py               # Parametric MLP encoder (D → 2)
│   │   └── surrogate.py             # Surrogate repulsion model & grid samplers
│   ├── losses/
│   │   ├── attraction.py            # Attraction term ϕ(zᵢ, zⱼ)
│   │   ├── repulsion.py             # Local negative repulsion
│   │   └── contrastive_loss.py      # Eq. (10) combined objective
│   ├── graph/
│   │   └── knn.py                   # Local kNN graph construction
│   ├── augmentation/
│   │   └── mixup.py                 # Neighbor Mixup data augmentation
│   ├── client/
│   │   └── client.py                # Local encoder/surrogate client training
│   ├── server/
│   │   └── server.py                # FedAvg coordinator & surrogate broadcast
│   ├── training/
│   │   └── trainer.py               # Orchestrator & checkpoint manager
│   ├── evaluation/
│   │   ├── metrics.py               # Trust., Continuity, kNN Acc, S&C
│   │   └── visualization.py         # Matplotlib 2D embedding plots
│   └── utils/
│       └── seed.py                  # Deterministic seeds helper
├── tests/                           # Unit & integration tests
├── experiments/
│   └── mnist/
│       └── scripts/
│           └── run_mnist_benchmark.py   # Paper replication benchmarks
└── data/                            # Auto-downloaded by torchvision (gitignored)
```

---

## Getting Started

### Prerequisites

- Python ≥ 3.8
- pip (or conda)
- (Optional) CUDA-capable GPU for faster training

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/FEDNE.git
cd FEDNE

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The `snc` library computes the official Steadiness & Cohesiveness metrics. If its C-compilation fails, the evaluation module automatically falls back to KMeans-based NMI/ARI metrics.

### Quick Start — Train & Evaluate

```bash
# Train with default MNIST configuration
python main.py --config fedne/configs/mnist_default.yaml

# Resume from a checkpoint
python main.py --config fedne/configs/mnist_default.yaml --resume checkpoints/latest
```

### Monitor Training

```bash
tensorboard --logdir runs
```

---

## Configuration

Edit `fedne/configs/mnist_default.yaml` or create custom config files:

| Parameter | Options / Default | Description |
|---|---|---|
| `dataset` | `MNIST`, `FashionMNIST` | Dataset to use |
| `partition` | `iid`, `dirichlet`, `shards` | Data split strategy |
| `alpha` | `0.5` | Dirichlet concentration (lower = more imbalanced) |
| `shards_per_client` | `2` | Number of class shards per client |
| `grid_step` | `0.3` | Surrogate grid query spacing |
| `margin` | `0.5` | Surrogate grid margin |
| `seed` | `42` | Random seed for reproducibility |

---

## Methodology

### Loss Objective (Equation 10)

The optimization minimizes:

$$\mathcal{L} = \mathcal{L}_{\text{att}} + \mathcal{L}_{\text{rep\_local}} + \mathcal{L}_{\text{rep\_surr}}$$

- **Attraction Loss** — pulls neighbor pairs together:

$$\mathcal{L}_{\text{att}} = -\log(\phi(z_i, z_j)) \quad \text{where} \quad \phi(z_i, z_j) = \frac{1}{1 + \|z_i - z_j\|^2}$$

- **Local Repulsion Loss** — pushes non-neighbor samples apart within each client:

$$\mathcal{L}_{\text{rep\_local}} = -\sum_{k=1}^b \log(1 - \phi(z_i, z_{\text{neg}_k}))$$

- **Surrogate Repulsion Loss** — approximates cross-client repulsion via surrogate networks:

$$\mathcal{L}_{\text{rep\_surr}} = \sum_{m' \neq m} \frac{|D_{m'}|}{|D|} S_{m'}(z_i)$$

### Evaluation Metrics

| Metric | What It Measures |
|---|---|
| **Trustworthiness** | Preservation of NN relationships from 2D → high-D |
| **Continuity** | Preservation of NN relationships from high-D → 2D |
| **kNN Accuracy** | Classification accuracy of kNN on embeddings |
| **Steadiness** | Embedding cluster purity (via `snc` or KMeans NMI) |
| **Cohesiveness** | Spatial compactness of clusters (via `snc` or KMeans ARI) |
| **Surrogate MSE** | Quality of surrogate repulsion approximation |

### Convergence Results (Validation Run)

1,000 MNIST samples · 2 clients · 10 rounds:

| Round | Total Loss | Attraction | Local Rep | Surr MSE | Trustworthiness | Continuity | kNN Acc | Steadiness |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 3.1556 | 0.6360 | 4.8510 | 1.3420 | 0.7001 | 0.8276 | 0.3000 | 0.5122 |
| 5 | 1.1061 | 0.5539 | 0.9873 | 0.3504 | 0.7943 | 0.8705 | 0.4000 | 0.6178 |
| 9 | 1.0346 | 0.5003 | 0.8237 | 0.2032 | 0.8367 | 0.8802 | 0.6750 | 0.6193 |

**Key observations:**
- Total loss: **3.15 → 1.03** (compact clusters)
- Surrogate MSE: **1.34 → 0.20** (accurate cross-client repulsion)
- Trustworthiness: **0.70 → 0.84** · kNN Accuracy: **30% → 67.5%**

---

## Paper Replication Experiments

Run the full benchmark suite matching NeurIPS 2024 paper tables:

```bash
python experiments/mnist/scripts/run_mnist_benchmark.py
```

Partitions tested:
- **Dirichlet**: α = 0.1, α = 0.5
- **Shards**: C = 2, C = 3
- **IID**

Results are written to `experiments/mnist/results/`.

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Individual test scripts
python tests/test_components.py
python tests/verify_toy_dataset.py
python tests/run_mnist_validation_benchmark.py
```

---

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{fedne2024neurips,
  title     = {Surrogate-Assisted Federated Neighbor Embedding for Dimensionality Reduction},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
