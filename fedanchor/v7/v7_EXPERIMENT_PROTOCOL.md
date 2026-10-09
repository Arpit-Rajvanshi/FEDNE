# FEDANCHOR v7 — Experiment Protocol

**Version**: 6.0.0
**Status**: Experimental Protocol

## 1. Objective
To evaluate the Shared Global Anchor design against the baseline Client-Specific Anchor design (V5), isolating the impact of the shared geometric reference frame.

## 2. Methodology
A strict controlled ablation. Both V5 and v7 run identical federated learning loops with identical hyperparameters. The ONLY difference is that v7 replaces client-specific anchors with a shared global anchor matrix updated via Hungarian alignment.

## 3. Dataset Configuration
| Parameter | Value | Notes |
|-----------|-------|-------|
| Dataset | MNIST | Standard benchmark |
| Partition | IID | Primary controlled setting |
| Clients | 2 | Minimal federated setup |
| Train Samples | 60,000 | 30,000 per client |
| Test Samples | 10,000 | Global evaluation |
| Alpha | 0.5 | Used for optional non-IID run |

## 4. Hyperparameters
| Parameter | Value |
|-----------|-------|
| Rounds | 20 |
| Local Epochs | 1 |
| Batch Size | 1280 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Max Grad Norm | 1.0 |
| Encoder Arch | 784 → 512 → 256 → 128 → 2 |
| Anchors (K) | 5 |

## 5. Loss Weights
Matches V5 exactly to prevent confounding variables.

| Component | Weight |
|-----------|--------|
| Attraction | 1.0 |
| Soft Anchor | 1.0 |
| Spread | 1.0 |
| Balance | 1.0 |
| Separation | 0.1 |
| Embedding Reg | 0.0 |
| Margin | 0.1 |
| Alpha (Student-t) | 1.0 |

*Note: Cross-client anchor repulsion is N/A in v7 as there is only one global anchor set.*

## 6. Initialization
- **Strategy**: KMeans on combined initial embeddings from all clients.
- **Max Samples**: 10,000 (randomly subsampled for efficiency).
- **Seed**: 42 (deterministic).

## 7. Evaluation Protocol
- **kNN Accuracy**: k=7, evaluated on the full test set.
- **Trustworthiness & Continuity**: k=7, evaluated on a randomly subsampled test set of 2,000 samples to manage O(N²) complexity.
- **Visualisation**: 15 distinct plots tracking both standard dimensionality reduction metrics and v7-specific anchor geometric consistency over rounds.

## 8. Execution Command
```bash
# From the root directory:
python -m fedanchor.v7.experiments.run_v7_iid_experiment
```
