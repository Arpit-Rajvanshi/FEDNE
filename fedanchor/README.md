# Federated Anchor for Dimensionality Reduction (FEDANCHOR)

> **Research Prototype Disclaimer**: This codebase is an independent, initial prototype for an anchor-based approach to federated dimensionality reduction. It is **not** a validated replacement for FEDNE. Baseline comparison experiments will follow.

---

## 1. Executive Summary & Context

In federated neighbor embedding (e.g., FEDNE), each client possesses private high-dimensional data. While clients can locally construct neighborhood graphs for attraction, modeling **cross-client repulsion** is challenging because raw samples cannot be shared across clients.

- **FEDNE Baseline**: Employs local neighbor embedding combined with **surrogate-assisted cross-client repulsion functions**.
- **FEDANCHOR Prototype**: Investigates whether a **small set of learned 2D anchors** $A_m = \{a_{m1}, a_{m2}, \dots, a_{mK}\} \subset \mathbb{R}^2$ can compactly represent key regions of client $m$'s embedding space, enabling anchor-based cross-client repulsion.

Existing FEDNE implementation files in `d:\FEDNE\` are kept strictly **READ-ONLY** as our baseline.

---

## 2. Terminology: What is an Anchor?

An **anchor** $a_{mj} \in \mathbb{R}^2$ is a **learned representative point in the 2D embedding space**. 
- It does **NOT** mean an "anchor client".
- Each client maintains $K$ learnable anchor points $A_m = [a_{m1}, a_{m2}, \dots, a_{mK}]^T \in \mathbb{R}^{K \times 2}$.

---

## 3. Mathematical Formulation & Phase 2 Improvements

### 3.1 Local Attraction Loss ($L_{att}$)
For local kNN graph edges $(i, j) \in E$:
$$\phi(z_i, z_j) = \frac{1}{1 + \|z_i - z_j\|^2}$$
$$L_{att} = \frac{1}{|E|} \sum_{(i,j) \in E} \log\left(1 + \|z_i - z_j\|^2 + \epsilon\right)$$

### 3.2 Anchor Coverage Loss ($L_{anchor}$)
Encourages client $m$'s anchors to cover its local low-dimensional embedding space (Vector Quantization distortion):
$$L_{anchor} = \frac{1}{N} \sum_{i=1}^N \min_{j=1..K} \|z_i - a_j\|^2$$

### 3.3 Anchor-Based Cross-Client Repulsion ($L_{rep\_anchor}$)
> [!IMPORTANT]
> $L_{rep\_anchor}$ is **our proposed prototype formulation** and is **NOT** an equation from FEDNE.

Given local embeddings $z_i$ on client $m$ and combined anchors from all other clients $A_{other} = \bigcup_{m' \neq m} A_{m'}$:
$$L_{rep\_raw} = \frac{1}{N \cdot |A_{other}|} \sum_{i=1}^N \sum_{a \in A_{other}} \log\left(1 + \frac{1}{\|z_i - a\|^2 + \epsilon}\right)$$

In Phase 2, a configurable scaling mechanism was introduced to manage loss scale imbalance:
$$L_{rep\_scaled} = \text{scale} \cdot L_{rep\_raw}$$

### 3.4 Optional Embedding Scale Regularization ($L_{emb\_reg}$)
$$L_{emb\_reg} = \frac{1}{N} \sum_{i=1}^N \|z_i\|^2$$

### 3.5 Total Client Objective
$$L_{total} = \lambda_{att} L_{att} + \lambda_{anchor} L_{anchor} + \lambda_{rep\_anchor} L_{rep\_scaled} + \lambda_{emb\_reg} L_{emb\_reg}$$

---

## 4. Phase 2 Key Enhancements & Findings

1. **Repulsion Loss Scaling**:
   - Added `loss.anchor_repulsion.normalization` (`none` | `scale`) and `scale` factor in YAML configs.
   - Reduced early-round repulsion dominance while keeping explicit raw vs. scaled loss logs.
2. **Loss Scale Diagnostics**:
   - Per-round tracking of `repulsion / attraction` and `repulsion / anchor` magnitude ratios.
3. **Anchor Initialization Strategies**:
   - **`kmeans`**: K-Means clustering on initial 2D local embeddings $Z_m^{(0)}$ (default).
   - **`farthest_point`**: Farthest Point Sampling (FPS) on $Z_m^{(0)}$.
   - **`representative_points`**: Even index sampling retained for baseline reproduction.
4. **Anchor Utilization & Geometric Diagnostics**:
   - Tracks mean and max distance to nearest anchor, as well as percentage utilization across anchors $1 \dots K$.
5. **Autograd & Unit Test Suite**:
   - 15 passing tests including non-zero repulsion encoder gradients, local anchor parameter updates, other-client anchor gradient isolation, and zero coverage at centroids.
6. **20-Round Extended Federated Run Results**:
   - kNN Classification Accuracy increased from **9.18% (Round 1) to 66.31% (Round 20)**.
   - Trustworthiness increased from **0.6202 to 0.8647**.
   - Continuity increased from **0.7888 to 0.9126**.

---

## 5. Privacy & Communication Boundaries

| Entity | Shared across Network? | Notes |
| :--- | :--- | :--- |
| **Raw Images / Labels** | ❌ **PRIVATE** | Stays on local client. |
| **kNN Neighborhood Graph** | ❌ **PRIVATE** | Constructed locally via `sklearn`. |
| **Local Embeddings ($z_i$)** | ❌ **PRIVATE** | Never uploaded to server. |
| **Encoder Weights ($\theta_m$)** | ✅ **SHARED** | Aggregated via dataset-size weighted FedAvg. |
| **Anchor Points ($A_m$)** | ✅ **SHARED** | $K \times 2$ float32 scalars exchanged per client. |

---

## 6. How to Run

### Run Unit & Autograd Test Suite
```bash
.venv\Scripts\python.exe -m unittest discover -s fedanchor/tests
```

### Run Main Experiment
```bash
.venv\Scripts\python.exe fedanchor/main_anchor.py --config fedanchor/configs/mnist_default.yaml
```

### Run Phase 2 Experiment Suite (Init Comparison, Ablations, 20-Round Run)
```bash
.venv\Scripts\python.exe fedanchor/run_phase2_experiments.py
```

---

## 7. Communication Analysis

- **Encoder upload/download per client**: **2,265,608 bytes** (566,402 float32 scalars)
- **Anchor upload per client ($K=5$)**: **40 bytes** (10 float32 scalars)
- **Anchor download per client ($M=2$)**: **40 bytes** (10 float32 scalars)
- **Anchor scalar ratio**: $\frac{10}{566402} \approx 0.0018\%$ of encoder size.

---

## 8. What Remains Unresolved & Future Directions
1. **Anchor Collapse over Long Horizons**: In high-epoch regimes without repulsion gradient clipping on anchors, points can collapse to dominant anchors (100% utilization on single anchor).
2. **Adaptive Dynamic Loss Weighting**: Automatic initial loss balancing is supported optionally, but dynamic adaptive balancing across rounds remains open for investigation.
