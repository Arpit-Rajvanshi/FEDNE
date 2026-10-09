# FEDANCHOR V6 — Shared Global Anchors Design Document

**Version**: 6.0.0  
**Status**: Research Implementation  
**Location**: `D:\FEDNE\fedanchor\v6\`

---

## 1. Research Question

> "Can a shared/global anchor set provide a common low-dimensional geometric reference frame across federated clients and improve dimensionality-reduction quality compared with independently learned client-specific anchors?"

---

## 2. Scientific Hypotheses

### H1 — Geometric Consistency
Shared anchors improve cross-client geometric consistency because every client optimises its embedding relative to the same reference frame.

### H2 — Quality Improvement
Shared anchors may improve dimensionality-reduction metrics by reducing inconsistencies between independently learned client anchor systems.

### H3 — Potential Failure Mode
Shared anchors may also introduce a failure mode: one client's geometry can influence the shared reference frame and potentially hurt another client.

> **Important**: None of these hypotheses are assumed true prior to the experiment. The experiment must test both benefits and risks.

---

## 3. Architectural Change: V5 → V6

### V5 (Client-Specific Anchors)
```
GLOBAL SERVER
    │ global encoder parameters
    ↓
CLIENT 1 → own anchor set A₁
CLIENT 2 → own anchor set A₂
CLIENT N → own anchor set Aₙ
```

Each client maintains its own independent anchor geometry. Anchor 1 on Client A may not correspond geometrically to Anchor 1 on Client B.

### V6 (Shared Global Anchors)
```
              GLOBAL SERVER
          ┌──────────────────┐
          │ Global Encoder θ │
          │ Shared Anchors A │
          └────────┬─────────┘
                   │
           Broadcast θ, A
                   │
    ┌──────────────┼──────────────┐
    ↓              ↓              ↓
CLIENT 1       CLIENT 2       CLIENT N
local data     local data     local data
    │              │              │
 encoder        encoder        encoder
    │              │              │
   z₁             z₂             zN
    │              │              │
    └────── SAME SHARED ANCHORS ──┘
                   │
              local updates
                   ↓
          Hungarian alignment
                   ↓
            server aggregation
                   ↓
           new global anchors
```

All clients use the **exact same** global anchor matrix A ∈ ℝ^(K × 2) for a given round.

---

## 4. Mathematical Formulation

### 4.1 Notation

| Symbol | Meaning |
|--------|---------|
| K | Number of shared anchor points |
| A ∈ ℝ^(K×2) | Global shared anchor matrix |
| aⱼ ∈ ℝ² | j-th anchor coordinate |
| zᵢ ∈ ℝ² | 2D embedding of sample i |
| qᵢⱼ | Soft assignment of sample i to anchor j |
| N_m | Number of training samples on client m |

### 4.2 Soft Assignment (Student-t Distribution)

```
           (1 + ‖zᵢ - aⱼ‖² / α)^(-(α+1)/2)
qᵢⱼ = ─────────────────────────────────────────────────
        Σₖ (1 + ‖zᵢ - aₖ‖² / α)^(-(α+1)/2)
```

with α = 1.0 (degrees of freedom parameter).

### 4.3 Loss Function

```
L_total = λ_att   · L_att
        + λ_anchor · L_soft
        + λ_spread · L_spread
        + λ_balance· L_balance
        + λ_sep    · L_sep
        + λ_reg    · L_reg
```

Where:
- **L_att**: Graph neighbourhood attraction (kNN graph preservation)
- **L_soft**: `(1/N) Σᵢ Σⱼ qᵢⱼ ‖zᵢ - aⱼ‖²`
- **L_spread**: Local repulsion between anchor pairs (spread)
- **L_balance**: Entropy-based balance penalty on soft assignments
- **L_sep**: Explicit margin-based separation: `(1/K(K-1)) Σᵢ≠ⱼ max(0, m - ‖aᵢ-aⱼ‖)²`
- **L_reg**: L2 embedding regularisation (disabled by default)

> **Note**: Cross-client anchor repulsion (L_rep from V5) is removed because all clients share the same anchor set — there are no other-client anchors to repel against.

---

## 5. Shared Anchor Update Algorithm

### The Permutation Problem

Anchors are **permutation-symmetric**: if Client A learns [a1, a2] and Client B learns the same geometry but with swapped indices [a2, a1], naive parameter averaging computes midpoints — collapsing both anchors.

### Solution: Hungarian Algorithm

```
Input: global anchors A_t, client updates {A_t^m}
Output: A_{t+1}

For each client m:
    1. Compute cost matrix C[i,j] = ‖A_t^m[i] - A_t[j]‖²    [K×K]
    2. Solve Hungarian assignment: (row_ind, col_ind) = linear_sum_assignment(C)
    3. Permute: Ã_t^m[col_ind[r]] = A_t^m[row_ind[r]]

After alignment:
    A_{t+1} = Σ_m (N_m / N_total) · Ã_t^m
```

This ensures that anchor j of each client's aligned update corresponds to anchor j of the global reference.

---

## 6. Global Anchor Initialization

Performed **ONCE** before federated training begins.

**Procedure**:
1. Run the initial (untrained) global encoder on all clients' training data.
2. Concatenate all initial embeddings: Z_combined ∈ ℝ^(N_total × 2).
3. Fit KMeans(K) on Z_combined.
4. Set A₀ = cluster centres.
5. Record initialization provenance to `anchor_initialization.json`.

**Properties**:
- Label-free: does not use y labels.
- Anchors are NOT called class prototypes.
- Deterministic: fixed random seed 42.
- Fallback: 'farthest_point' strategy if KMeans fails.

---

## 7. Anchor Consistency Guarantee

At the start of every round, before local training:

```
max |A_client_m - A_global| ≈ 0
```

This is verified programmatically via `init_consistency_max` returned by each client. A value > 1e-5 indicates a broadcast bug.

---

## 8. Anchor Utilisation

### Per-Client
```
u_j^(m) = (1/N_m) Σᵢ qᵢⱼ^(m)
```

### Global (sample-weighted)
```
u_j^(global) = (1/N_total) Σ_m N_m · u_j^(m)
```

Both are reported every round. A globally balanced distribution can hide severe per-client imbalance.

---

## 9. New V6 Cross-Client Consistency Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| L1 divergence | Σⱼ |u_j^(i) - u_j^(j)| | Assignment distribution distance |
| Jensen-Shannon divergence | JS(u_i ‖ u_j) | Symmetric distribution distance |
| Pearson correlation | r(u_i, u_j) | Utilisation pattern similarity |
| Mean nearest-anchor distance | (1/N_m) Σᵢ min_j ‖zᵢ - aⱼ‖ | Embedding-anchor fit |
| Broadcast consistency check | max|A_m - A_global| | Verifies identical broadcast |

---

## 10. Failure Mode Detectors

| Mode | Detection Condition |
|------|-------------------|
| Anchor collapse | min_pairwise_dist < 0.01 |
| Anchor drift | per-round displacement > 0.5 |
| Client domination | one client has weight ≥ 90% |
| Uniform util / no separation | std(u_global) < 0.05 AND min_dist < 0.5 |
| Encoder-anchor mismatch | mean nearest dist > 2.0 |

---

## 11. Implementation Files

```
fedanchor/v6/
├── __init__.py
├── models/
│   ├── __init__.py
│   └── global_anchor.py       — GlobalAnchorSet (shared nn.Parameter)
├── aggregation/
│   ├── __init__.py
│   └── hungarian_align.py     — Hungarian alignment + weighted aggregation
├── losses/
│   ├── __init__.py
│   └── total_loss_v6.py       — TotalLossV6 (V5 structure, no cross-client repulsion)
├── client/
│   ├── __init__.py
│   └── client_v6.py           — AnchorClientV6 (resets anchors from server each round)
├── training/
│   ├── __init__.py
│   ├── server_v6.py           — V6ServerWithSharedAnchors
│   └── trainer_v6.py          — AnchorTrainerV6 (main orchestrator)
├── evaluation/
│   ├── __init__.py
│   └── metrics_v6.py          — Extended V5 metrics + cross-client diagnostics
├── diagnostics/
│   ├── __init__.py
│   └── round_diagnostics.py   — Per-round V6 diagnostic builder + failure detector
├── experiments/
│   ├── __init__.py
│   ├── run_v6_iid_experiment.py    — Primary controlled IID experiment
│   ├── run_v6_noniid_experiment.py — Optional non-IID experiment
│   └── generate_v6_plots.py        — All 15 visualisations
├── tests/
│   ├── __init__.py
│   └── test_v6.py             — 15 unit tests including synthetic permutation test
└── outputs/
    ├── review/
    ├── plots/
    ├── controlled_shared_anchor/
    └── reports/
```

---

## 12. Experimental Protocol

### Primary (IID) Experiment

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST |
| Train | 60,000 |
| Test | 10,000 |
| Clients | 2 |
| Partition | IID |
| Seed | 42 |
| Rounds | 20 |
| Local epochs | 1 |
| Encoder | 784→512→256→128→2 |
| Optimizer | Adam |
| LR | 0.001 |
| Batch size | 1280 |
| Anchors K | 5 |
| kNN k | 7 |
| T&C subsample | 2000 |

### Comparison

V5 and V6 must use **identical** settings for all parameters above. The **only** intended major difference is the anchor design.

---

*Created for FEDANCHOR V6 research. Do not modify V1–V5.*
