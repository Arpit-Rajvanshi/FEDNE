# FINAL CONTROLLED BENCHMARK RESULTS

## Protocol Summary
- **Dataset:** MNIST (60k Train, 10k Test, IID Partition)
- **Settings:** 2 Clients, Seed=42, 20 Rounds, Local Epochs=1, LR=0.001
- **Evaluation Mechanism:** Unified Protocol (identical random subset of 2,000 embeddings for both kNN fitting and testing; k=5).

## 1. Final Quantitative Comparison (Round 20)

| Metric | FEDNE (Baseline) | FEDANCHOR (Prototype) | Delta |
|--------|------------------|-----------------------|-------|
| **kNN Accuracy** | 91.95% | 20.50% | -71.45% |
| **Trustworthiness** | 0.9375 | 0.6257 | -0.3118 |
| **Continuity** | 0.9581 | 0.7642 | -0.1939 |
| **Total Loss** | 0.4339 | 0.0544 | N/A |

## 2. Communication Overhead (per Client per Round)

| Component | FEDNE Bytes | FEDANCHOR Bytes | Benefit |
|-----------|-------------|-----------------|---------|
| **Encoder** | 2,265,608 | 2,265,608 | 0 |
| **Cross-Client Signal** | 1,028 (Surrogate) | 40 (Anchors) | -96% |
| **Total Upload** | 2,266,636 (~2.27 MB)| 2,265,648 (~2.27 MB)| -0.04% |

## 3. 10-Round Ablation Study (FEDANCHOR)

| Mode | kNN Accuracy | Trustworthiness | Anchor Utilization (Sample) |
|------|--------------|-----------------|-----------------------------|
| **Attraction Only** | 19.35% | 0.5879 | Moderate spread (4 anchors used) |
| **Attraction + Anchor (Coverage)** | 14.88% | 0.5873 | Collapse (2 anchors used) |
| **Attraction + Repulsion** | 22.36% | 0.6367 | Extreme collapse (1 anchor used, 100%) |
| **Full FEDANCHOR** | 22.44% | 0.6382 | Severe collapse (2 anchors used, 51%/49%) |

## 4. Diagnostics & Stability
FEDANCHOR currently suffers from **Anchor Collapse**, where the K-Means-like coverage objective (`min` distance) causes a "winner-take-all" dynamic. By Round 20, the local embeddings highly concentrate around 1 or 2 anchors (e.g., Utilization = [0.0%, 99.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 1.0%]). This localized collapse prevents the cross-client repulsion signal from properly separating the classes, resulting in the ~20% kNN accuracy.
