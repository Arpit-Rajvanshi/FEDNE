# FEDANCHOR v7 — Results

*Note: This file should be populated with the actual metrics after running `run_v7_iid_experiment.py`.*

## 1. Baseline Comparison (IID)

| Method | Anchor Design | kNN Acc | Trustworthiness | Continuity | Entropy | Effective Anchors | Min Anchor Dist |
|--------|---------------|---------|-----------------|------------|---------|-------------------|-----------------|
| V5 | Client-specific | ~0.95 | ~0.95 | ~0.95 | ~0.98 | 5 | >0.5 |
| v7 | Shared global | 0.1010 | 0.5220 | 0.5015 | 0.9669 | 1 | 1.0274 |

## 2. Cross-Client Consistency Metrics (v7 Only)

| Metric | Value |
|--------|-------|
| Mean Client-to-Global Alignment Cost | 0.000000 |
| Max Client-to-Global Alignment Cost | 0.000000 |
| Pairwise Client L1 Divergence | ~0.0 |
| Pairwise Client JS Divergence | ~0.0 |
| Global Soft Utilisation Uniformity | 0.9669 (Entropy) |

## 3. Failure Mode Analysis

| Failure Mode | Detected? (Yes/No) | Indicator Value |
|--------------|--------------------|-----------------|
| Anchor Collapse | No | min_dist = 1.0274 |
| Anchor Drift | No | displacement = 0.00019 |
| Client Domination | No | max_weight = 0.5 |
| Uniform Util w/o Sep | Yes | flagged in rounds 1-2 |
| Encoder-Anchor Mismatch| Yes | embedding collapse |

## 4. Visualisations

Generated plots are located in: `outputs/plots/`
- `13_comparison_knn.png`
- `14_comparison_trustworthiness.png`
- `15_comparison_continuity.png`
- `05_anchor_trajectories.png`

## 5. Conclusion

No, shared anchors did not improve dimensionality reduction quality. In fact, they destroyed it (kNN=10.1%, effectively random guessing). 

The experiment revealed a critical mathematical flaw in the v7 design. Because the shared global anchor design removed the cross-client anchor repulsion (`L_rep` from V5), the objective function completely lacked any repulsive force for the embeddings. As a result, the local data embeddings collapsed to a single point. Once collapsed, the distance from all data points to the surrounding anchors became identical, causing the assignment probabilities to become uniform (minimizing the balance loss trivially). Thus, while shared anchors technically maintained a common reference frame, they destroyed the representation learning process because the necessary geometric embedding repulsion was inadvertently removed.
