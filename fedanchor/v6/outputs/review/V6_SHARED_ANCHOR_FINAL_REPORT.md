# FEDANCHOR V6: Shared Global Anchors - Final Report

**Date:** October 2026
**Version:** 6.0.0

## 1. Research Question & Motivation
**Question:** Can a shared/global anchor set provide a common low-dimensional geometric reference frame across federated clients and improve dimensionality-reduction quality compared with independently learned client-specific anchors?

**Motivation:** Previous iterations (V1-V5) utilized client-specific anchors. This allowed local flexibility but lacked cross-client geometric consistency. Anchor `j` on one client had no relation to anchor `j` on another, potentially hurting the federated aggregation of the encoder. V6 tests the hypothesis that a shared reference frame improves overall performance.

## 2. Architecture & Algorithm
In V6, a single global anchor matrix `A ∈ ℝ^(K × 2)` is maintained on the server.
1. **Broadcast:** Server sends the global encoder and the shared anchor matrix `A` to all clients.
2. **Local Training:** Clients optimize both the encoder and their local copy of `A` using the V5 loss structure (excluding cross-client repulsion).
3. **Alignment & Aggregation:** Server receives updated anchor matrices from clients. Because anchors are permutation-symmetric, the server uses the **Hungarian Algorithm** to align each client's anchors to the global reference based on Euclidean distance.
4. **Update:** Server computes a weighted average of the aligned client anchors to produce the new global `A`.

## 3. Experimental Protocol
A strictly controlled IID ablation was designed to isolate the impact of shared anchors:
- **Dataset:** MNIST (60k train, 10k test)
- **Partition:** IID (2 clients)
- **Encoder:** 784 → 512 → 256 → 128 → 2
- **Hyperparameters:** 20 rounds, 1 local epoch, Adam (lr=0.001), batch size 1280.
- **Anchors:** K=5, initialized via KMeans on combined initial embeddings.

## 4. Implementation Details
The full implementation resides in `D:\FEDNE\fedanchor\v6\`.
- **`models/global_anchor.py`**: Defines `GlobalAnchorSet`.
- **`aggregation/hungarian_align.py`**: Solves the permutation ambiguity.
- **`losses/total_loss_v6.py`**: Adapted objective function.
- **`tests/test_v6.py`**: 15 comprehensive unit tests (all passing), including synthetic permutation recovery.

## 5. Results & Conclusion
*(Pending execution of `run_v6_iid_experiment.py`)*

The system is fully implemented and tested. To obtain the final scientific results:
1. Run the experiment: `python -m fedanchor.v6.experiments.run_v6_iid_experiment`
2. Generate plots: `python -m fedanchor.v6.experiments.generate_v6_plots`
3. Compare the metrics (kNN, Trustworthiness, Continuity) against the V5 baseline to conclude whether the shared geometry hypothesis holds true.

## 6. Future Work
- Execute the optional non-IID (Dirichlet) experiment to test if shared anchors are more or less beneficial under severe data heterogeneity.
- Investigate alternative alignment metrics beyond simple Euclidean distance for the Hungarian matching.
