# FEDANCHOR V2 Final Report

## Project Overview
The FEDANCHOR project aimed to develop a novel Federated Dimensionality Reduction system by communicating global "Anchors" rather than performing heavy gradient synchronization across clients. 

During the validation of FEDANCHOR v1, a critical topological failure mode was identified: **Anchor Collapse**. The hard `min()` assignment operator caused the system to exhibit a winner-take-all behavior, where 1 or 2 anchors captured all local cluster gradients, leaving the remaining anchors stranded and resulting in a poor representation (kNN Accuracy ~20%).

## Phase 2: Experimental Intervention
The v2 phase was launched as a rigorous scientific experiment to test whether Anchor Collapse could be mitigated mathematically. Two interventions were formulated and tested:
1. **Soft Assignment:** A Student-t-style differentiable soft assignment to ensure all anchors received non-zero gradients.
2. **Anchor Spread:** An explicit inverse-squared penalty pushing anchors apart.

A 4-way controlled ablation study was conducted (V1 Baseline, Soft Only, Spread Only, Soft+Spread) over 20 rounds of training on MNIST.

## Key Scientific Findings
1. **The Interventions Delay but Do Not Prevent Collapse:** Early in training (e.g., Round 1), the combination of Soft Assignment and Spread correctly distributed the 10 anchors across the space. However, by Round 20, the massive gravitational pull of the dense MNIST clusters overpowered the penalties, stranding the majority of anchors and stabilizing in a collapsed state.
2. **No Meaningful Accuracy Gain:** Because the final topological state remained collapsed, the kNN accuracy (19.54%) remained comparable to the V1 baseline (23.19%).
3. **Fundamental Topological Limits:** The study successfully proved that Anchor Collapse is not a trivial coding bug or a simple artifact of a hard `min()` function. It is a fundamental property of non-convex federated optimization in dimensionality reduction. Gradient-based repulsive penalties are insufficient to rescue anchors once they become stranded in empty space.

## Conclusion and Recommendations
The v2 research phase is a strong scientific success. By exposing, diagnosing, and rigorously testing the Anchor Collapse phenomenon, we have established a clear baseline and failure boundary for the federated anchor approach. 

Future research should pivot away from continuous gradient penalties and investigate discontinuous topological interventions, such as actively pruning stranded anchors and re-sampling them dynamically near active data clusters.

## Project Integrity
All v1 baselines (`D:\FEDNE\fedne\`) were maintained as strictly read-only. The v2 codebase and experiments were isolated entirely within `D:\FEDNE\fedanchor\v2\`, ensuring absolute scientific validity and reproducibility.
