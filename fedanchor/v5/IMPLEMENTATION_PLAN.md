# FEDANCHOR v2 Implementation Plan

## 1. Goal Description
The purpose of this phase is to experimentally validate whether Soft Anchor Assignment and Explicit Anchor Spread (Anchor-to-Anchor Repulsion) can mitigate the "Anchor Collapse" phenomenon observed in FEDANCHOR v1, where local embeddings trivially concentrate on 1 or 2 anchors. This is a scientific investigation, prioritizing rigorous diagnostic metrics over top-line kNN accuracy improvements.

## 2. Current v1 Formulation & Identified Failure Mode
**Current Formulation (`AnchorCoverageLoss`):**
$$ L_{anchor} = \frac{1}{N} \sum_{i=1}^{N} \min_{j} \|z_i - a_j\|^2 $$
**Failure Mode:** The hard `min` operator acts as a "winner-take-all" assignment. Once an anchor is slightly closer to a dense cluster, it captures all gradients from that cluster, stranding the other anchors in empty space without gradient updates. This leads to severe anchor collapse (e.g. utilization = `[0%, 99%, 0%, 0%, 1%]`), preventing the anchors from spreading to cover the full topological structure.

## 3. Proposed v2 Changes

### A. Soft Anchor Assignment
Replace the hard `min` assignment with a differentiable soft assignment inspired by the Student-t distribution:
1. **Similarity Computation:**
   $$ q_{ij} = \frac{(1 + \|z_i - a_j\|^2 / \alpha)^{-\frac{\alpha+1}{2}}}{\sum_{k=1}^{K} (1 + \|z_i - a_k\|^2 / \alpha)^{-\frac{\alpha+1}{2}}} $$
   Where $\alpha$ is a degree-of-freedom hyperparameter (default $\alpha=1$).
2. **Soft Anchor Loss:**
   $$ L_{soft\_anchor} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{K} q_{ij} \|z_i - a_j\|^2 $$

### B. Anchor-to-Anchor Spread (Repulsion)
Penalize anchors from clustering too closely together:
$$ L_{spread} = \frac{1}{K(K-1)} \sum_{i \neq j} \frac{1}{\|a_i - a_j\|^2 + \epsilon} $$
Controlled by a new config hyperparameter `lambda_spread`.

### C. Total v2 Objective
$$ L_{total\_v2} = \lambda_{att} L_{att} + \lambda_{anchor} L_{soft\_anchor} + \lambda_{rep} L_{rep} + \lambda_{spread} L_{spread} + \lambda_{reg} L_{reg} $$

## 4. Proposed File Architecture
To preserve FEDANCHOR v1 while experimenting, we will duplicate and modify components into a standalone `fedanchor/v2` module:
- `v2/losses/anchor_soft_loss.py`: Implements $L_{soft\_anchor}$
- `v2/losses/anchor_spread_loss.py`: Implements $L_{spread}$
- `v2/losses/total_loss_v2.py`: Integrates all losses according to the v2 objective
- `v2/client/client_v2.py`: Uses TotalLossV2
- `v2/training/trainer_v2.py`: Instantiates ClientV2 and handles the new diagnostics
- `v2/tests/`: New test suite validating shapes, finite gradients, and $q_{ij}$ properties
- `v2/diagnostics/`: Synthetic micro-experiment scripts and outputs
- `v2/run_controlled_ablation_v2.py`: The rigorous 4-way ablation script (V1, Soft, Spread, Soft+Spread)

## 5. Experiment Plan
1. **Unit Testing:** Prove numerical stability, gradient flow to all anchors, and expected output shapes.
2. **Micro-Experiment:** A tiny synthetic 2D test demonstrating that v1 hard-assignment concentrates anchors while v2 soft-assignment and spread push anchors apart.
3. **Controlled Ablation:** Run 20 rounds of MNIST (2 clients, Seed 42, Identical Protocol) across 4 variants (V1, V1+Soft, V1+Spread, V1+Soft+Spread) to determine independent and synergistic effects.
4. **Diagnostic Metrics:** Track anchor utilization, anchor entropy, and pairwise distances alongside kNN accuracy and Trustworthiness.
5. **Sensitivity Analysis:** Test variations of `lambda_spread` and `alpha`.
6. **Scientific Reporting:** Produce `SCIENTIFIC_RESULTS.md`, `RESEARCH_POSITIONING.md`, and `FEDANCHOR_V2_FINAL_REPORT.md` analyzing the true impact of the changes.

> [!WARNING] 
> **Scientific Integrity Rule**
> The goal is to evaluate if the algorithmic changes reduce the mathematical phenomenon of "Anchor Collapse", *not* to force a higher kNN accuracy. If the changes distribute anchors evenly but degrade representation quality, the report will honestly state that finding.
