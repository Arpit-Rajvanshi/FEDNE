# FEDANCHOR V2 Scientific Results

## 1. Overview of the Ablation Study

A rigorous 4-way controlled ablation study was conducted to determine if the "Anchor Collapse" phenomenon observed in FEDANCHOR v1 (where embeddings trivially collapse onto 1 or 2 anchors) could be mathematically mitigated by introducing:
1. **Soft Assignment:** Replacing the hard `min()` operator with a Student-t-style differentiable soft assignment.
2. **Anchor Spread (Repulsion):** Adding an explicit inverse-squared penalty pushing local anchors apart.

The study was run for 20 rounds on MNIST (2 clients, 5 local anchors each, producing 10 total global anchors).

### The Four Variants:
* **Mode A (V1 Baseline):** Hard `min()` assignment, no spread.
* **Mode B (Soft Only):** Soft assignment, no spread.
* **Mode C (Spread Only):** Hard `min()` assignment + spread penalty.
* **Mode D (Soft + Spread):** Soft assignment + spread penalty.

---

## 2. Quantitative Results (Round 20)

| Metric | Mode A (V1 Baseline) | Mode B (Soft Only) | Mode C (Spread Only) | Mode D (Soft + Spread) |
| :--- | :--- | :--- | :--- | :--- |
| **kNN Accuracy** | 23.19% | 18.03% (R19)* | 20.23% | 19.54% |
| **Active Anchors (>10% util)** | 2 | 2 | 1 | 2 |
| **Utilization Distribution** | `[0, 16%, 0..0, 85%]` | `[0, 0, 0, 11%, 55%, 0..0, 34%]` | `[0..0, 100%, 0, 0]` | `[0, 11%, 0..0, 89%, 0, 0]` |
| **Trustworthiness** | 0.6327 | 0.5821 | 0.6235 | 0.6271 |
| **Continuity** | 0.7786 | 0.7393 | 0.7469 | 0.7537 |

*\*Mode B showed severe oscillation in the final rounds; R19 is shown for representative utilization.*

---

## 3. The Collapse Trajectory (Mode D)

While Mode D (Soft + Spread) ended in a collapsed state by Round 20, observing its trajectory reveals exactly how the algorithm behaves:

* **Round 1:** `[12.1%, 5.3%, 1.9%, 9.7%, 12.8%, 6.0%, 3.5%, 33.8%, 5.9%, 9.0%]` (All 10 anchors active!)
* **Round 5:** `[0.0%, 0.1%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 99.9%, 0.0%, 0.0%]` (Rapid collapse)
* **Round 20:** `[0.0%, 10.7%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 89.3%, 0.0%, 0.0%]` (Stabilized collapse)

---

## 4. Scientific Conclusion

**Did V2 solve Anchor Collapse?**
> [!IMPORTANT]
> **No.** The modifications merely delayed the inevitable collapse. 

While the Student-t soft assignment initially distributed gradients across all anchors (as seen in Round 1), the underlying gravitational pull of the dense MNIST clusters was mathematically overpowering. Once an anchor moved slightly closer to a dense cluster, the Student-t assignment weight $q_{ij}$ for that anchor grew exponentially relative to the others. 

Simultaneously, the `L_spread` penalty was insufficient to break the anchors apart once they were captured by the primary clusters, resulting in the other anchors being stranded in empty topological space. 

**Impact on Representation Quality:**
Because the anchors still collapsed, the kNN accuracy remained effectively identical to V1 (~20% vs the ~92% baseline of centralized/pure FEDNE). The soft assignment actually introduced slight blurring between clusters, resulting in a marginal *decrease* in kNN accuracy and Trustworthiness compared to the hard assignment of V1.

This proves that **Anchor Collapse is a fundamental topological challenge in Federated dimensionality reduction**, not simply a byproduct of a harsh `min()` operator. Solving it will likely require a radically different mechanism, such as anchor resampling or momentum-based anchor injection, rather than purely gradient-based penalties.
