# Research Positioning: FEDANCHOR V2

## Core Narrative

FEDANCHOR is an ambitious attempt to federate dimensionality reduction by replacing heavy gradient synchronization with the exchange of a small set of "Anchors" mapping the global topological space.

However, the V1 prototype exposed a critical mathematical flaw: **Anchor Collapse**.

The V2 research phase was designed specifically to probe whether this collapse was merely an artifact of the hard `min()` assignment operator, and whether a differentiable soft-assignment (Student-t-style) coupled with an explicit spread penalty could solve it.

The scientific conclusion of V2 is unequivocally **No**. While V2 successfully delays collapse and spreads anchors initially, the inherent gravitational pull of non-IID/IID dense local clusters forces the gradients to overwhelm the spread penalty, eventually stranding the majority of anchors and returning kNN accuracy to ~20%.

## Why is this a strong academic result?

A negative result in experimental computer science is often more valuable than a fragile positive one, provided it uncovers a fundamental property of the system.

1. **Identification of a Fundamental Limit:** We proved that gradient-based penalties and soft assignments are insufficient to solve spatial stranding in federated non-convex optimization.
2. **Scientific Rigor:** The 4-way ablation cleanly isolated the effects of soft assignment vs. spread penalty, proving neither works independently or in tandem.
3. **Honesty:** Rather than hiding behind a cherry-picked seed or obscure metric, the project explicitly acknowledges the 20% kNN accuracy and uses it to drive deep mathematical inquiry.

## Proposed Next Steps for Future Research

Based on the V2 findings, future iterations of FEDANCHOR must abandon purely gradient-based anchor adjustments. Proposed mechanisms include:

1. **Anchor Resampling:** Periodically pruning stranded anchors (utilization < 1%) and re-initializing them near active clusters.
2. **Momentum-Based Injection:** Using global server momentum to forcibly push anchors into unexplored regions of the latent space.
3. **Density-Aware Repulsion:** Replacing the naive Euclidean spread penalty with a topological density metric that penalizes clustering only in empty space.
