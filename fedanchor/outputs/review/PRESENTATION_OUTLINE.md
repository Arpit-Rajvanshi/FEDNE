# FEDANCHOR PRESENTATION OUTLINE

## General Guidelines
- **Time Allocation:** 15 minutes (approx. 1 minute per slide + 3 mins Q&A).
- **Tone:** Academic, transparent, and rigorous.
- **Rule of Thumb:** DO NOT present unverified historical numbers. Use ONLY the data generated from the controlled benchmark.

---

## Slide 1: Title Slide
**Title:** Federated Anchor for Dimensionality Reduction (FEDANCHOR)
**Subtitle:** An Empirical Evaluation of Cross-Client Representation

**Speaker Notes:**
- Welcome the panel.
- State the goal: We propose and evaluate FEDANCHOR, an alternative method for federated dimensionality reduction focusing on reducing communication overhead.

---

## Slide 2: The Problem and Motivation
**Title:** Federated Neighbor Embedding

**Content:**
- **Goal:** Learn a global low-dimensional (2D) embedding of distributed high-dimensional data (e.g., images) without sharing the raw data.
- **Challenge:** Neighbor embedding relies on repulsion (pushing dissimilar points apart). In a federated setting, clients cannot see each other's data to compute cross-client repulsion.
- **Consequence:** Without cross-client repulsion, overlapping clusters collapse into the same region of the embedding space.

**Speaker Notes:**
- Explain the core issue: If Client A has digits 0-4 and Client B has digits 5-9, without cross-client repulsion, they will map all digits to the center, destroying the global structure.

---

## Slide 3: The FEDNE Baseline
**Title:** Surrogate-Assisted Repulsion (FEDNE)

**Content:**
- **Mechanism:** FEDNE learns a small multi-layer perceptron (MLP) "surrogate" model to approximate the repulsion density of each client.
- **Communication:** Exchanges encoder weights + surrogate models each round.
- **Overhead:** ~2.27 MB encoder + ~1 KB surrogate per client.
- **Performance:** Achieves strong topological preservation but adds complexity.

**Speaker Notes:**
- This is the baseline we are comparing against.
- Note that our baseline is a faithful re-implementation, not the exact original code, but it captures the surrogate mechanism accurately.

---

## Slide 4: The FEDANCHOR Proposal
**Title:** Anchor-Based Repulsion (FEDANCHOR)

**Content:**
- **Mechanism:** Instead of learning an MLP, each client learns $K$ (e.g., 5) 2D "anchor points" that represent the spatial distribution of their embeddings.
- **Communication:** Exchanges encoder weights + $K$ anchors (just 10 floats = 40 bytes).
- **Goal:** Achieve similar cross-client repulsion with a drastically simpler, more interpretable, and computationally lighter mechanism.

**Speaker Notes:**
- Explain the intuition: If Client A tells Client B "My data is located at these 5 coordinates," Client B can repel its embeddings from those coordinates.

---

## Slide 5: Mathematical Formulation
**Title:** FEDANCHOR Loss Objective

**Content:**
- **Attraction Loss:** Pulls local high-dimensional neighbors together in 2D.
- **Anchor Coverage Loss:** $L_{anchor} = \frac{1}{N} \sum_{i} \min_{j} \|z_i - a_j\|^2$
- **Anchor Repulsion Loss:** $L_{rep} = \sum_{m \neq m'} \sum_{i, j} \log\left(1 + \frac{1}{\|z_i^{(m)} - a_j^{(m')}\|^2 + \epsilon}\right)$

**Speaker Notes:**
- The coverage loss acts like a differentiable K-Means, forcing anchors to spread out and cover the embeddings.
- The repulsion loss forces local embeddings away from other clients' anchors.

---

## Slide 6: Experimental Setup (Controlled Protocol)
**Title:** Rigorous Benchmark Protocol

**Content:**
- **Dataset:** MNIST (60k Train, 10k Test, IID Partition)
- **Architecture:** 784 $\rightarrow$ 512 $\rightarrow$ 256 $\rightarrow$ 128 $\rightarrow$ 2
- **Metrics Evaluated:**
  - kNN Accuracy ($k=5$, measured identically for both methods)
  - Trustworthiness & Continuity
- **Integrity Note:** Previous evaluation protocols for kNN were mismatched; we unified them for an apples-to-apples comparison.

**Speaker Notes:**
- Emphasize that you identified a protocol mismatch in earlier iterations and corrected it. This demonstrates scientific rigor and maturity.

---

## Slide 7: Quantitative Results (Trustworthiness & Continuity)
**Title:** Topological Preservation

**Content:**
- *(Insert Chart: Trustworthiness vs Round)*
- *(Insert Chart: Continuity vs Round)*
- **FEDNE Final T/C:** [Insert Controlled T] / [Insert Controlled C]
- **FEDANCHOR Final T/C:** [Insert Controlled T] / [Insert Controlled C]

**Speaker Notes:**
- Honestly report the results.
- FEDANCHOR preserves topology but currently underperforms the richer surrogate mechanism of FEDNE.

---

## Slide 8: Quantitative Results (Classification Accuracy)
**Title:** kNN Accuracy Comparison

**Content:**
- *(Insert Chart: kNN Accuracy vs Round)*
- **FEDNE Accuracy:** [Insert Controlled kNN]%
- **FEDANCHOR Accuracy:** [Insert Controlled kNN]%

**Speaker Notes:**
- Point out the trajectory of learning.
- Discuss how well the classes are separated.
- Acknowledge any plateauing behavior observed.

---

## Slide 9: Qualitative Results
**Title:** Embedding Visualizations

**Content:**
- *(Insert FEDANCHOR `embeddings_with_anchors.png`)*
- Highlight the anchor positions relative to the data clusters.

**Speaker Notes:**
- Show that the encoder *is* successfully learning a 2D structure.
- Point out the star markers (anchors) and how they position themselves.

---

## Slide 10: Communication and Complexity
**Title:** Communication Overhead Audit

**Content:**
- **Encoder:** 2,265,608 bytes per client.
- **FEDNE Surrogate:** 1,028 bytes per client.
- **FEDANCHOR Anchors:** 40 bytes per client.
- **Conclusion:** While anchors are mathematically smaller, the encoder dominates the bandwidth. FEDANCHOR's true benefit is algorithmic simplicity, not necessarily network savings.

**Speaker Notes:**
- This is a very strong, mature point to make. It shows you audited your own claims critically.

---

## Slide 11: Scientific Limitations (Honest Disclosure)
**Title:** Identified Limitations

**Content:**
- **Anchor Collapse:** We observed that after early rounds, embeddings heavily cluster around 1 or 2 anchors, leaving others unutilized.
- **Repulsion Dominance:** Without mini-batching, full-batch attraction gradients become extremely small, allowing repulsion to dominate.
- **Historical Data Correction:** Earlier claims (e.g., 66.31% accuracy) were found to be unverified under the strict controlled protocol and have been corrected.

**Speaker Notes:**
- Do not shy away from this slide. Panels reward students who critically evaluate their own work and honestly report flaws.

---

## Slide 12: Conclusion & Future Work
**Title:** Summary & Next Steps

**Content:**
- **Summary:** FEDANCHOR is a working prototype that successfully executes federated dimensionality reduction using an interpretable anchor mechanism.
- **Future Work 1:** Introduce entropy regularization to prevent anchor collapse.
- **Future Work 2:** Implement mini-batch local training to restore the attraction gradient magnitude.
- **Conclusion:** Anchor-based federated learning is computationally elegant but requires advanced balancing mechanisms to match surrogate performance.

**Speaker Notes:**
- End on a forward-looking note. You have identified exactly how to improve the prototype for the next phase of research.
- Thank the panel and open for questions.
