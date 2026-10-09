# FEDANCHOR V2 Viva Preparation

This document anticipates panel questions regarding the FEDANCHOR V2 experimental validation.

---

## Q1: Your kNN accuracy is still around 20%, far below the baseline. Is the project a failure?

**Answer:** 
No, the project is a scientific success because it definitively identifies and isolates a mathematical failure mode in federated dimensionality reduction. Our controlled ablation proves that "Anchor Collapse" is not just an artifact of a harsh $min()$ function. Even when we implemented a Student-t-style differentiable soft assignment and an explicit inverse-squared spread penalty, the local gradients from dense clusters eventually overpowered the penalties, leading to stranded anchors and low accuracy. 

A project that hides a 20% accuracy is a failure; a project that exposes it, dissects it, and proves why naive gradient penalties can't fix it is strong experimental science.

---

## Q2: How exactly does your Soft Assignment work? Is it just t-SNE?

**Answer:** 
We use a Student-t-style similarity to construct differentiable soft assignments between embeddings and anchors. It is inspired by the heavy-tailed kernel used in t-SNE and DEC, which prevents the vanishing gradient problem in high-dimensional space. However, we are not implementing t-SNE itself; we are strictly using the kernel to compute assignment probabilities $q_{ij}$ so that every anchor receives some gradient signal, avoiding the absolute "winner-take-all" cutoff of V1.

---

## Q3: Why didn't the Anchor Spread penalty solve the collapse?

**Answer:**
The spread penalty $L_{spread}$ pushes anchors apart, but it doesn't tell them *where* to go. Our ablation showed that while it successfully distributed anchors in the very early rounds, the massive gradient magnitude from the dense data clusters eventually pulled 1 or 2 anchors in, while the remaining anchors were pushed into empty topological space. Once in empty space, they receive near-zero attractive gradients, becoming permanently stranded.

---

## Q4: What is the most important finding of this project?

**Answer:**
The most important finding is that gradient-based penalties are fundamentally insufficient for solving spatial stranding in federated non-convex optimization. Because dimensionality reduction relies on topological layout, if an anchor gets "stuck" in a void, it cannot simply gradient-descent its way out. Future federated anchor systems will likely require discontinuous mechanisms, such as actively pruning stranded anchors and re-sampling them near active data.
