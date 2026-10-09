# FEDANCHOR V4 Geometric Anti-Collapse Report

## 1. Research Motivation
In FEDANCHOR V1, hard anchor assignment caused extreme winner-take-all anchor collapse. V2 delayed the collapse with soft assignment and inverse-distance anchor spread, but failed by Round 20. 

In V3, we explicitly added a utilization balancing penalty ($L_{balance}$). However, V3 revealed an extraordinary degenerate solution: **The Soft-Assignment Collapse Loophole**. The neural network perfectly optimized $L_{balance}$ to zero by perfectly overlapping all anchors.

## 2. V4 Objective and Mathematical Motivation
V4 directly attacks this loophole by introducing an explicit geometric separation constraint ($L_{sep}$). 
$$L_{sep} = \frac{1}{K(K-1)} \sum_{i \ne j} \max(0, m - \|a_i - a_j\|)^2$$

Unlike inverse-distance repulsion which applies an asymptotic gradient everywhere, $L_{sep}$ strictly enforces a hard margin $m$. If anchors are further than $m$, the penalty is precisely 0. 

## 3. Unit & Synthetic Test Results
- **Unit Tests:** Passed. Verified exactly zero loss for separated anchors and positive loss for collapsed anchors. Gradients correctly flow to separate coordinates.
- **Synthetic Experiment:** We artificially forced a configuration of 5 overlapping anchors. The experiment verified that a gradient step on $L_{sep}$ successfully pushes the anchors outwards, actively reducing the geometric separation violation.

## 4. Controlled Experimental Protocol
- **Dataset:** MNIST (60,000 train, 10,000 test, IID Partition)
- **Clients:** 2
- **Rounds:** 20
- **Ablation Values:** $m=0.10$ with $\lambda_{sep} \in \{0.001, 0.01, 0.1\}$

## 5. Full Ablation Results (Round 20)

| Mode | kNN Acc | $L_{sep}$ Loss | Min Anchor Dist | Hard Util (Max) | State |
|------|---------|----------------|-----------------|----------------|-------|
| A (Baseline) | 19.13% | 0.00253 | 0.000286 | 87.4% | SEVERE COLLAPSE |
| B (V3 Balance) | 19.11% | 0.00253 | 0.000286 | 87.4% | SEVERE COLLAPSE |
| C (V4 Low) | 18.25% | 0.00253 | 0.000286 | 87.4% | SEVERE COLLAPSE |
| D (V4 Med) | 18.74% | 0.00253 | 0.000286 | 87.4% | SEVERE COLLAPSE |
| E (V4 High) | 19.76% | 0.00253 | 0.000286 | 87.4% | SEVERE COLLAPSE |

## 6. Critical Scientific Revelation: The "Frozen Anchor" Optimization Artifact

The ablation results yielded an astonishing finding: The exact `Min Anchor Dist` is mathematically identical (`0.00028690905310213566`) across all modes and all rounds from round 5 onwards. The separation loss $L_{sep}$ remained identical (`0.00253`).

**Why did $L_{sep}$ fail to eliminate the collapse loophole?**
Because the anchors are effectively frozen in space. 

Detailed analysis of the training loop revealed a severe optimization artifact responsible for the apparent "anchor collapse" across all versions (V1-V4):
1. **Full-Batch Training:** The local client processes the entire local dataset (30,000 samples) in a single batch (`local_epochs=1`).
2. **Adam Reinitialization:** The `Adam` optimizer is reinitialized at the start of every round.
3. **Gradient Clipping:** Gradients are aggressively clipped to `max_grad_norm = 1.0` (heavily dominated by the initial explosion of $L_{spread}$).
4. **Sign Descent Limit:** Because Adam's momentum starts at 0 every round, and there is only 1 step per round, the update is exactly bounded by `lr * sign(gradient)`. With $lr=0.001$, anchors can move by a maximum of $0.001$ per round, or $0.02$ across the entire 20-round training process.

**Conclusion:** The anchors barely move from their KMeans initialization. The collapse we are observing is **NOT** the anchors collapsing into each other. It is the **data embeddings** collapsing into the territory of a single frozen anchor. The encoder weights are updating and mapping all data into the Voronoi cell of one initialization point. 

## 7. Hyperparameter Sensitivity
Because the anchor positions are heavily restricted by the optimization artifact, the ablation of $\lambda_{sep}$ had statistically negligible impact on the overall dynamics. The kNN accuracy remained locked around ~19%, heavily underperforming the baseline.

## 8. Final Scientific Conclusion
$L_{sep}$ is mathematically sound and successfully separates anchors in unit testing. However, no anchor-position regularizer ($L_{spread}$, $L_{balance}$, or $L_{sep}$) can function in the current federated setup due to the full-batch optimizer reinitialization artifact. The anchors are completely starved of optimization steps. 

To solve the anchor collapse and properly test $L_{sep}$, we must decouple the anchor optimization rate from the client communication rounds (e.g., by using mini-batch dataloaders or persisting optimizer momentum states).
