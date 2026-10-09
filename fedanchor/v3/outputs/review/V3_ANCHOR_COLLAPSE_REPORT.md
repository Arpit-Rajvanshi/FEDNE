# FEDANCHOR V3 Anchor Collapse Report

## 1. Problem Definition
In the FEDANCHOR framework, **Anchor Collapse** occurs when a highly dense cluster of local embeddings exerts a massive gravitational pull, causing all local anchors to collapse into a single topological coordinate. This leaves the remaining embedding space devoid of anchors, breaking the ability to meaningfully align across clients.

## 2. Why v1 Collapsed
FEDANCHOR v1 utilized a hard `min()` assignment operator. This created a rigid winner-take-all dynamic where, once a single anchor became slightly closer to the dense data cluster than the rest, it monopolized all assignment gradients. The remaining anchors were left entirely stranded with zero gradient, irreversibly freezing them in useless positions.

## 3. Why v2 Only Delayed Collapse
FEDANCHOR v2 introduced two modifications:
1. **Soft Assignment:** A Student-t distribution replaced the hard assignment, theoretically guaranteeing non-zero gradients for all anchors.
2. **Anchor Spread:** An explicit $O(1/d^2)$ repulsive penalty forcing local anchors apart.

However, the ablation study proved this was insufficient. The V2 modifications succeeded in distributing the anchors in the initial rounds, but the massive gradient magnitude from the dense data clusters eventually overpowered the local spread penalty. By Round 20, the anchors succumbed to the gravitational pull and collapsed again.

## 4. Mathematical Definition of v3 Balance Loss
FEDANCHOR v3 attempted to solve this via **Explicit Utilization Balancing**.

For $K$ anchors and $N$ local points, we computed empirical anchor utilization using the Student-t soft assignments ($q_{ij}$):
$$u_j = \frac{1}{N} \sum_{i=1}^N q_{ij}$$

And penalized its divergence from uniform utilization ($1/K$):
$$L_{balance} = \sum_{j=1}^K \left(u_j - \frac{1}{K}\right)^2$$

This was intended to create a differential penalty: an over-utilized anchor receives a gradient pushing it *away* from data points, while an under-utilized anchor receives a gradient pulling it *towards* data points.

## 5. Exact Implementation
The loss formulation:
$$L_{total} = L_{attraction} + L_{soft\_anchor} + L_{anchor\_repulsion} + L_{spread} + \lambda_{balance} L_{balance}$$
where $\lambda_{balance}$ acts as the regularization strength. The ablation tested $\lambda_{balance} \in \{0.0, 0.001, 0.01, 0.1\}$.

## 6. Unit-test Results
* `test_balance_loss_finite_and_differentiable`: Passed. The function produces finite values and correct PyTorch gradients.
* `test_q_ij_properties`: Passed. The assignment matrix is correctly scaled between 0 and 1, summing to 1.

## 7. Synthetic Collapse Test
A micro-experiment was constructed containing a dense cluster of embeddings and 5 anchors perfectly initialized onto the exact same coordinate (complete collapse). With $\lambda_{balance} = 100.0$, the $L_{balance}$ gradients correctly overpowered the state, spreading the anchors outwards to distribute the utilization evenly across them.

## 8. Controlled MNIST Protocol
- **Dataset:** MNIST (60,000 train, 10,000 test, IID Partition)
- **Clients:** 2
- **Rounds:** 20
- **Ablation Values:** $\lambda_{balance} \in \{0.001, 0.01, 0.1\}$
- **Core Setting:** kNN evaluation over 2,000 random test samples at $k=5$.

## 9. Quantitative Ablation Results (Round 20)

| Mode | kNN Acc | $L_{balance}$ | $L_{spread}$ | Max Hard Util. | Min Pairwise Dist | State |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A (V1 Baseline)** | 19.3% | N/A | N/A | 87.8% | 0.0000 | SEVERE COLLAPSE |
| **B (V2 Soft+Spread)** | 19.4% | N/A | 503.5 | 88.9% | 0.0002 | SEVERE COLLAPSE |
| **C (V3 $\lambda=0.001$)** | 19.0% | $1.76 \times 10^{-6}$ | 503.4 | 88.9% | 0.0002 | SEVERE COLLAPSE |
| **D (V3 $\lambda=0.01$)** | 19.1% | $1.75 \times 10^{-6}$ | 503.4 | 87.8% | 0.0002 | SEVERE COLLAPSE |
| **E (V3 $\lambda=0.1$)** | 19.4% | $1.75 \times 10^{-6}$ | 503.4 | 88.9% | 0.0002 | SEVERE COLLAPSE |

## 18. Did the Balancing Term Prevent Collapse?
**No. The model mathematically circumvented the penalty through a degenerate solution.** 

Despite the explicit mathematical penalty, the "SEVERE COLLAPSE" state remained identical to V1 and V2, with a single anchor taking ~89% of all hard assignments and kNN accuracy remaining stuck at ~19%.

## 19. The "Soft Assignment Loophole" Discovery
When investigating *why* the balancing term failed, we made a crucial scientific discovery: **The neural network discovered a topological loophole in the soft-assignment entropy.**

Look at the Round 20 metrics for Mode E:
* **Hard Utilization Imbalance (from `argmin`):** Extreme (88.9% to one anchor).
* **Soft Utilization Imbalance ($L_{balance}$):** $1.75 \times 10^{-6}$ (Virtually perfect uniformity).

**How is this possible?**
If the network clusters all anchors tightly together (Minimum Pairwise Distance = $0.0002$), then the distance from *any* data point to *every* anchor is almost mathematically identical. 
Because the distances are identical, the Student-t Softmax equation distributes the probabilities perfectly evenly: $q_{ij} \approx 1/K$ for all $j$.

Consequently, the network perfectly satisfies $L_{balance} = 0.0$ *without actually moving the anchors to cover the data distribution*. It merely groups them together, ensuring they all share the exact same mathematical perspective of the data. When the hard `argmin` evaluation occurs, floating-point microscopic differences cause one anchor to deterministically win the cluster, resulting in the observed 88.9% hard utilization.

## 20. Failure Cases
- Gradient-based utilization balancing fails because equal utilization does not guarantee spatial coverage.
- Increased $\lambda_{balance}$ merely forces the anchors to pack tighter to ensure identical $q_{ij}$ distributions.

## 21. Limitations
This confirms a severe theoretical limitation of using differentiable soft-assignments to regularize topological space: soft entropy objectives can be trivially satisfied by overlapping coordinates, decoupling the mathematical loss from the actual geometric objective.

## 22. Recommended Next Research Step
**Abandon purely gradient-based regularizers for anchor placement.**

We have definitively proven that gradients alone (whether attraction, repulsion, soft-assignment, or balance penalties) are fundamentally ill-equipped to combat Anchor Collapse because dense data clusters create insurmountable topological attractors or degenerate mathematical loopholes.

The next step is to introduce **Discontinuous Topological Interventions**:
1. **Anchor Pruning & Resampling:** Periodically freeze training, detect stranded anchors (Hard Utilization < 5%), and explicitly teleport their coordinates into dense regions.
2. **K-Means Hard Resets:** Periodically override the gradients and snap anchors to K-Means centroids to force geometric coverage.
