# SCIENTIFIC INTERPRETATION

## 1. Does FEDANCHOR Outperform FEDNE?
**Short Answer:** No, not in the current prototype configuration. 

**Detailed Explanation:** Under a rigorously controlled and mathematically identical evaluation protocol, FEDNE achieves a kNN classification accuracy of **91.95%** at Round 20, while FEDANCHOR plateaus at **20.50%**. FEDNE also achieves significantly higher Trustworthiness (0.937 vs 0.625) and Continuity (0.958 vs 0.764). 

## 2. Why Does FEDNE Outperform FEDANCHOR?
FEDNE's surrogate model is a Multi-Layer Perceptron (`Linear(2, 64) -> ReLU -> Linear(64, 1)`). This allows it to learn a continuous, non-linear, and highly nuanced density map of the embedding space. When Client B queries Client A's surrogate, it gets a smooth gradient directing its embeddings away from Client A's specific clusters.

FEDANCHOR's mechanism is fundamentally simpler, attempting to represent that same density map using only $K=5$ discrete 2D points (anchors). While mathematically elegant, this drastic reduction in capacity struggles to capture the complex topology of MNIST classes. 

## 3. The "Anchor Collapse" Phenomenon
The primary reason for FEDANCHOR's underperformance is an optimization pathology we term "Anchor Collapse." 
- The Anchor Coverage loss uses a `min()` operator, similar to K-Means.
- During early rounds, if one anchor gets slightly closer to the data centroid than the others, it "wins" the gradients for all nearby embeddings.
- The "losing" anchors receive zero gradients and remain stranded in empty space.
- As observed in our diagnostics, by Round 20, often 99% of a client's embeddings are clustered around a single anchor.
- Because the anchors collapse to a single point, the cross-client repulsion signal becomes a single point source, pushing all of Client B's embeddings away from the center but failing to separate the individual digit classes.

## 4. The 66.31% Discrepancy Resolved
Previous project documentation cited an unverified historical run claiming FEDANCHOR achieved 66.31% kNN accuracy at Round 20. 
Our rigorous forensic audit confirms that **this number cannot be reproduced** from any saved artifact, CSV, or model checkpoint in the repository. The highest reproducibly recorded number under the proper benchmark protocol is ~20-22%. In accordance with academic integrity, the unverified 66.31% claim has been completely removed from all final presentation materials. 

## 5. Communication Overhead vs Algorithmic Simplicity
The mathematical communication savings of FEDANCHOR's cross-client signal are real: transmitting 10 anchors requires only 40 bytes, whereas FEDNE's surrogate requires 1,028 bytes (a 96% reduction in cross-client overhead).

However, the neural network *encoder* requires 2,265,608 bytes per client to transmit. Therefore, the *overall* bandwidth reduction of FEDANCHOR compared to FEDNE is a statistically negligible 0.04%. 

**Conclusion:** FEDANCHOR's primary contribution is not practical bandwidth savings over FEDNE, but rather algorithmic simplicity. Replacing a black-box MLP surrogate with transparent 2D coordinates is highly desirable for interpretability and edge-device computation, provided the anchor collapse issue can be resolved in future iterations (e.g., via entropy regularization).
