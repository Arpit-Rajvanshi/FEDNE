# FEDANCHOR V6 — Viva Preparation

This document anticipates questions regarding the scientific and architectural decisions in V6.

### Q: Why did you switch from client-specific anchors in V5 to shared global anchors in V6?
**A:** In V5, every client learned its own anchor geometry independently. While this allowed local flexibility, it meant that "Anchor 1" on Client A had no geometric relationship to "Anchor 1" on Client B. This lack of a shared reference frame could hurt cross-client consistency during federated aggregation of the encoder. V6 enforces a shared global geometry so that all clients embed their local data relative to the exact same reference points, theoretically improving consistency.

### Q: Why use the Hungarian Algorithm for anchor aggregation?
**A:** Anchors are permutation-symmetric. If Client A's anchors are [top-left, bottom-right] and Client B's are [bottom-right, top-left], a naive average computes the midpoint for both, collapsing the geometry. The Hungarian algorithm solves the minimum-cost bipartite matching problem based on Euclidean distance, allowing us to align Client B's anchors to the global reference ordering before averaging.

### Q: Does the server use labels to initialize the anchors?
**A:** No. Initialization is purely geometric and label-free. The server runs the initial untrained global encoder on the clients' data to get initial embeddings, then fits a standard KMeans clustering algorithm to find representative geometric centers.

### Q: Are the anchors class prototypes?
**A:** No. Anchors are low-dimensional geometric reference points. They act as "pegs" in the embedding space to encourage structural spread and preserve local topology via soft assignment. They do not represent classes, and their number (K) does not have to equal the number of classes.

### Q: What is the risk of "Client Domination" in V6?
**A:** Because all clients update the same global anchor set, a client with a disproportionately large amount of data (or wildly different local geometry) could pull the global anchors away from positions that are optimal for other clients. We detect this by monitoring alignment costs and sample weight fractions.

### Q: How did you ensure a fair comparison between V5 and V6?
**A:** We ran a strictly controlled ablation. The dataset, data partition (IID), random seed, network architecture, optimizer, hyperparameter settings, and evaluation protocols were identical. The only changing variable was the architectural flow of the anchors (client-specific vs. shared global).
