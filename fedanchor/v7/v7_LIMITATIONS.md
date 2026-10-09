# FEDANCHOR v7 — Limitations

The shared global anchor design introduced in v7 provides a common reference frame, but introduces several theoretical and practical limitations:

## 1. Assumption of Common Optimal Geometry
v7 forces all clients to use the exact same anchor geometry. If clients have highly non-IID data with disjoint feature distributions, the optimal low-dimensional manifold for Client A might be fundamentally incompatible with Client B. Forcing them to share anchors could degrade local embedding quality compared to the flexibility of V5.

## 2. Sensitivity of Hungarian Alignment
The Hungarian alignment algorithm matches anchors based on Euclidean distance. If local optimization causes a client's anchors to move significantly (e.g., rotating or shifting dramatically), the Euclidean distance may yield an incorrect permutation assignment. If anchors are misaligned and then averaged, it can lead to catastrophic geometric collapse on the server.

## 3. Communication Overhead
While small (K points in 2D), broadcasting and uploading the anchor matrix introduces a new synchronization dependency. The server must wait for all client anchor updates to perform the Hungarian alignment before producing the new global anchor matrix for the next round.

## 4. Initialization Bias
The global anchors are initialized using KMeans on the initial embeddings. If the initial encoder state produces a highly skewed projection, the initial anchors might be poorly placed. While they are trainable and can move, overcoming a very poor initialization in a federated setting is challenging.

## 5. Potential for Anchor "Tug-of-War"
During local training, Client A might pull an anchor left to fit its local data manifold, while Client B pulls the same anchor right. The resulting server average might leave the anchor in a sub-optimal middle ground that serves neither client well.

## 6. Lack of Embedding Repulsion (Fundamental Mathematical Flaw)
By removing the cross-client anchor repulsion (`L_rep`) that was present in V5 (because all clients now share the same anchors), v7 inadvertently removed the only volume-expanding repulsive force acting on the data embeddings. Without this force, the local data embeddings completely collapse to a single point (e.g. the origin). Once collapsed, the geometric distance to all surrounding anchors becomes identical, trivially minimizing the anchor balance loss but destroying the representation learning capability.
