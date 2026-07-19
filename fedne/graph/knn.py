import numpy as np
from sklearn.neighbors import NearestNeighbors

def construct_knn_graph(X: np.ndarray, k: int) -> np.ndarray:
    """
    Construct a local kNN graph for dataset X of shape [N, D] using sklearn NearestNeighbors.
    Returns:
        edges: np.ndarray of shape [N * k, 2] containing neighbor pairs (i, j) where j in NN_k(i).
    """
    N = X.shape[0]
    # We want exactly k neighbors, excluding the point itself.
    # So we query for k + 1 neighbors.
    k_query = min(k + 1, N)
    
    nbrs = NearestNeighbors(n_neighbors=k_query, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    refined_indices = []
    for i in range(N):
        row = indices[i]
        # Filter out self-loops (where row element matches the index i)
        filtered = row[row != i]
        
        # If we have more than k neighbors, keep the first k
        if len(filtered) > k:
            filtered = filtered[:k]
        # If we have fewer than k neighbors (due to duplicate points or small N), pad it
        elif len(filtered) < k:
            pad_val = row[0] if len(row) > 0 else i
            filtered = np.pad(filtered, (0, k - len(filtered)), mode='constant', constant_values=pad_val)
            
        refined_indices.append(filtered)
        
    refined_indices = np.array(refined_indices, dtype=np.int64)
    
    # Generate edge list
    src = np.repeat(np.arange(N), k)
    dst = refined_indices.flatten()
    edges = np.stack([src, dst], axis=1)
    
    return edges
