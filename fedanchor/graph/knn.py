import numpy as np
from sklearn.neighbors import NearestNeighbors

def construct_knn_graph(X: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Construct a local kNN graph for client dataset X of shape [N, D] using sklearn NearestNeighbors.
    
    Returns:
        edges: np.ndarray of shape [N * k, 2] containing neighbor pairs (i, j) where j is in NN_k(i).
    """
    N = X.shape[0]
    k_query = min(k + 1, N)
    
    nbrs = NearestNeighbors(n_neighbors=k_query, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    refined_indices = []
    for i in range(N):
        row = indices[i]
        # Exclude self-loops
        filtered = row[row != i]
        
        if len(filtered) > k:
            filtered = filtered[:k]
        elif len(filtered) < k:
            pad_val = row[0] if len(row) > 0 else i
            filtered = np.pad(filtered, (0, k - len(filtered)), mode='constant', constant_values=pad_val)
            
        refined_indices.append(filtered)
        
    refined_indices = np.array(refined_indices, dtype=np.int64)
    
    src = np.repeat(np.arange(N), k)
    dst = refined_indices.flatten()
    edges = np.stack([src, dst], axis=1)
    
    return edges
