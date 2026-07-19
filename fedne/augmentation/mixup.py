import numpy as np
from typing import Tuple
from sklearn.neighbors import NearestNeighbors

def mixup_dataset(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    alpha: float = 0.2,
    mixup_ratio: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform intra-client data mixing (Mixup with neighbors).
    For each selected sample, interpolates with one of its k-nearest neighbors.
    Returns:
        X_augmented, y_augmented: Original data concatenated with mixed samples.
    """
    if mixup_ratio <= 0.0:
        return X, y
        
    N = X.shape[0]
    num_mixed = int(N * mixup_ratio)
    if num_mixed == 0:
        return X, y
        
    # Find neighbors on the original dataset
    k_query = min(k + 1, N)
    nbrs = NearestNeighbors(n_neighbors=k_query, algorithm='auto').fit(X)
    _, indices = nbrs.kneighbors(X)
    
    # Select indices to mix
    mix_indices = np.random.choice(N, size=num_mixed, replace=False)
    
    X_mixed = []
    y_mixed = []
    
    for idx in mix_indices:
        row = indices[idx]
        # Exclude self if possible
        neighbors = row[row != idx]
        if len(neighbors) == 0:
            neighbor_idx = idx
        else:
            neighbor_idx = np.random.choice(neighbors)
            
        # Sample lambda from Beta(alpha, alpha)
        lam = np.random.beta(alpha, alpha)
        
        x_i = X[idx]
        x_j = X[neighbor_idx]
        x_new = lam * x_i + (1 - lam) * x_j
        
        # Label is determined by the dominant parent
        y_new = y[idx] if lam >= 0.5 else y[neighbor_idx]
        
        X_mixed.append(x_new)
        y_mixed.append(y_new)
        
    X_mixed = np.array(X_mixed, dtype=np.float32)
    y_mixed = np.array(y_mixed, dtype=np.int64)
    
    X_augmented = np.concatenate([X, X_mixed], axis=0)
    y_augmented = np.concatenate([y, y_mixed], axis=0)
    
    return X_augmented, y_augmented
