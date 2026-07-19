import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from typing import Tuple, Optional

# Attempt to import snc library
try:
    from snc.snc import SNC
    _HAS_SNC = True
except ImportError:
    _HAS_SNC = False

def compute_trustworthiness_and_continuity(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k: int = 7
) -> Tuple[float, float]:
    """
    Compute Trustworthiness and Continuity of a low-dimensional embedding X_low
    with respect to high-dimensional data X_high.
    
    Trustworthiness measures how well nearest-neighbor relationships are preserved
    from low-dim to high-dim (i.e. avoids false neighbors).
    Continuity measures how well nearest-neighbor relationships are preserved
    from high-dim to low-dim (i.e. avoids tearing apart true neighbors).
    """
    n = X_high.shape[0]
    if n <= k + 1:
        return 1.0, 1.0
        
    # Compute pairwise Euclidean distance matrices
    D_high = pairwise_distances(X_high, metric="euclidean")
    D_low = pairwise_distances(X_low, metric="euclidean")
    
    # Get ranks of points (0 is self, 1 is nearest neighbor, etc.)
    # np.argsort(D, axis=1) gives indices of sorted distances.
    # Sorting again gives the rank of each point.
    ranks_high = np.argsort(np.argsort(D_high, axis=1), axis=1)
    ranks_low = np.argsort(np.argsort(D_low, axis=1), axis=1)
    
    # Neighborhood sets (excluding the point itself)
    nbrs_high = np.argsort(D_high, axis=1)[:, 1:k+1]
    nbrs_low = np.argsort(D_low, axis=1)[:, 1:k+1]
    
    # Trustworthiness summation
    sum_t = 0.0
    for i in range(n):
        high_nbrs_set = set(nbrs_high[i])
        for j in nbrs_low[i]:
            if j not in high_nbrs_set:
                r = ranks_high[i, j]
                sum_t += (r - k)
                
    # Continuity summation
    sum_c = 0.0
    for i in range(n):
        low_nbrs_set = set(nbrs_low[i])
        for j in nbrs_high[i]:
            if j not in low_nbrs_set:
                r_hat = ranks_low[i, j]
                sum_c += (r_hat - k)
                
    # Normalization factor
    norm = n * k * (2 * n - 3 * k - 1)
    t_val = 1.0 - (2.0 / norm) * sum_t
    c_val = 1.0 - (2.0 / norm) * sum_c
    
    return float(t_val), float(c_val)

def compute_knn_accuracy(
    X_train_low: np.ndarray,
    y_train: np.ndarray,
    X_test_low: np.ndarray,
    y_test: np.ndarray,
    k: int = 7
) -> float:
    """
    Compute kNN classification accuracy of low-dimensional embedding.
    """
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train_low, y_train)
    score = clf.score(X_test_low, y_test)
    return float(score)

def compute_steadiness_cohesiveness(
    X_high: np.ndarray,
    X_low: np.ndarray,
    num_classes: int = 10,
    k: int = 7
) -> Tuple[float, float, bool]:
    """
    Compute Steadiness and Cohesiveness.
    Tries to use the 'snc' package if available, else falls back to KMeans clustering
    consistency (Normalized Mutual Information and Adjusted Rand Index).
    
    Returns:
        steadiness: float
        cohesiveness: float
        using_snc: bool (True if official library was used, False if fallback was used)
    """
    if _HAS_SNC:
        try:
            # Configure SNC parameters
            # Use snc package to compute metrics
            metrics = SNC(raw=X_high, emb=X_low, iteration=100)
            metrics.fit()
            return float(metrics.steadiness()), float(metrics.cohesiveness()), True
        except Exception:
            pass
            
    # Fallback clustering consistency metrics
    kmeans_high = KMeans(n_clusters=num_classes, random_state=42, n_init='auto').fit(X_high)
    kmeans_low = KMeans(n_clusters=num_classes, random_state=42, n_init='auto').fit(X_low)
    
    # NMI represents how clean/unmixed the clusters are (Steadiness proxy)
    nmi = normalized_mutual_info_score(kmeans_high.labels_, kmeans_low.labels_)
    # ARI represents the overall partition alignment (Cohesiveness proxy)
    ari = adjusted_rand_score(kmeans_high.labels_, kmeans_low.labels_)
    
    return float(nmi), float(ari), False
