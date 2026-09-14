import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, Dict, Any, List

def compute_trustworthiness_and_continuity(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k: int = 7
) -> Tuple[float, float]:
    """
    Compute Trustworthiness and Continuity of low-dimensional embedding X_low
    with respect to high-dimensional data X_high.
    """
    n = X_high.shape[0]
    if n <= k + 1:
        return 1.0, 1.0
        
    D_high = pairwise_distances(X_high, metric="euclidean")
    D_low = pairwise_distances(X_low, metric="euclidean")
    
    ranks_high = np.argsort(np.argsort(D_high, axis=1), axis=1)
    ranks_low = np.argsort(np.argsort(D_low, axis=1), axis=1)
    
    nbrs_high = np.argsort(D_high, axis=1)[:, 1:k+1]
    nbrs_low = np.argsort(D_low, axis=1)[:, 1:k+1]
    
    sum_t = 0.0
    for i in range(n):
        high_nbrs_set = set(nbrs_high[i])
        for j in nbrs_low[i]:
            if j not in high_nbrs_set:
                r = ranks_high[i, j]
                sum_t += (r - k)
                
    sum_c = 0.0
    for i in range(n):
        low_nbrs_set = set(nbrs_low[i])
        for j in nbrs_high[i]:
            if j not in low_nbrs_set:
                r_hat = ranks_low[i, j]
                sum_c += (r_hat - k)
                
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

def compute_anchor_diagnostics(
    embeddings: np.ndarray,
    anchors: np.ndarray
) -> Dict[str, Any]:
    """
    Compute detailed anchor geometric & utilization diagnostics.
    
    Returns:
        mean_min_distance: Average distance to closest anchor
        max_min_distance: Maximum distance to closest anchor
        anchor_utilization: List of percentages [%] of embeddings assigned to anchor 0..K-1
    """
    N = embeddings.shape[0]
    K = anchors.shape[0]
    if N == 0 or K == 0:
        return {
            "mean_min_distance": 0.0,
            "max_min_distance": 0.0,
            "anchor_utilization": [0.0] * K
        }
        
    D = pairwise_distances(embeddings, anchors, metric="euclidean")
    nearest_anchor_idx = np.argmin(D, axis=1)
    min_dists = np.min(D, axis=1)
    
    mean_dist = float(np.mean(min_dists))
    max_dist = float(np.max(min_dists))
    
    counts = np.bincount(nearest_anchor_idx, minlength=K)
    utilization_pct = [(float(c) / float(N)) * 100.0 for c in counts]
    
    return {
        "mean_min_distance": mean_dist,
        "max_min_distance": max_dist,
        "anchor_utilization": utilization_pct
    }

def compute_metrics(
    X_high_train: np.ndarray,
    X_low_train: np.ndarray,
    y_train: np.ndarray,
    X_high_test: np.ndarray,
    X_low_test: np.ndarray,
    y_test: np.ndarray,
    client_anchors: Dict[int, torch.Tensor],
    k: int = 7
) -> Dict[str, Any]:
    """
    Computes all standard evaluation metrics and anchor diagnostics.
    """
    if X_high_test.shape[0] > 2000:
        sub_idx = np.random.choice(X_high_test.shape[0], 2000, replace=False)
        X_h_eval, X_l_eval = X_high_test[sub_idx], X_low_test[sub_idx]
    else:
        X_h_eval, X_l_eval = X_high_test, X_low_test
        
    trustworthiness, continuity = compute_trustworthiness_and_continuity(X_h_eval, X_l_eval, k=k)
    knn_acc = compute_knn_accuracy(X_low_train, y_train, X_low_test, y_test, k=k)
    
    all_anc_np = []
    for anc in client_anchors.values():
        all_anc_np.append(anc.numpy())
        
    if len(all_anc_np) > 0:
        concat_anc = np.concatenate(all_anc_np, axis=0)
        anc_diag = compute_anchor_diagnostics(X_low_train, concat_anc)
    else:
        anc_diag = {"mean_min_distance": 0.0, "max_min_distance": 0.0, "anchor_utilization": []}
        
    return {
        "trustworthiness": trustworthiness,
        "continuity": continuity,
        "knn_accuracy": knn_acc,
        "anchor_coverage": anc_diag["mean_min_distance"],
        "anchor_max_distance": anc_diag["max_min_distance"],
        "anchor_utilization": anc_diag["anchor_utilization"]
    }
