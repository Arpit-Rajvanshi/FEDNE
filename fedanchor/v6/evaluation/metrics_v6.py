"""
V6 Evaluation Metrics
======================

Extends V5 metrics with V6-specific cross-client consistency metrics.

New in V6:
  - Per-client soft utilisation (u^(client)_j)
  - Global soft utilisation (averaged across clients)
  - Anchor assignment distribution distance (L1 / Jensen-Shannon)
  - Mean nearest-anchor distance per client
  - Anchor usage correlation between client pairs
  - Anchor coordinate consistency verification

All standard metrics (trustworthiness, continuity, kNN) are preserved
from V5 with the same evaluation protocol.
"""

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from scipy.special import rel_entr
from scipy.stats import pearsonr
from typing import Dict, Any, List, Tuple, Optional


# -----------------------------------------------------------------------
# Standard metrics (same as V5)
# -----------------------------------------------------------------------

def compute_trustworthiness_and_continuity(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k: int = 7,
) -> Tuple[float, float]:
    """
    Compute Trustworthiness and Continuity.
    Identical implementation to V5 for fair comparison.
    """
    n = X_high.shape[0]
    if n <= k + 1:
        return 1.0, 1.0

    D_high = pairwise_distances(X_high, metric="euclidean")
    D_low = pairwise_distances(X_low, metric="euclidean")

    ranks_high = np.argsort(np.argsort(D_high, axis=1), axis=1)
    ranks_low = np.argsort(np.argsort(D_low, axis=1), axis=1)

    nbrs_high = np.argsort(D_high, axis=1)[:, 1 : k + 1]
    nbrs_low = np.argsort(D_low, axis=1)[:, 1 : k + 1]

    sum_t = 0.0
    for i in range(n):
        high_set = set(nbrs_high[i])
        for j in nbrs_low[i]:
            if j not in high_set:
                sum_t += ranks_high[i, j] - k

    sum_c = 0.0
    for i in range(n):
        low_set = set(nbrs_low[i])
        for j in nbrs_high[i]:
            if j not in low_set:
                sum_c += ranks_low[i, j] - k

    norm = n * k * (2 * n - 3 * k - 1)
    t_val = 1.0 - (2.0 / norm) * sum_t
    c_val = 1.0 - (2.0 / norm) * sum_c
    return float(t_val), float(c_val)


def compute_knn_accuracy(
    X_train_low: np.ndarray,
    y_train: np.ndarray,
    X_test_low: np.ndarray,
    y_test: np.ndarray,
    k: int = 7,
) -> float:
    """kNN classification accuracy (same as V5)."""
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train_low, y_train)
    return float(clf.score(X_test_low, y_test))


# -----------------------------------------------------------------------
# Soft utilisation helpers
# -----------------------------------------------------------------------

def compute_soft_q(
    embeddings: np.ndarray,
    anchors: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Compute Student-t soft assignment matrix q [N, K].

    q_ij = (1 + ||z_i - a_j||² / α)^(-(α+1)/2)
           ----------------------------------------
           Σ_k (1 + ||z_i - a_k||² / α)^(-(α+1)/2)
    """
    D = pairwise_distances(embeddings, anchors, metric="euclidean")
    sq_dists = D ** 2
    num = 1.0 / (1.0 + sq_dists / alpha) ** ((alpha + 1.0) / 2.0)
    denom = np.sum(num, axis=1, keepdims=True) + 1e-12
    return num / denom


def compute_per_client_utilisation(
    client_embeddings: List[np.ndarray],
    global_anchors: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Compute per-client soft utilisation and global utilisation.

    u_j^(client_m) = (1 / N_m) Σ_i q_ij^(m)
    u_j^(global)   = (1 / N) Σ_m Σ_i q_ij^(m)

    Args:
        client_embeddings: list of [N_m, 2] arrays
        global_anchors:    [K, 2] shared global anchors

    Returns:
        per_client_u: list of [K] arrays — per-client soft utilisation
        global_u:     [K] array — global soft utilisation (sample-weighted)
    """
    per_client_u = []
    total_samples = sum(e.shape[0] for e in client_embeddings)

    weighted_sum = np.zeros(global_anchors.shape[0])
    for emb in client_embeddings:
        if emb.shape[0] == 0:
            per_client_u.append(np.zeros(global_anchors.shape[0]))
            continue
        q = compute_soft_q(emb, global_anchors, alpha=alpha)
        u_m = np.mean(q, axis=0)
        per_client_u.append(u_m)
        weighted_sum += u_m * emb.shape[0]

    global_u = weighted_sum / max(total_samples, 1)
    return per_client_u, global_u


# -----------------------------------------------------------------------
# Anchor diagnostics (V6 extended)
# -----------------------------------------------------------------------

def compute_anchor_diagnostics_v6(
    embeddings: np.ndarray,
    global_anchors: np.ndarray,
    client_embeddings: Optional[List[np.ndarray]] = None,
    margin: float = 0.1,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute comprehensive anchor diagnostics for V6.

    Includes all V5 metrics plus new V6 cross-client consistency metrics.
    """
    N = embeddings.shape[0]
    K = global_anchors.shape[0]

    if N == 0 or K == 0:
        return _empty_diagnostics(K)

    # ----- Pairwise anchor geometry -----
    pairwise_anc = pairwise_distances(global_anchors, metric="euclidean")
    anc_dists = pairwise_anc[np.triu_indices(K, k=1)].tolist()
    min_anc_dist = float(np.min(anc_dists)) if anc_dists else 0.0
    mean_anc_dist = float(np.mean(anc_dists)) if anc_dists else 0.0
    max_anc_dist = float(np.max(anc_dists)) if anc_dists else 0.0

    # ----- Soft utilisation (global over all training embeddings) -----
    q = compute_soft_q(embeddings, global_anchors, alpha=alpha)
    u_soft = np.mean(q, axis=0)

    # ----- Hard utilisation -----
    D = pairwise_distances(embeddings, global_anchors, metric="euclidean")
    nearest_idx = np.argmin(D, axis=1)
    counts = np.bincount(nearest_idx, minlength=K)
    u_hard = counts.astype(float) / float(N)

    min_dists = np.min(D, axis=1)
    mean_nearest_dist = float(np.mean(min_dists))
    max_nearest_dist = float(np.max(min_dists))

    # ----- Entropy -----
    eps = 1e-9
    entropy = -np.sum(u_soft * np.log(u_soft + eps))
    normalized_entropy = float(entropy / max(np.log(K), eps))

    max_hard_u = float(np.max(u_hard))
    effective_anchors = int(np.sum(u_hard >= 0.05))

    # ----- Collapse state -----
    collapse_state = _classify_collapse(
        max_hard_u, min_anc_dist, effective_anchors, margin
    )

    result: Dict[str, Any] = {
        "mean_min_distance": mean_nearest_dist,
        "max_min_distance": max_nearest_dist,
        "soft_utilization_frac": u_soft.tolist(),
        "hard_utilization_frac": u_hard.tolist(),
        "normalized_entropy": normalized_entropy,
        "max_hard_utilization": max_hard_u,
        "effective_anchors": effective_anchors,
        "collapse_state": collapse_state,
        "nearest_anchor_distances": min_dists.tolist(),
        "anchor_pairwise_distances": anc_dists,
        "min_pairwise_anchor_distance": min_anc_dist,
        "mean_pairwise_anchor_distance": mean_anc_dist,
        "max_pairwise_anchor_distance": max_anc_dist,
    }

    # ----- V6 NEW: Cross-client consistency metrics -----
    if client_embeddings is not None and len(client_embeddings) > 0:
        per_client_u, global_u = compute_per_client_utilisation(
            client_embeddings, global_anchors, alpha=alpha
        )
        result["per_client_soft_utilization"] = [
            u.tolist() for u in per_client_u
        ]
        result["global_soft_utilization"] = global_u.tolist()

        # Per-client entropy
        client_entropies = []
        for u_m in per_client_u:
            ent = -np.sum(u_m * np.log(u_m + eps))
            client_entropies.append(float(ent / max(np.log(K), eps)))
        result["per_client_entropy"] = client_entropies

        # Per-client mean nearest-anchor distance
        client_mean_dists = []
        for emb_m in client_embeddings:
            if emb_m.shape[0] == 0:
                client_mean_dists.append(0.0)
                continue
            D_m = pairwise_distances(emb_m, global_anchors, metric="euclidean")
            client_mean_dists.append(float(np.mean(np.min(D_m, axis=1))))
        result["per_client_mean_nearest_anchor_dist"] = client_mean_dists

        # Pairwise client utilisation distance (L1 and Jensen-Shannon)
        n_clients = len(per_client_u)
        l1_matrix = []
        js_matrix = []
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                u_i = per_client_u[i] + eps
                u_j = per_client_u[j] + eps
                # L1
                l1 = float(np.sum(np.abs(u_i - u_j)))
                # Jensen-Shannon divergence
                m = 0.5 * (u_i / u_i.sum() + u_j / u_j.sum())
                js = 0.5 * (
                    np.sum(rel_entr(u_i / u_i.sum(), m))
                    + np.sum(rel_entr(u_j / u_j.sum(), m))
                )
                l1_matrix.append({"clients": (i, j), "l1": l1})
                js_matrix.append({"clients": (i, j), "js": float(js)})
        result["pairwise_client_l1_divergence"] = l1_matrix
        result["pairwise_client_js_divergence"] = js_matrix

        # Anchor usage correlation between clients (if ≥ 2 clients)
        if n_clients >= 2:
            corr_list = []
            for i in range(n_clients):
                for j in range(i + 1, n_clients):
                    if (
                        np.std(per_client_u[i]) > 1e-9
                        and np.std(per_client_u[j]) > 1e-9
                    ):
                        r, _ = pearsonr(per_client_u[i], per_client_u[j])
                        corr_list.append({"clients": (i, j), "r": float(r)})
            result["pairwise_client_utilisation_correlation"] = corr_list

    return result


def _classify_collapse(
    max_hard_u: float,
    min_anc_dist: float,
    effective_anchors: int,
    margin: float,
) -> str:
    if max_hard_u >= 0.80 and (min_anc_dist < 1e-5 or min_anc_dist < 0.1 * margin):
        return "SEVERE COLLAPSE"
    elif max_hard_u >= 0.60 or min_anc_dist < 0.5 * margin:
        return "MODERATE COLLAPSE"
    elif max_hard_u < 0.60 and min_anc_dist >= 0.5 * margin and effective_anchors >= 4:
        return "BALANCED"
    else:
        return "MILD COLLAPSE"


def _empty_diagnostics(K: int) -> Dict[str, Any]:
    return {
        "mean_min_distance": 0.0,
        "max_min_distance": 0.0,
        "soft_utilization_frac": [0.0] * K,
        "hard_utilization_frac": [0.0] * K,
        "normalized_entropy": 0.0,
        "max_hard_utilization": 0.0,
        "effective_anchors": 0,
        "collapse_state": "SEVERE COLLAPSE",
        "nearest_anchor_distances": [],
        "anchor_pairwise_distances": [],
        "min_pairwise_anchor_distance": 0.0,
        "mean_pairwise_anchor_distance": 0.0,
        "max_pairwise_anchor_distance": 0.0,
        "per_client_soft_utilization": [],
        "global_soft_utilization": [0.0] * K,
        "per_client_entropy": [],
        "per_client_mean_nearest_anchor_dist": [],
        "pairwise_client_l1_divergence": [],
        "pairwise_client_js_divergence": [],
        "pairwise_client_utilisation_correlation": [],
    }


# -----------------------------------------------------------------------
# Full metric computation (main entry point)
# -----------------------------------------------------------------------

def compute_metrics_v6(
    X_high_train: np.ndarray,
    X_low_train: np.ndarray,
    y_train: np.ndarray,
    X_high_test: np.ndarray,
    X_low_test: np.ndarray,
    y_test: np.ndarray,
    global_anchors: np.ndarray,
    client_embeddings: Optional[List[np.ndarray]] = None,
    margin: float = 0.1,
    k: int = 7,
    eval_subsample: int = 2000,
) -> Dict[str, Any]:
    """
    Compute all V6 evaluation metrics.

    Identical kNN / T&C protocol as V5. New: cross-client metrics.

    Args:
        X_high_train:      [N_train, D] raw features
        X_low_train:       [N_train, 2] training embeddings (all clients combined)
        y_train:           [N_train] training labels
        X_high_test:       [N_test, D] test raw features
        X_low_test:        [N_test, 2] test embeddings
        y_test:            [N_test] test labels
        global_anchors:    [K, 2] current global anchor matrix
        client_embeddings: list of per-client [N_m, 2] embeddings
        margin:            anchor separation margin (for collapse classifier)
        k:                 k-NN parameter
        eval_subsample:    max test samples for T&C (same as V5 default 2000)

    Returns:
        dict with all metrics
    """
    # Subsample for T&C (expensive O(n²))
    if X_high_test.shape[0] > eval_subsample:
        sub_idx = np.random.choice(
            X_high_test.shape[0], eval_subsample, replace=False
        )
        X_h_eval = X_high_test[sub_idx]
        X_l_eval = X_low_test[sub_idx]
    else:
        X_h_eval, X_l_eval = X_high_test, X_low_test

    trustworthiness, continuity = compute_trustworthiness_and_continuity(
        X_h_eval, X_l_eval, k=k
    )
    knn_acc = compute_knn_accuracy(
        X_low_train, y_train, X_low_test, y_test, k=k
    )

    anc_diag = compute_anchor_diagnostics_v6(
        embeddings=X_low_train,
        global_anchors=global_anchors,
        client_embeddings=client_embeddings,
        margin=margin,
    )

    return {
        "trustworthiness": trustworthiness,
        "continuity": continuity,
        "knn_accuracy": knn_acc,
        **anc_diag,
    }
