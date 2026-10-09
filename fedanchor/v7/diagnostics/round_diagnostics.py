"""
v7 Diagnostics Module
=====================

Reports per-round v7-specific diagnostic signals:

  1. Global anchor coordinates
  2. Pairwise anchor distances (min / mean / max)
  3. Anchor displacement ||A_t - A_{t-1}||
  4. Client-to-global anchor alignment distance
  5. Per-client soft utilisation
  6. Global soft utilisation
  7. Per-client entropy
  8. Global entropy
  9. Effective anchors (u_hard ≥ 0.05)
 10. Hard utilisation (argmin)
 11. Soft utilisation (q-based)

New v7 failure-mode detectors:
  - Anchor collapse (min pairwise distance < threshold)
  - Anchor drift (displacement > threshold)
  - Client domination (one client dominates global anchor geometry)
  - Uniform utilisation without geometric separation
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple


# -----------------------------------------------------------------------
# Failure-mode detectors
# -----------------------------------------------------------------------

COLLAPSE_DIST_THRESHOLD = 0.01   # anchors closer than this → collapse risk
DRIFT_THRESHOLD = 0.5            # per-round displacement larger → drift risk
DOMINATION_THRESHOLD = 0.9       # if one client's weight > this → domination


def detect_anchor_collapse(min_pairwise_dist: float) -> bool:
    """True if anchors are geometrically too close (collapse risk)."""
    return min_pairwise_dist < COLLAPSE_DIST_THRESHOLD


def detect_anchor_drift(displacement: float) -> bool:
    """True if anchor displacement per round is unusually large."""
    return displacement > DRIFT_THRESHOLD


def detect_client_domination(
    alignment_costs: Dict[int, float],
    client_sample_counts: Dict[int, int],
) -> Tuple[bool, Optional[int]]:
    """
    Check if one client's anchor update dominates the aggregation because
    of its sample-weight fraction.

    Returns (dominated: bool, dominant_client_id: int or None).
    """
    if not client_sample_counts:
        return False, None
    total = max(sum(client_sample_counts.values()), 1)
    for cid, n in client_sample_counts.items():
        if float(n) / float(total) >= DOMINATION_THRESHOLD:
            return True, cid
    return False, None


def detect_uniform_utilisation_without_separation(
    global_u: np.ndarray,
    min_pairwise_dist: float,
    uniformity_threshold: float = 0.05,
) -> bool:
    """
    True if anchor utilisation is nearly uniform BUT geometric spread
    is poor.  This indicates the anchors are spread in embedding space
    but all are equally used — which can mean the encoder is not well
    structured.
    """
    uniformity = float(np.std(global_u))
    return uniformity < uniformity_threshold and min_pairwise_dist < 0.5


def detect_encoder_anchor_mismatch(
    per_client_mean_dists: List[float],
    threshold: float = 2.0,
) -> bool:
    """
    True if the mean nearest-anchor distance is excessively large,
    suggesting the encoder output and anchor positions have diverged.
    """
    if not per_client_mean_dists:
        return False
    return float(np.mean(per_client_mean_dists)) > threshold


# -----------------------------------------------------------------------
# Round-level diagnostic report
# -----------------------------------------------------------------------

def build_round_diagnostics(
    round_num: int,
    global_anchors: np.ndarray,
    prev_global_anchors: Optional[np.ndarray],
    eval_metrics: Dict[str, Any],
    alignment_costs: Dict[int, float],
    permutations: Dict[int, np.ndarray],
    client_sample_counts: Dict[int, int],
    init_consistency_max_per_client: Dict[int, float],
) -> Dict[str, Any]:
    """
    Build a comprehensive diagnostic dict for a single federated round.

    Args:
        round_num:              current round index
        global_anchors:         [K, 2] NEW global anchors (after aggregation)
        prev_global_anchors:    [K, 2] PREVIOUS global anchors (before round)
        eval_metrics:           output of compute_metrics_v7()
        alignment_costs:        {cid → float} Hungarian alignment cost
        permutations:           {cid → array [K]} anchor permutations applied
        client_sample_counts:   {cid → int}
        init_consistency_max_per_client: {cid → float}
            max |A_client_start - A_global| before local training,
            should be ≈ 0 if broadcast was correct.

    Returns:
        diag: dict with all v7 diagnostics for this round
    """
    K = global_anchors.shape[0]

    # Anchor displacement
    if prev_global_anchors is not None:
        displacement = float(
            np.linalg.norm(global_anchors - prev_global_anchors)
        )
    else:
        displacement = 0.0

    # Pairwise distances
    from sklearn.metrics import pairwise_distances as _pw

    if K > 1:
        pw = _pw(global_anchors, metric="euclidean")
        triu = pw[np.triu_indices(K, k=1)]
        min_anc_dist = float(np.min(triu))
        mean_anc_dist = float(np.mean(triu))
        max_anc_dist = float(np.max(triu))
    else:
        min_anc_dist = 0.0
        mean_anc_dist = 0.0
        max_anc_dist = 0.0

    # Alignment stats
    if alignment_costs:
        mean_align_cost = float(np.mean(list(alignment_costs.values())))
        max_align_cost = float(np.max(list(alignment_costs.values())))
    else:
        mean_align_cost = 0.0
        max_align_cost = 0.0

    # Broadcast integrity check
    max_init_diff = max(
        init_consistency_max_per_client.values()
    ) if init_consistency_max_per_client else 0.0

    # Failure-mode flags
    collapse_flag = detect_anchor_collapse(min_anc_dist)
    drift_flag = detect_anchor_drift(displacement)
    dominated, dominant_cid = detect_client_domination(
        alignment_costs, client_sample_counts
    )

    global_u = np.array(
        eval_metrics.get("global_soft_utilization", [1.0 / K] * K)
    )
    unif_nosep_flag = detect_uniform_utilisation_without_separation(
        global_u, min_anc_dist
    )

    per_client_mean_dists = eval_metrics.get(
        "per_client_mean_nearest_anchor_dist", []
    )
    mismatch_flag = detect_encoder_anchor_mismatch(per_client_mean_dists)

    diag: Dict[str, Any] = {
        "round": round_num,
        # Anchor geometry
        "global_anchor_coordinates": global_anchors.tolist(),
        "min_pairwise_anchor_distance": min_anc_dist,
        "mean_pairwise_anchor_distance": mean_anc_dist,
        "max_pairwise_anchor_distance": max_anc_dist,
        "anchor_displacement": displacement,
        # Alignment
        "alignment_costs": {
            str(k): v for k, v in alignment_costs.items()
        },
        "mean_alignment_cost": mean_align_cost,
        "max_alignment_cost": max_align_cost,
        "permutations": {
            str(k): v.tolist() for k, v in permutations.items()
        },
        # Broadcast integrity
        "max_init_consistency_diff": max_init_diff,
        "broadcast_correct": max_init_diff < 1e-5,
        # Utilisation
        "global_soft_utilization": eval_metrics.get(
            "global_soft_utilization", []
        ),
        "per_client_soft_utilization": eval_metrics.get(
            "per_client_soft_utilization", []
        ),
        "hard_utilization": eval_metrics.get("hard_utilization_frac", []),
        "effective_anchors": eval_metrics.get("effective_anchors", 0),
        # Entropy
        "global_entropy": eval_metrics.get("normalized_entropy", 0.0),
        "per_client_entropy": eval_metrics.get("per_client_entropy", []),
        # Cross-client consistency
        "pairwise_client_l1_divergence": eval_metrics.get(
            "pairwise_client_l1_divergence", []
        ),
        "pairwise_client_js_divergence": eval_metrics.get(
            "pairwise_client_js_divergence", []
        ),
        "pairwise_client_utilisation_correlation": eval_metrics.get(
            "pairwise_client_utilisation_correlation", []
        ),
        "per_client_mean_nearest_anchor_dist": per_client_mean_dists,
        # Failure mode flags
        "failure_modes": {
            "anchor_collapse": collapse_flag,
            "anchor_drift": drift_flag,
            "client_domination": dominated,
            "dominant_client": dominant_cid,
            "uniform_util_no_separation": unif_nosep_flag,
            "encoder_anchor_mismatch": mismatch_flag,
        },
        # Standard metrics
        "knn_accuracy": eval_metrics.get("knn_accuracy", 0.0),
        "trustworthiness": eval_metrics.get("trustworthiness", 0.0),
        "continuity": eval_metrics.get("continuity", 0.0),
        "collapse_state": eval_metrics.get("collapse_state", "UNKNOWN"),
        # Embedding diagnostics
        "embedding_variance_dim0": eval_metrics.get("embedding_variance_dim0", 0.0),
        "embedding_variance_dim1": eval_metrics.get("embedding_variance_dim1", 0.0),
        "embedding_variance_total": eval_metrics.get("embedding_variance_total", 0.0),
        "embedding_norm_mean": eval_metrics.get("embedding_norm_mean", 0.0),
        "embedding_norm_max": eval_metrics.get("embedding_norm_max", 0.0),
        "embedding_lambda_min": eval_metrics.get("embedding_lambda_min", 0.0),
        "embedding_lambda_max": eval_metrics.get("embedding_lambda_max", 0.0),
        "embedding_pwd_mean": eval_metrics.get("embedding_pwd_mean", 0.0),
        "embedding_pwd_min": eval_metrics.get("embedding_pwd_min", 0.0),
        "embedding_pwd_max": eval_metrics.get("embedding_pwd_max", 0.0),
    }

    return diag


def print_round_summary(diag: Dict[str, Any]) -> None:
    """Print a concise per-round summary to stdout."""
    r = diag["round"]
    print(f"\n{'='*70}")
    print(f"[Round {r:03d}]")
    print(
        f"  kNN={diag['knn_accuracy']:.4f}  "
        f"Trust={diag['trustworthiness']:.4f}  "
        f"Cont={diag['continuity']:.4f}"
    )
    print(
        f"  MinAncDist={diag['min_pairwise_anchor_distance']:.4f}  "
        f"Displacement={diag['anchor_displacement']:.6f}  "
        f"EffAnchors={diag['effective_anchors']}"
    )
    print(
        f"  GlobalEntropy={diag['global_entropy']:.4f}  "
        f"Collapse={diag['collapse_state']}"
    )
    
    if 'embedding_variance_total' in diag:
        print(f"  Embed Variance: Total={diag.get('embedding_variance_total',0):.4f}  "
              f"Dim0={diag.get('embedding_variance_dim0',0):.4f}  Dim1={diag.get('embedding_variance_dim1',0):.4f}")
        print(f"  Embed PWD: Mean={diag.get('embedding_pwd_mean',0):.4f}  "
              f"Min={diag.get('embedding_pwd_min',0):.4f}  Max={diag.get('embedding_pwd_max',0):.4f}")
        print(f"  Embed Cov Eigenvalues: lambda_max={diag.get('embedding_lambda_max',0):.4f}  "
              f"lambda_min={diag.get('embedding_lambda_min',0):.4f}")

    if diag.get("per_client_soft_utilization"):
        for m, u in enumerate(diag["per_client_soft_utilization"]):
            util_str = ", ".join([f"{x*100:.1f}%" for x in u])
            print(f"  Client {m} soft util: [{util_str}]")
    glob_u = diag.get("global_soft_utilization", [])
    if glob_u:
        g_str = ", ".join([f"{x*100:.1f}%" for x in glob_u])
        print(f"  Global soft util:   [{g_str}]")
    # Alignment stats
    print(
        f"  MeanAlignCost={diag['mean_alignment_cost']:.6f}  "
        f"MaxAlignCost={diag['max_alignment_cost']:.6f}"
    )
    # Broadcast check
    bc = diag.get("broadcast_correct", True)
    ic = diag.get("max_init_consistency_diff", 0.0)
    print(f"  BroadcastCorrect={bc}  MaxInitDiff={ic:.2e}")
    # Failure modes
    fm = diag.get("failure_modes", {})
    active = [k for k, v in fm.items() if v and k not in ("dominant_client",)]
    if active:
        print(f"  WARNING  Failure modes detected: {active}")
    print(f"{'='*70}")
