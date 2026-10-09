"""
Hungarian Anchor Alignment and Weighted Aggregation for FEDANCHOR V6
====================================================================

The Problem
-----------
Anchors are permutation-symmetric: if client A learns anchors [a1, a2]
and client B learns the same geometry but with swapped indices [a2, a1],
naive parameter averaging would compute the midpoints of a1↔a2 and
a2↔a1 — collapsing both anchors to the same centroid.

The Solution: Hungarian Assignment
------------------------------------
Before aggregating, each client's updated anchor matrix is ALIGNED to the
current global anchor matrix using minimum-cost bipartite matching
(Hungarian algorithm on Euclidean distance).  This produces a permutation
π such that client_anchors[π] ≈ global_anchors in ordering.

After alignment, weighted averaging is valid because anchor j of client m
now corresponds to anchor j of the global reference.

Algorithm (executed on server after every round):
  1. For each client m:
       a. Compute cost matrix C[i,j] = ||a_m_i - a_global_j||^2
       b. Solve linear_sum_assignment(C) → permutation π_m
       c. aligned_m = a_m[π_m]        # re-ordered to match global indexing
  2. Weighted average:
       A_new = Σ_m (n_m / N) * aligned_m
  3. Server stores A_new as the next global anchor matrix.
  4. Broadcast A_new to all clients before the next round.

This module provides:
    - align_anchors_to_reference(): per-client alignment
    - aggregate_anchors_with_alignment(): full aggregation pipeline
    - compute_alignment_distance(): diagnostic metric
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple


def align_anchors_to_reference(
    client_anchors: np.ndarray,
    reference_anchors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Align a client's anchor matrix to a reference anchor matrix using
    the Hungarian algorithm on squared Euclidean distances.

    Args:
        client_anchors:    shape [K, 2] — client's updated anchors
        reference_anchors: shape [K, 2] — current global anchors (reference)

    Returns:
        aligned_anchors:   shape [K, 2] — client anchors reordered to match
                                          the reference anchor indexing
        permutation:       shape [K]    — int array, permutation applied to
                                          client_anchors rows
        alignment_cost:    float        — mean squared distance between
                                          aligned_anchors and reference_anchors
                                          (diagnostic metric)
    """
    K = reference_anchors.shape[0]
    assert client_anchors.shape[0] == K, (
        f"Anchor count mismatch: client has {client_anchors.shape[0]}, "
        f"reference has {K}."
    )

    # Cost matrix: C[i, j] = squared Euclidean distance between
    #              client anchor i and reference anchor j
    diff = client_anchors[:, np.newaxis, :] - reference_anchors[np.newaxis, :, :]
    cost_matrix = np.sum(diff ** 2, axis=2)  # [K, K]

    # Hungarian assignment: minimize total cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # row_ind[i] == i always for square matrices, but col_ind gives the
    # optimal column assignment.

    # Build permutation: permutation[j] = i means "client row i → global row j"
    permutation = np.zeros(K, dtype=np.int64)
    for r, c in zip(row_ind, col_ind):
        permutation[c] = r  # global slot c gets client row r

    aligned_anchors = client_anchors[permutation]

    # Alignment quality metric
    alignment_cost = float(
        np.mean(np.sum((aligned_anchors - reference_anchors) ** 2, axis=1))
    )

    return aligned_anchors, permutation, alignment_cost


def aggregate_anchors_with_alignment(
    global_anchors: np.ndarray,
    client_anchor_updates: Dict[int, np.ndarray],
    client_sample_counts: Dict[int, int],
) -> Tuple[np.ndarray, Dict[int, float], Dict[int, np.ndarray]]:
    """
    Aggregate client anchor updates into a new global anchor matrix.

    Steps:
      1. For each client: Hungarian-align updated anchors to global_anchors.
      2. Weighted average of aligned anchors (weight = n_m / N_total).
      3. Return the new global anchor matrix.

    Args:
        global_anchors:        shape [K, 2] — server's current global anchors
        client_anchor_updates: {client_id → array [K, 2]} — end-of-round anchors
        client_sample_counts:  {client_id → int} — number of training samples

    Returns:
        new_global_anchors:    shape [K, 2] — updated global anchor matrix
        alignment_costs:       {client_id → float} — per-client alignment distance
        permutations:          {client_id → array [K]} — applied permutations
    """
    if len(client_anchor_updates) == 0:
        return global_anchors.copy(), {}, {}

    K = global_anchors.shape[0]
    total_samples = max(sum(client_sample_counts.values()), 1)

    aligned_list: List[np.ndarray] = []
    weights: List[float] = []
    alignment_costs: Dict[int, float] = {}
    permutations: Dict[int, np.ndarray] = {}

    for cid, anchors in client_anchor_updates.items():
        aligned, perm, cost = align_anchors_to_reference(anchors, global_anchors)
        aligned_list.append(aligned)
        weights.append(float(client_sample_counts.get(cid, 1)) / float(total_samples))
        alignment_costs[cid] = cost
        permutations[cid] = perm

    # Weighted sum
    new_anchors = np.zeros_like(global_anchors)
    for aligned, w in zip(aligned_list, weights):
        new_anchors += w * aligned

    return new_anchors, alignment_costs, permutations


def compute_alignment_distance(
    client_anchors: np.ndarray,
    reference_anchors: np.ndarray,
) -> float:
    """
    Compute mean alignment distance between two anchor matrices WITHOUT
    modifying either.  Used as a diagnostic before training starts to
    verify that clients received identical global anchors.

    If the two matrices are identical (max element-wise difference ≈ 0),
    the client correctly received the global anchors.

    Returns: mean squared distance after optimal alignment.
    """
    _, _, cost = align_anchors_to_reference(client_anchors, reference_anchors)
    return cost
