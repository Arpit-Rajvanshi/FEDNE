# SCIENTIFIC LIMITATIONS — FEDANCHOR

> This document honestly documents all identified problems with severity classifications.
> These were identified through direct code inspection, not assumption.

---

## RED FLAG 1 — kNN Accuracy Measurement Protocol Mismatch

**Classification**: MAJOR

**Problem**: The FEDNE and FEDANCHOR baselines measure kNN accuracy differently.

- FEDANCHOR: `clf.fit(X_train_low [60,000 pts], y_train)` then `score(X_test_low [10,000 pts], y_test)`
- FEDNE: Splits 2000-sample TEST subset 80/20, trains on 1600 test embeddings, evaluates on 400

**Impact**: kNN accuracy comparisons between the two systems are not apples-to-apples.
The previously reported numbers (FEDNE ~90%, FEDANCHOR ~20%) use different protocols and reflect
different things, not just model quality differences.

**Must Fix Before Review?**: YES — must disclose this in presentation. Do not claim FEDANCHOR beats FEDNE
on kNN accuracy without acknowledging the protocol difference.

---

## RED FLAG 2 — Previously Reported kNN Accuracy for FEDANCHOR Is Inconsistent with CSV

**Classification**: CRITICAL

**Problem**: The README and previous reports state:
```
FEDANCHOR kNN Accuracy = 66.31% (Round 20)
```

The **actual `fedanchor_results.csv`** shows:
```
Round 1:  knn_accuracy = 0.0702  (7.02%)
Round 20: knn_accuracy = 0.2052  (20.52%)
```

The 66.31% figure does NOT appear anywhere in the actual experimental CSV files. This discrepancy is large (66.31% vs 20.52%) and must be resolved before the review. Possible causes:
- The 66.31% was from a different run with different config that was not saved to this CSV
- The 66.31% was from the `exp_20_rounds` experiment (which only has image files, not a CSV)
- The CSV in `fedne_vs_fedanchor` is from a separate, possibly later run with different settings

**Also check**: The FEDNE CSV shows kNN_accuracy = 0.905 (Round 19), suggesting very strong performance.
If the earlier FEDANCHOR run genuinely produced 66.31%, the experimental conditions need to be documented.

**Must Fix Before Review?**: YES — if you cannot reproduce 66.31%, you must use the actual 20.52%.
Do NOT present 66.31% without the supporting CSV.

---

## RED FLAG 3 — Anchor Collapse: Single-Anchor Utilization

**Classification**: MAJOR

**Problem**: From the `fedanchor_results.csv`, anchor utilization from Round 5 onwards:
```
Round 5:  [0.0%, 0.2%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 99.7%]
Round 6:  [0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 100.0%]
Round 10: [0.0%, 48.8%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 51.2%]
Round 20: [0.0%, 88.8%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 11.2%]
```

Note: The experiment used `num_anchors=5` but 10 anchor utilization percentages are shown.
This indicates the utilization is being measured across ALL anchors from BOTH clients (5+5=10).
Still, nearly all mass concentrates on 1-2 anchors, meaning the K=5 anchor representation collapses.

**Root Cause**: The anchor coverage loss (min assignment) has a winner-take-all effect — once an anchor 
is nearest to most points, it remains nearest, starving other anchors of gradient signal.

**Impact on Results**: With anchor collapse, the cross-client repulsion has very little structural diversity 
to exploit. The repulsion is effectively coming from 1-2 point anchors rather than K=5 well-spread anchors.

**README acknowledgment**: Section 8 of the README does acknowledge this: "In high-epoch regimes without
repulsion gradient clipping on anchors, points can collapse to dominant anchors."

**Must Fix Before Review?**: SHOULD disclose clearly. Can present as a known limitation with future work.

---

## RED FLAG 4 — Increasing Nearest-Anchor Distance (Anchor Drift)

**Classification**: MAJOR

**Problem**: From CSV, `mean_nearest_anchor_distance` increases over rounds:
```
Round 1:  0.0412
Round 4:  0.1670
Round 10: 0.1226
Round 20: 0.1047
```

And max distance increases sharply:
```
Round 1:  0.2482
Round 4:  0.3455
Round 10: 0.2346
Round 20: 0.2026
```

This means anchors are drifting away from the embedding distribution they should represent.
Good anchors should have low mean_nearest_anchor_distance (embeddings close to anchors).

**Combined with collapse**: A single dominant anchor at large distance from most points means
the anchor coverage loss is doing its job poorly.

**Must Fix Before Review?**: No code fix needed. Must be disclosed as observed behavior.

---

## RED FLAG 5 — FEDANCHOR kNN Accuracy Does Not Improve Over Rounds

**Classification**: MAJOR

**Problem**: From the actual CSV:
```
Round 1:  knn_accuracy = 0.0702  (7.02%)
Round 5:  knn_accuracy = 0.2437  (24.37%)
Round 10: knn_accuracy = 0.2262  (22.62%)
Round 15: knn_accuracy = 0.2149  (21.49%)
Round 20: knn_accuracy = 0.2052  (20.52%)
```

kNN accuracy starts low, rises slightly by round 5, then oscillates and slightly DECREASES by round 20.
This is the opposite of the "66.31% at Round 20" claim in the README.

Trustworthiness also stays roughly flat:
```
Round 1:  0.641
Round 20: 0.627
```

This suggests FEDANCHOR with the current config is not significantly learning useful 2D structure 
compared to random initialization. The attraction loss is extremely small (0.00001-0.0002),
suggesting most of the loss signal is from the repulsion term, which dominates.

**Must Fix Before Review?**: YES — need to either re-run FEDANCHOR with corrected hyperparameters
that demonstrate actual learning, OR honestly present these numbers and explain why.

---

## RED FLAG 6 — FEDANCHOR Attraction Loss Extremely Small vs FEDNE

**Classification**: MAJOR

**Problem**: FEDANCHOR attraction values in CSV:
```
Round 1:  attraction = 6.87e-5
Round 20: attraction = 1.62e-5
```

FEDNE attraction values in CSV:
```
Round 1:  attraction = 0.422
Round 20: attraction = 0.236
```

FEDANCHOR's attraction is ~10,000x smaller than FEDNE's. This suggests the kNN graph
is effectively not learning from the neighborhood structure, likely because:
1. The total loss is dominated by repulsion (which is at ~5.2 scaled or ~0.05 with scale=0.01)
2. Or the neighbor embedding task is being overwhelmed

**Root cause suspect**: FEDANCHOR computes attraction over the ENTIRE dataset in one pass
(full batch, 30,000 samples), which means the attraction per pair is averaged over many edges
and becomes very small. FEDNE uses mini-batches with explicit edge sampling.

**Must Fix Before Review?**: YES — this is the core technical issue. Consider noting that
FEDANCHOR's training regime needs further tuning, or running a mini-batch version.

---

## RED FLAG 7 — Communication Byte Count Discrepancy in main_anchor.py

**Classification**: MINOR

**Problem**: `main_anchor.py` line 45 references:
```python
print(f"  - Encoder Parameter Count: {comm['encoder_parameter_count']} scalars ({comm['encoder_bytes']} bytes)")
```

But `server.py` compute_communication_bytes() returns key `encoder_bytes_per_client`, NOT `encoder_bytes`.
This would cause a `KeyError` if `main_anchor.py` is run directly.

**Must Fix Before Review?**: YES — this is a runtime error in the demo script. Fix before live demo.

---

## RED FLAG 8 — FEDNE Baseline Is Not Paper-Exact

**Classification**: ACCEPTABLE LIMITATION

**Assessment**: The FEDNE implementation uses:
- The correct attraction loss: `L_att = -E_ij log(phi(z_i, z_j))` (exact match)
- The correct local repulsion: `L_rep = -E_i sum_s log(1 - phi(z_i, z_neg_s))` (exact match)
- Surrogate-based cross-client repulsion: `L_rep_surr = scale * surrogate(z_i)` (approximate)

The surrogate model is a small MLP trained to approximate the other client's local repulsion density.
This is the paper's proposed mechanism. The implementation appears to follow the approach described.

However:
- The FEDNE paper (if this refers to Vitter et al. or another specific paper) may have specific hyperparameter choices not reproduced exactly
- Surrogate training uses MSE loss fitting to the local repulsion values
- No reference to specific hyperparameter values from the original paper

**For academic presentation**: State clearly that "our FEDNE baseline is our re-implementation of the 
surrogate-assisted federated neighbor embedding approach. It is our best-effort reproduction but may not 
reproduce the exact numbers from the original publication."

---

## RED FLAG 9 — Ablation Study Only Runs 2 Rounds

**Classification**: MINOR

**Problem**: Ablation modes (A, B, C, D) each run for only 2 rounds (see `run_fedne_vs_fedanchor.py` line 257).
2 rounds is insufficient to differentiate the effect of different loss components.

**Must Fix Before Review?**: For a stronger presentation, re-run ablation for at least 10 rounds.
For the review as-is, disclose that ablation results are from quick 2-round runs (sanity check only).

---

## RED FLAG 10 — 13 vs 15 Tests Discrepancy in Previous Report

**Classification**: MINOR — RESOLVED

**Assessment**: The test suite reports "Ran 15 tests in 5.912s" in `test_results.txt`. Previous reports
may have said 13. The actual count is 15 (8 in test_anchor_components + 4 in test_anchor_training + 3 in test_federated_anchor). All 15 PASS.

---

## Summary Risk Table

| Issue | Severity | Must Fix | Action |
|---|---|---|---|
| kNN protocol mismatch | MAJOR | YES | Disclose in presentation |
| 66.31% vs 20.52% discrepancy | CRITICAL | YES | Verify which is correct, use only verified number |
| Anchor collapse (1 anchor dominates) | MAJOR | NO | Disclose as limitation |
| Anchor drift (increasing distance) | MAJOR | NO | Disclose as observation |
| FEDANCHOR kNN not improving | MAJOR | YES | Investigate / disclose honestly |
| Tiny attraction loss | MAJOR | YES | Explain full-batch vs mini-batch |
| main_anchor.py KeyError in demo | MINOR | YES | Fix before live demo |
| FEDNE is re-implementation | ACCEPTABLE | YES | State clearly in presentation |
| Ablation only 2 rounds | MINOR | SHOULD | Re-run or disclose |
