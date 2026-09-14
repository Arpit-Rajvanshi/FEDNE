# FINAL RESULTS TABLE — FEDANCHOR Review

> **Source**: Directly from `fedne_vs_fedanchor/fedne_results.csv` and `fedanchor_results.csv`.
> All values taken from actual CSV files. No fabricated numbers.

---

## Critical Note on Measurement Protocols

> [!WARNING]
> The kNN accuracy values for FEDNE and FEDANCHOR use **different evaluation protocols** and
> CANNOT be directly compared. FEDNE uses a 80/20 split of 2000 test embeddings. FEDANCHOR
> uses all 60,000 training embeddings to fit the classifier and all 10,000 test embeddings to score.
> Trustworthiness and Continuity ARE comparable (both use 2000-sample subsets).

---

## Round 20 Final Metrics

| Metric | FEDNE (Round 19 final) | FEDANCHOR (Round 20 final) | Difference | Interpretation |
|---|---|---|---|---|
| kNN Accuracy (%) | 90.5 | 20.52 | −69.98 | **NOT COMPARABLE** — different protocol |
| Trustworthiness | 0.9332 | 0.6268 | −0.3064 | FEDNE preserves local structure better |
| Continuity | 0.9574 | 0.7760 | −0.1814 | FEDNE preserves high-dim neighbors better |
| Total Loss | 0.4395 | 0.0543 | −0.3852 | Different formulations; not comparable |
| Attraction Loss | 0.2360 | 0.0000162 | −0.2360 | FEDANCHOR attraction near-zero (full-batch) |

---

## Controlled Experiment Setup Verification

| Setup Variable | FEDNE | FEDANCHOR | Match? |
|---|---|---|---|
| Dataset | MNIST | MNIST | YES |
| Partition | IID | IID | YES |
| Num Clients | 2 | 2 | YES |
| Rounds | 20 (rounds 0–19) | 20 (rounds 1–20) | YES |
| Random Seed | 42 | 42 | YES |
| Encoder Architecture | [784,512,256,128,2] | [784,512,256,128,2] | YES |
| Input Normalization | /255.0 flat | /255.0 flat | YES |
| Optimizer | Adam, lr=0.001 | Adam, lr=0.001 | YES |
| Eval Subset | 2000 test samples | 2000 test samples (T/C) | YES for T/C |
| kNN Eval Protocol | 80/20 split of 2000 test | All train → all test | **DIFFERENT** |
| Local Epochs | 1 | 1 | YES |
| Batch Size | 32 (mini-batch) | Full batch (30,000) | **DIFFERENT** |

**Note**: FEDNE uses mini-batch training (batch_size=32 based on config), while FEDANCHOR processes
the entire local dataset in one forward pass. This is a fundamental algorithmic difference that
affects convergence and loss scale.

---

## FEDANCHOR kNN Accuracy Over Rounds (Actual)

| Round | kNN Accuracy |
|---|---|
| 1 | 7.02% |
| 2 | 22.93% |
| 3 | 27.85% |
| 4 | 23.66% |
| 5 | 24.37% |
| 10 | 22.62% |
| 15 | 21.49% |
| 20 | 20.52% |

**Observation**: Accuracy peaks early (~Round 3) and then plateaus/decreases. No sustained improvement.

---

## FEDNE Metrics Over Rounds (Actual)

| Round | kNN Accuracy | Trustworthiness | Continuity | Loss |
|---|---|---|---|---|
| 0 (Round 1) | 58.75% | 0.8158 | 0.9482 | 1.0634 |
| 4 | 85.00% | 0.9203 | 0.9604 | 0.5475 |
| 10 | 91.00% | 0.9301 | 0.9595 | 0.4800 |
| 19 | 90.50% | 0.9332 | 0.9574 | 0.4395 |

**Observation**: FEDNE shows clear monotonic improvement and strong convergence.

---

## README Claim vs Actual Results

| Claimed | Actual (from CSV) | Status |
|---|---|---|
| FEDANCHOR kNN = 66.31% at Round 20 | 20.52% at Round 20 | **DISCREPANCY — not reproducible from saved CSV** |
| FEDANCHOR T = 0.8647 at Round 20 | 0.6268 at Round 20 | **DISCREPANCY** |
| FEDANCHOR C = 0.9126 at Round 20 | 0.7760 at Round 20 | **DISCREPANCY** |
| FEDANCHOR Loss = 0.1444 at Round 20 | 0.0543 at Round 20 | Different (possibly different run) |

> [!CAUTION]
> The numbers in the README (66.31%, 0.8647, 0.9126) do NOT match the actual fedanchor_results.csv.
> These must be reconciled before the review. Use only numbers that can be verified from saved files.
