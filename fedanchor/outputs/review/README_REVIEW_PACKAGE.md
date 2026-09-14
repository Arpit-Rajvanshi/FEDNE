# REVIEW PACKAGE — README

## FEDANCHOR Academic Review Package

Generated: 2026-09-14
Review: Intermediate Project Review

---

## What Is In This Package

This directory contains all materials needed for the academic review of:

**"Federated Anchor for Dimensionality Reduction"**

---

## Files

| File | Purpose |
|---|---|
| README_REVIEW_PACKAGE.md | This file |
| FINAL_RESULTS_TABLE.md | Actual results from CSV files with interpretation |
| FINAL_RESULTS_TABLE.csv | Machine-readable results table |
| PRESENTATION_OUTLINE.md | 12-slide deck with exact content and speaker notes |
| VIVA_QUESTIONS.md | 35 likely panel questions with accurate answers |
| MATHEMATICAL_FLOW.md | Step-by-step math walkthrough of FEDANCHOR |
| IMPLEMENTATION_EVIDENCE.md | Evidence inventory: what exists, where, and what is tested |
| TECHNICAL_ACCURACY_AUDIT.md | Equation-by-equation verification against code |
| SCIENTIFIC_LIMITATIONS.md | All red flags identified, classified by severity |
| FEDNE_ACCURACY_AUDIT.md | FEDNE baseline assessment and FEDANCHOR distinction |
| REVIEW_SCORE_ASSESSMENT.md | Strict rubric score with evidence and action items |

---

## Verified Source Files

| Source | Location |
|---|---|
| FEDANCHOR main 20-round CSV | fedanchor/outputs/fedne_vs_fedanchor/fedanchor_results.csv |
| FEDNE main 20-round CSV | fedanchor/outputs/fedne_vs_fedanchor/fedne_results.csv |
| Test results (15 passing) | fedanchor/outputs/fedne_vs_fedanchor/test_results.txt |
| Experiment metadata | fedanchor/outputs/fedne_vs_fedanchor/experiment_metadata.json |
| Anchor initializations | fedanchor/outputs/fedne_vs_fedanchor/initial_anchors.json |
| Comparison plots (6 graphs) | fedanchor/outputs/fedne_vs_fedanchor/*.png |
| Embedding visualizations | fedanchor/outputs/fedne_vs_fedanchor/fedanchor_run/*.png |

---

## Current Score Assessment (Strict)

| Criterion | Score |
|---|---|
| Implementation | 4/5 |
| Technical Accuracy | 3.5/5 |
| Results Obtained So Far | 3/5 |
| Presentation & Clarity | 3.5/5 |
| **TOTAL** | **14/20** |

---

## CRITICAL Issues to Resolve Before Review

1. **README claims 66.31% kNN accuracy; actual CSV shows 20.52%**
   - Must reconcile before presenting
   - Use only numbers verifiable from saved files

2. **kNN accuracy measurement protocol differs between FEDNE and FEDANCHOR**
   - FEDNE: 80/20 split of 2000 test embeddings
   - FEDANCHOR: 60k training → 10k test evaluation
   - Must disclose when presenting comparison

3. **main_anchor.py KeyError fixed** (encoder_bytes → encoder_bytes_per_client)
   - Already patched in this session

---

## Demo Commands

Run tests (6 seconds):
```powershell
.venv\Scripts\python.exe -m unittest discover -s fedanchor/tests -v
```

Full FEDANCHOR run (~5-10 min):
```powershell
.venv\Scripts\python.exe fedanchor/main_anchor.py --config fedanchor/configs/mnist_default.yaml
```

Show actual results:
```powershell
.venv\Scripts\python.exe -c "import csv; rows=list(csv.DictReader(open('fedanchor/outputs/fedne_vs_fedanchor/fedanchor_results.csv'))); print('Round 20:', rows[-1]['knn_accuracy'], rows[-1]['trustworthiness'])"
```

---

## 5 Most Important Graphs to Show

1. `fedne_vs_fedanchor/2_trustworthiness_vs_round.png` — T over 20 rounds both models
2. `fedne_vs_fedanchor/3_continuity_vs_round.png` — C over 20 rounds both models
3. `fedne_vs_fedanchor/6_final_metrics_comparison.png` — Final bar chart comparison
4. `fedanchor/outputs/fedne_vs_fedanchor/fedanchor_run/embeddings_with_anchors.png` — Anchors in 2D space
5. `fedanchor/outputs/fedne_vs_fedanchor/fedanchor_run/embeddings_by_class.png` — Class structure in 2D

---

## 5 Most Important Numbers to Remember (ACTUAL FROM CSV)

1. **FEDANCHOR Trustworthiness (Round 20)**: 0.627
2. **FEDNE Trustworthiness (Round 20)**: 0.933
3. **Anchor communication overhead**: 40 bytes (vs 2.27 MB encoder)
4. **15 tests**: All PASS in 5.9 seconds
5. **Encoder parameters**: 566,402 (exact, verifiable)

---

## 10 Most Likely Professor Questions

1. What is an anchor?
2. How are anchors initialized?
3. Why not average anchors in FedAvg?
4. What data does the server see?
5. Why is FEDANCHOR's kNN accuracy lower than FEDNE?
6. How did you verify gradients are isolated?
7. What is the anchor coverage loss and why does it help?
8. Is your FEDNE baseline the exact paper implementation?
9. What is the ablation study proving?
10. What would you do differently next time?

All answers are prepared in VIVA_QUESTIONS.md.
