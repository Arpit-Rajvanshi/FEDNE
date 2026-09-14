# REVIEW SCORE ASSESSMENT — FEDANCHOR

> Strict assessment against the four marking criteria.
> Based on actual code inspection and actual experimental results.
> NO inflation of scores.

---

## CRITERION 1 — IMPLEMENTATION (5 marks)

**Rubric**: "Demonstrates working modules representing approximately half of the approved scope; 
the evidence is executable and attributable to the team."

### Evidence Currently Exists:

**Working modules confirmed by inspection**:
- `fedanchor/models/encoder.py` — Encoder (784→2), 566,402 params
- `fedanchor/models/anchor.py` — AnchorSet with 3 initialization strategies
- `fedanchor/graph/knn.py` — kNN graph construction (self-excluded)
- `fedanchor/losses/attraction.py` — Attraction loss
- `fedanchor/losses/anchor_loss.py` — Anchor coverage loss (min-VQ)
- `fedanchor/losses/anchor_repulsion.py` — Cross-client anchor repulsion
- `fedanchor/losses/total_loss.py` — Combined objective with ablation flags
- `fedanchor/client/client.py` — AnchorClient with full local training loop
- `fedanchor/server/server.py` — AnchorServer with FedAvg + anchor storage
- `fedanchor/training/trainer.py` — Full federated training orchestrator
- `fedanchor/evaluation/metrics.py` — T, C, kNN accuracy, anchor diagnostics
- `fedanchor/evaluation/visualization.py` — 4 visualization functions

**Test suite**: 15 tests, ALL PASS (test_results.txt confirmed)

**Executable evidence**:
- 20-round FEDANCHOR run: `fedanchor/outputs/fedne_vs_fedanchor/fedanchor_results.csv`
- 20-round FEDNE run: `fedanchor/outputs/fedne_vs_fedanchor/fedne_results.csv`
- Ablation directories with plots for all 4 configurations
- 3 initialization strategy outputs in `exp_init_*/` directories
- Checkpoint: `fedanchor/checkpoints/fedanchor_checkpoint.pt` (2.27 MB)

### What Is Missing:

- No mini-batch training (full-batch only, limits learning quality)
- Ablation runs only 2 rounds (weak evidence)
- FEDNE comparison has a kNN accuracy measurement protocol discrepancy

### What a Professor Expects:
- Working prototype with at least half the scope: ✅ (well over half)
- Executable and attributable: ✅ (15 passing tests, CSV evidence)
- FedAvg implemented: ✅
- Loss functions implemented: ✅

### What to Demonstrate:
- Run the test suite (6 seconds, 15 tests pass)
- Show the fedanchor_results.csv with 20 rounds
- Show the embeddings_with_anchors.png visualization

### Estimated Score: **4/5**

**Rationale**: All major modules are implemented and executable. The test suite passes.
There is strong experimental evidence from 20-round runs. Score is not 5/5 because:
- Ablation study is too short (2 rounds)
- Full-batch training is a known limitation that reduces embedding quality
- The single-anchor collapse issue shows the prototype has not been fully resolved

---

## CRITERION 2 — TECHNICAL ACCURACY (5 marks)

**Rubric**: "Uses technically correct methods, algorithms, parameters, and implementation practices 
consistent with the approved design."

### Evidence Currently Exists:

**All core equations verified correct**:
- Encoder: correct architecture (ReLU hidden, linear output) ✅
- kNN: self-loops correctly excluded via `row[row != i]` ✅
- Attraction: `log(1 + d^2 + eps)` = `-log phi(z_i,z_j)` ✅
- Anchor Coverage: `(1/N) sum_i min_j ||z_i - a_j||^2` exactly ✅
- Anchor Repulsion: `log(1 + 1/(d^2+eps))` with mean normalization ✅
- FedAvg: `sum_m (n_m/n) * theta_m` — verified by unit test (2.5 exact) ✅
- Trustworthiness/Continuity: Venna & Kaski formula — verified ✅
- Gradient isolation: other-client anchors detached — verified by test ✅

**Hyperparameters match design**:
- k=5 (kNN), K=5 (anchors), lambda_att=lambda_anchor=lambda_rep=1.0, scale=0.01 ✅
- Adam lr=0.001, grad_clip=1.0 ✅

### What Is Missing / Issues:

**MAJOR**: kNN accuracy measurement protocol differs between FEDNE and FEDANCHOR.
This is a technical inaccuracy in the benchmark design, even if individual implementations are correct.

**MINOR**: The attraction loss in FEDANCHOR is numerically near-zero (~1e-5) compared to FEDNE (~0.24).
This indicates full-batch averaging over 30,000 samples × 5 edges = 150,000 terms dilutes the signal.
The implementation is technically correct but the training regime creates a near-zero attraction.

### What a Professor Expects:
- Correct mathematical formulation: ✅
- Parameter choices justified: ✅ (scale=0.01 to prevent repulsion dominance)
- Gradient flow verified: ✅ (by autograd tests)
- Technical accuracy in benchmark design: ⚠️ (protocol mismatch)

### What to Demonstrate:
- Walk through the math of L_anchor = (1/N) sum_i min_j ||z_i - a_j||^2
- Show test_other_anchors_no_gradients: proves gradient isolation
- Show test_fedavg_weighted_aggregation: proves FedAvg is exact

### Estimated Score: **3.5/5**

**Rationale**: All mathematical components are correctly implemented and verifiable.
The deduction comes from: (1) kNN protocol mismatch in the benchmark — a real technical flaw,
(2) the near-zero attraction in practice suggests a training regime issue not fully resolved.
These are not severe enough to drop below 3, but prevent a 5.

---

## CRITERION 3 — RESULTS OBTAINED SO FAR (5 marks)

**Rubric**: "Presents interim metrics, tables, graphs, or outputs and provides technically sound 
interpretation and comparison."

### Evidence Currently Exists:

**Graphs** (from `fedne_vs_fedanchor/`):
1. `1_accuracy_vs_round.png` — kNN accuracy over 20 rounds, both models
2. `2_trustworthiness_vs_round.png` — T over rounds
3. `3_continuity_vs_round.png` — C over rounds
4. `4_loss_vs_round.png` — Loss over rounds
5. `5_communication_vs_round.png` — Bandwidth per round
6. `6_final_metrics_comparison.png` — Bar chart final metrics

**Embedding plots** (in multiple experiment dirs):
- `embeddings_by_class.png` — 2D space colored by digit class
- `embeddings_by_client.png` — 2D space colored by client
- `embeddings_with_anchors.png` — embeddings + anchor stars
- `anchor_positions.png` — anchor positions with labels

**CSV data** (real, unmodified):
- `fedne_results.csv` — 20 rounds FEDNE metrics
- `fedanchor_results.csv` — 20 rounds FEDANCHOR metrics

**Problems with results**:
1. CRITICAL: README claims 66.31% kNN accuracy but CSV shows 20.52%. These conflict.
2. MAJOR: FEDANCHOR kNN accuracy does not improve over rounds (7% → 20% → decreasing)
3. MAJOR: Anchor collapse from Round 5 (>90% single anchor)
4. MAJOR: kNN accuracy comparison is not valid due to protocol mismatch

### What Is Missing:
- Ablation results CSV (only images, no CSV with numbers per ablation)
- No quantitative comparison of initialization strategies
- No reproducibility run to verify numbers are consistent

### What a Professor Expects:
- Real data from actual runs: ✅
- Graphs with proper labels: ✅
- Technically sound interpretation: ⚠️ (must acknowledge limitations honestly)
- Comparison: PARTIAL (protocol mismatch must be disclosed)

### What to Demonstrate:
- Show the 6 comparison graphs (real data)
- Honestly interpret: "FEDANCHOR achieves 0.627 trustworthiness vs FEDNE's 0.933, with K=5 anchors"
- Show embeddings_with_anchors.png — visible 2D structure + anchor positions
- Explicitly acknowledge anchor collapse and explain it

### Estimated Score: **3/5**

**Rationale**: Real data exists and graphs are generated from actual CSVs. However:
- The most prominent claimed result (66.31%) appears to be from a different run not reproducible from the saved CSV
- FEDANCHOR results show no sustained improvement in kNN accuracy
- The comparison metric has a protocol flaw that must be disclosed

This prevents a score above 3. With honest disclosure and technically sound interpretation,
a 3 is achievable. Fabricating or hiding the 66.31% vs 20.52% discrepancy would be academically problematic.

---

## CRITERION 4 — PRESENTATION AND CLARITY (5 marks)

**Rubric**: "Explains the work logically, justifies decisions, and responds accurately to panel questions."

### Evidence Currently Exists:
- MATHEMATICAL_FLOW.md: complete step-by-step walkthrough
- VIVA_QUESTIONS.md: 35 questions with technically accurate answers
- PRESENTATION_OUTLINE.md: 12 slides with exact content and speaker notes
- TECHNICAL_ACCURACY_AUDIT.md: verified each equation vs implementation
- SCIENTIFIC_LIMITATIONS.md: honest disclosure of all red flags
- REVIEW_DEMO.md: exact commands for live demonstration
- fedanchor/README.md: well-structured documentation (though some numbers are inconsistent)
- test_results.txt: clean evidence for 15 passing tests

### What Is Missing:
- Cleared inconsistency between README claims and actual CSV results
- Architecture diagram (SVG/PNG — not yet created, text version in REVIEW_DEMO.md)

### What a Professor Expects:
- Clear problem statement: ✅
- Mathematical justification: ✅
- Understanding of federated setting: ✅
- Honest acknowledgment of limitations: ✅ (prepared in SCIENTIFIC_LIMITATIONS.md)
- Able to answer panel questions: ✅ (35 Q&A prepared)
- Consistent numbers: ⚠️ (66.31% vs 20.52% must be resolved before presenting)

### What to Demonstrate:
- Present the FEDANCHOR vs FEDNE distinction clearly (Slide 5)
- Walk through the math on the board if asked
- Run the test suite live (Demo 1)

### Estimated Score: **3.5/5**

**Rationale**: The prepared materials are thorough and technically grounded. The presentation
outline is complete. The viva preparation covers likely questions accurately. Score deducted because:
- The inconsistency between README (66.31%) and actual CSV (20.52%) must be resolved
- If this discrepancy is not resolved before the review, a panel question about the numbers
  could create a credibility problem

---

## TOTAL ESTIMATED SCORE

| Criterion | Score | /5 |
|---|---|---|
| Implementation | 4/5 | 5 |
| Technical Accuracy | 3.5/5 | 5 |
| Results Obtained So Far | 3/5 | 5 |
| Presentation & Clarity | 3.5/5 | 5 |
| **TOTAL** | **14/20** | **20** |

---

## What Would Push This to 18-20/20

1. **Resolve the 66.31% vs 20.52% discrepancy** — Either find and document the run that produced 66.31% with reproducible command, OR re-run with better hyperparameters and document the new results honestly. **(+1-2 marks)**

2. **Fix the kNN protocol mismatch** — Use the same measurement protocol for both FEDNE and FEDANCHOR in the comparison. This makes the comparison scientifically valid. **(+1 mark)**

3. **Run a proper ablation study** — 10-20 rounds per ablation configuration instead of 2. **(+0.5 marks)**

4. **Mini-batch training investigation** — Even a brief experiment showing that mini-batch attraction produces non-negligible loss signal would demonstrate deeper understanding. **(+0.5-1 mark)**

5. **Resolve the README inconsistency** — Ensure all numbers in README match actual CSV files. **(+0.5 marks)**

---

## Actions Required Before Review

| Priority | Action | Effort |
|---|---|---|
| CRITICAL | Resolve 66.31% vs 20.52% discrepancy | 1-2 hours |
| HIGH | Fix kNN measurement protocol to match FEDNE | 1 hour |
| HIGH | Update README with actual verified numbers | 30 min |
| MEDIUM | Re-run ablation for 10 rounds | 2-3 hours |
| LOW | Run mini-batch experiment | 4+ hours |
| DONE | Fix main_anchor.py KeyError (encoder_bytes) | Fixed |
