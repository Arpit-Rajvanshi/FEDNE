# FEDANCHOR v2: Matching FEDNE Accuracy

The original prototype plateaued at **~20% kNN accuracy** (T ≈ 0.63, C ≈ 0.78) vs FEDNE's **~92%** in the controlled benchmark. The logs showed a collapsed embedding: attraction loss ≈ 1e-5 and 1–2 anchors owning ~100% of the points. Three root causes:

1. **No local repulsion.** The objective was attraction + anchor coverage + repulsion from the *other* client's anchors only. Attraction alone is minimised by putting every point in the same place, and the coverage loss pulls points onto their nearest anchor too, so both terms pushed towards collapse.
2. **One gradient step per round.** `local_epochs: 1` ran a single full-batch step, so 20 rounds meant 20 optimizer steps in total. FEDNE runs ~290 mini-batch steps per client per round.
3. **Anchors learned by gradient descent**, with no mass or spread information, so they drifted onto the dominant cluster and summarised the other client badly.

## What changed (`training.mode: "neighbor_embedding"`, now the default in `configs/mnist_default.yaml`)

| Component | v2 behaviour |
|---|---|
| Local training | Mini-batch over kNN edges (batch 512, Adam, grad-clip 1.0), same as FEDNE |
| Local repulsion | FEDNE negative-sample term `Σ_b −log(1−φ)`, b = 5, scaled by `|D_m|/|D|` (`losses/local_repulsion.py`) |
| Anchors | Re-fitted every round by K-Means on the client's 2D embeddings (K = 32). Each anchor carries its **mass** (cluster size) and **spread** (mean sq. radius) (`models/anchor.py::summarize_embeddings_as_anchors`) |
| Cross-client repulsion | Mass-weighted, spread-smoothed anchor mixture that approximates the other clients' negatives: `b · Σ_k (n_k/|D|) · log(1 + 1/(‖z−a_k‖² + s_k²))` (`losses/anchor_repulsion.py`). Optional `estimator: "sampled"` draws virtual negatives from the anchor mixture instead |
| Communication | 4·K floats per client per round (512 B for K = 32) on top of the encoder, vs a full surrogate MLP in FEDNE |

Files touched: `client/client.py`, `server/server.py`, `training/trainer.py`, `losses/anchor_repulsion.py`, `losses/local_repulsion.py` (new), `losses/__init__.py`, `models/anchor.py`, `configs/mnist_default.yaml`, `run_controlled_benchmark.py`, `run_v2_benchmark.py` (new), `tests/test_anchor_v2.py` (new).

The legacy prototype is still available with `training.mode: "fullbatch"`, and all 15 original tests still pass. The 4 new tests pass too (19/19).

## Results (MNIST, 2 clients, 20 rounds, identical evaluator & partitions, seed 42)

| Partition | Method | kNN acc | Trustworthiness | Continuity |
|---|---|---|---|---|
| IID | FEDNE | 0.9155 | 0.9371 | 0.9600 |
| IID | FEDANCHOR (original prototype) | 0.2052 | 0.6268 | 0.7760 |
| IID | **FEDANCHOR v2** | **0.9130** | **0.9359** | **0.9572** |
| IID | FEDANCHOR v2, no anchor repulsion | 0.9205 | 0.9364 | 0.9576 |
| Dirichlet α=0.1 | FEDNE | 0.7285 | 0.9028 | 0.9555 |
| Dirichlet α=0.1 | **FEDANCHOR v2** | **0.7845** | 0.8893 | 0.9514 |
| Dirichlet α=0.1 | FEDANCHOR v2 (sampled estimator) | 0.7610 | 0.9087 | 0.9528 |
| Dirichlet α=0.1 | FEDANCHOR v2, no anchor repulsion | 0.7560 | 0.9058 | 0.9536 |

Curves: `outputs/v2_benchmark/fedne_vs_fedanchor_v2.png`. Raw per-round numbers: `outputs/v2_benchmark/results_all_runs.csv`.

**How to read these results:**
- Most of the gain comes from fixes 1 and 2 (proper neighbor-embedding training). In the IID setting the anchor term makes no measurable difference, because each client's local negatives already cover the whole distribution.
- In the non-IID setting, anchor repulsion raises kNN accuracy (+2.9 pts closed-form) but lowers trustworthiness slightly (−1.6 pts). The sampled estimator keeps T/C at the no-anchor level and still beats FEDNE on kNN and T.
- All numbers come from a single seed, and round-to-round noise is about ±1–2 kNN points. Run several seeds before drawing firm conclusions.

## Reproduce (from the repo root)

```bash
python fedanchor/run_v2_benchmark.py fedne  20 iid
python fedanchor/run_v2_benchmark.py anchor 20 iid
python fedanchor/run_v2_benchmark.py anchor 20 dirichlet0.1
python fedanchor/run_v2_benchmark.py anchor 20 dirichlet0.1 '{"loss":{"anchor_repulsion":{"estimator":"sampled"}}}'
python fedanchor/run_v2_benchmark.py norep  20 dirichlet0.1
python -m unittest discover -s fedanchor/tests -t .
```

On 2 CPU cores, one FEDANCHOR v2 round takes ~30 s and one FEDNE round takes ~70 s.
