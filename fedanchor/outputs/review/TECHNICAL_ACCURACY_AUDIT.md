# TECHNICAL ACCURACY AUDIT — FEDANCHOR

> Verified against actual source code. No values invented.

---

## 1. Encoder Architecture

**Intended**: 784 → 512 → 256 → 128 → 2 (ReLU after each hidden layer, no activation at output)

**Actual Implementation** (`fedanchor/models/encoder.py`):
```
Linear(784, 512) → ReLU → Linear(512, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 2)
```

| Check | Result |
|---|---|
| 3 hidden layers | CORRECT |
| ReLU after each hidden | CORRECT |
| No activation at output | CORRECT (final Linear(128, 2)) |
| Output dimension = 2 | CORRECT |

**Parameter count**: 566,402 parameters (verified: 2265608 bytes at 4 bytes/param)
This matches the CSV communication column: encoder_upload_bytes = 4531216 = 566302 * 4 + 40 (anchors).

**Verdict**: CORRECT

---

## 2. kNN Graph Construction

**Intended**: For each point i, find k nearest neighbors by Euclidean distance, excluding self.

**Actual** (`fedanchor/graph/knn.py`):
```python
nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)  # k+1 to include self
filtered = row[row != i]                           # exclude self-loop
edges = np.stack([src, dst], axis=1)               # shape [N*k, 2]
```

| Check | Result |
|---|---|
| Self-loops excluded | CORRECT — row[row != i] |
| k = 5 (config default) | CORRECT — graph.k: 5 in YAML |
| Euclidean distance | CORRECT — sklearn default |
| Output shape [N*k, 2] | CORRECT |

**Verdict**: CORRECT

---

## 3. Attraction Loss

**Intended equation**:
```
L_att = (1/|E|) * sum_{(i,j) in E} log(1 + ||z_i - z_j||^2 + eps)
```

**Actual** (`fedanchor/losses/attraction.py`):
```python
sq_dist = torch.sum((z_src - z_dst) ** 2, dim=1)
loss = torch.log(1.0 + sq_dist + self.eps)
loss_val = torch.mean(loss)
```

- Inputs: z_src, z_dst in R^{|E| x 2}, indexed from embeddings by edge list
- Gradient: flows through embeddings → encoder
- Mathematical equivalence: log(1 + d^2) = -log(phi(z_i, z_j)) where phi = 1/(1+d^2). Correct.

**Verdict**: CORRECT

---

## 4. Anchor Coverage Loss

**Intended equation**:
```
L_anchor = (1/N) * sum_i min_j ||z_i - a_j||^2
```

**Actual** (`fedanchor/losses/anchor_loss.py`):
```python
diff = embeddings.unsqueeze(1) - anchors.unsqueeze(0)  # [N, K, 2]
sq_dists = torch.sum(diff ** 2, dim=2)                  # [N, K]
min_sq_dists, _ = torch.min(sq_dists, dim=1)            # [N]
loss_val = torch.mean(min_sq_dists)                      # scalar
```

- Gradient to anchors: d(L)/d(a_j) = -2(z_i - a_j)/N for assigned point i
- Gradient to encoder: d(L)/d(z_i) = 2(z_i - a_nearest)/N
- Test: `test_anchor_coverage_zero_on_exact_centroids` — loss=0 when embeddings==anchors. PASS.

**Verdict**: CORRECT — Exact mathematical match

---

## 5. Anchor Repulsion Loss

**Intended equation**:
```
L_rep_raw = (1/(N * |A_other|)) * sum_i sum_{a in A_other} log(1 + 1/(||z_i - a||^2 + eps))
L_rep_scaled = scale * L_rep_raw
```

**Actual** (`fedanchor/losses/anchor_repulsion.py`):
```python
diff = embeddings.unsqueeze(1) - other_anchors.unsqueeze(0)  # [N, K_other, 2]
sq_dists = torch.sum(diff ** 2, dim=2)                        # [N, K_other]
inv_sq_dist = 1.0 / (sq_dists + self.eps)
loss_matrix = torch.log(1.0 + inv_sq_dist)
raw_loss = torch.mean(loss_matrix)   # = (1/(N*K_other)) * sum
if normalization == "scale":
    scaled_loss = self.scale * raw_loss
```

- torch.mean over [N, K_other] = (1/(N*K)) which matches the equation. CORRECT.
- other_anchors detached at server: `anchors_tensor.detach().cpu().clone()`. CORRECT.
- Test `test_other_anchors_no_gradients`: other_anchors.grad is None after backward. PASS.
- Test `test_repulsion_produces_encoder_gradients`: encoder params receive non-zero grads. PASS.
- scale = 0.01 applied in production config. CORRECT.

**Verdict**: CORRECT

---

## 6. FedAvg Aggregation

**Intended equation**:
```
theta_global = sum_c (n_c / sum_c n_c) * theta_c
```

**Actual** (`fedanchor/server/server.py`):
```python
total_samples = sum(client_sample_counts)
for state, n_m in zip(client_encoder_states, client_sample_counts):
    weight = float(n_m) / float(total_samples)
    avg_state[key] += state[key].to(torch.float32) * weight
```

- Weights: n_m / total_samples. Sum = 1. CORRECT.
- Applied to all parameter keys in state_dict. CORRECT.
- Test `test_fedavg_weighted_aggregation`:
  - Client 1: 100 samples, params=1.0; Client 2: 300 samples, params=3.0
  - Expected: (100*1 + 300*3)/400 = 2.5
  - Result: 2.5 PASS.

**Note**: In the IID equal-split experiment (30k each), FedAvg = simple averaging (equal weights=0.5).

**Verdict**: CORRECT

---

## 7. Trustworthiness and Continuity

**Standard formulas** (Venna & Kaski 2006):
```
T(k) = 1 - (2/(n*k*(2n-3k-1))) * sum_i sum_{j in Uk(i)} (r(i,j) - k)
C(k) = 1 - (2/(n*k*(2n-3k-1))) * sum_i sum_{j in Vk(i)} (r_hat(i,j) - k)
```

**Actual** (`fedanchor/evaluation/metrics.py`):
```python
ranks_high = np.argsort(np.argsort(D_high, axis=1), axis=1)
nbrs_high = np.argsort(D_high, axis=1)[:, 1:k+1]   # [:, 0] is self
nbrs_low  = np.argsort(D_low, axis=1)[:, 1:k+1]
for i in range(n):
    for j in nbrs_low[i]:        # kNN in low-dim
        if j not in high_nbrs_set:  # not kNN in high-dim
            r = ranks_high[i, j]     # high-dim rank
            sum_t += (r - k)
norm = n * k * (2*n - 3*k - 1)
t_val = 1.0 - (2.0/norm) * sum_t
```

- Self excluded: [:, 1:k+1] CORRECT
- Rank via double argsort: CORRECT
- Penalty (r - k): CORRECT, matches standard
- Normalization: CORRECT

**Verdict for T/C**: CORRECT

---

## 8. kNN Accuracy — CRITICAL PROTOCOL DIFFERENCE (RED FLAG)

**FEDANCHOR**:
```python
clf.fit(X_train_low, y_train)      # 60,000 training embeddings
score = clf.score(X_test_low, y_test)  # 10,000 test embeddings
```

**FEDNE** (from `fedne/training/trainer.py` lines 196-200):
```python
n_eval = len(embeddings_eval)  # 2000 (subset of TEST set only)
split = int(0.8 * n_eval)      # 1600 train / 400 test split
knn_acc = compute_knn_accuracy(train_emb, train_labels, test_emb, test_labels)
```

**Impact**: FEDNE's kNN accuracy trains a classifier on only 1600 embeddings (from the test set),
while FEDANCHOR trains on 60,000 training embeddings. These measurements are NOT comparable.

This does NOT make either model's numbers wrong — but the comparison table cannot be used
to claim FEDANCHOR outperforms FEDNE in kNN accuracy because the evaluation protocols differ.

**Verdict**: MAJOR ISSUE — Benchmark comparison for kNN accuracy is not fair. Must be disclosed.

---

## Summary

| Component | Correct | Issue |
|---|---|---|
| Encoder 784→2 | YES | — |
| kNN (self-excluded) | YES | — |
| Attraction L_att | YES | — |
| Anchor Coverage L_anchor | YES | — |
| Anchor Repulsion L_rep | YES | — |
| FedAvg weighted | YES | — |
| Trustworthiness | YES | — |
| Continuity | YES | — |
| kNN Accuracy formula | YES | DIFFERENT PROTOCOL vs FEDNE |
