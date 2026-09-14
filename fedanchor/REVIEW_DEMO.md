# REVIEW DEMO — FEDANCHOR

> This document provides exact commands for demonstrating the working FEDANCHOR prototype.
> All commands are run from the D:\FEDNE directory.

---

## Prerequisites

```powershell
# Activate the virtual environment
.venv\Scripts\python.exe --version
# Expect: Python 3.11.0rc1

# Verify required packages
.venv\Scripts\python.exe -c "import torch; import sklearn; import numpy; print(torch.__version__, sklearn.__version__, numpy.__version__)"
```

---

## Demo 1: Run the Full Test Suite (Fastest — 6 seconds)

```powershell
# Run from D:\FEDNE
.venv\Scripts\python.exe -m unittest discover -s fedanchor/tests -v 2>&1 | Tee-Object -FilePath fedanchor/outputs/review/latest_test_run.txt
```

**Expected output**:
```
test_anchor_coverage_loss_finite ... ok
test_anchor_coverage_zero_on_exact_centroids ... ok
test_anchor_initialization_strategies ... ok
test_anchor_repulsion_loss_finite ... ok
test_attraction_loss_finite ... ok
test_encoder_output_shape ... ok
test_knn_graph_shape ... ok
test_total_loss_computation ... ok
test_client_local_training_step ... ok
test_local_anchors_update ... ok
test_other_anchors_no_gradients ... ok
test_repulsion_produces_encoder_gradients ... ok
test_anchor_storage_independent ... ok
test_end_to_end_toy_experiment ... ok
test_fedavg_weighted_aggregation ... ok

Ran 15 tests in ~6s

OK
```

**What this demonstrates**:
- All 15 unit and autograd tests pass
- Encoder output shape is correct (32, 2)
- Anchor initialization works for all 3 strategies
- FedAvg weighted aggregation is exact (2.5 = (100x1 + 300x3)/400)
- Other-client anchors receive NO gradients (gradient isolation)
- Repulsion loss produces non-zero encoder gradients (training signal exists)
- End-to-end 2-round toy experiment completes without error

---

## Demo 2: Single-Command FEDANCHOR Mini-Run (Quick demo — ~5 minutes)

```powershell
# Run from D:\FEDNE
# This uses test_config.yaml which runs 2 rounds with 200 samples per client
.venv\Scripts\python.exe -c "
import sys, os
sys.path.insert(0, '.')
from fedanchor.training.trainer import FederatedAnchorTrainer
trainer = FederatedAnchorTrainer(config_path='fedanchor/configs/test_config.yaml')
res = trainer.train()
print('=== RESULTS ===')
print(f'Rounds completed: {len(res[\"history\"])}')
h = res['history'][-1]
em = h['eval_metrics']
print(f'Final Loss: {h[\"loss\"]:.4f}')
print(f'Trustworthiness: {em[\"trustworthiness\"]:.4f}')
print(f'Continuity: {em[\"continuity\"]:.4f}')
print(f'kNN Accuracy: {em[\"knn_accuracy\"]*100:.2f}%')
print(f'Anchor Utilization: {em[\"anchor_utilization\"]}')
comm = res['communication']
print(f'Encoder bytes per client: {comm[\"encoder_bytes_per_client\"]}')
print(f'Anchor bytes per client: {comm[\"anchor_bytes_per_client\"]}')
print(f'Checkpoint saved: {res[\"checkpoint_path\"]}')
print(f'Embeddings saved: {res[\"visualizations\"][\"embeddings_with_anchors\"]}')
"
```

**Expected execution flow**:
1. MNIST data loaded (60k train, 10k test), partitioned IID to 2 clients (200 each for test config)
2. kNN graphs constructed locally on each client
3. Initial embeddings computed, anchors initialized via KMeans
4. Round 1: local training, FedAvg, evaluation
5. Round 2: repeat
6. Visualizations saved to `fedanchor/checkpoints/test_run/` (or wherever test_config.yaml specifies)
7. Results printed

**Expected files generated**:
- `fedanchor/checkpoints/test_run/fedanchor_checkpoint.pt` — saved model
- Visualization PNGs: `embeddings_by_class.png`, `embeddings_by_client.png`, `embeddings_with_anchors.png`, `anchor_positions.png`

---

## Demo 3: Full MNIST FEDANCHOR Run (For actual demonstration — ~5-10 minutes)

```powershell
# Run from D:\FEDNE
.venv\Scripts\python.exe fedanchor/main_anchor.py --config fedanchor/configs/mnist_default.yaml
```

**What this runs**:
1. **Data Loading**: Downloads/loads MNIST, normalizes to [0,1], flattens to 784D
2. **Client Partitioning**: Splits 60,000 training samples IID into 2 clients × 30,000 samples
3. **Encoder Initialization**: Random init of Linear(784→512→256→128→2) — 566,402 params
4. **kNN Graph**: Constructed on each client's 30,000 samples (k=5 neighbors, excludes self)
5. **Anchor Initialization**: KMeans on initial 2D embeddings (5 anchors per client)
6. **Federated Training**: 20 rounds of:
   - Server broadcasts global encoder + other client's anchors
   - Each client: encode → compute L_total → backprop → update encoder + anchors
   - FedAvg encoder weights
   - Evaluate: T, C, kNN accuracy, anchor diagnostics

**Expected Output Per Round** (example format):
```
[Round 001/020]
Loss: 0.0457 | Attraction: 0.0001 | Anchor Loss: 0.0000
Repulsion Raw: 4.5588 | Repulsion Scaled: 0.0456
Trustworthiness: 0.6411 | Continuity: 0.8186 | kNN Acc: 0.0702
Anchor Coverage (Mean Min Dist): 0.0412 | Max Min Dist: 0.2482
Anchor Utilization: [5.2%, 9.2%, 2.2%, 1.6%, 24.6%, 18.6%, 0.7%, 15.9%, 3.1%, 18.9%]
Time: 6.02s
```

**Expected Files Generated**:
- `fedanchor/checkpoints/fedanchor_checkpoint.pt`
- `fedanchor/outputs/embeddings_by_class.png` — global test embeddings, colored by digit class
- `fedanchor/outputs/embeddings_by_client.png` — embeddings colored by which client trained them
- `fedanchor/outputs/embeddings_with_anchors.png` — embeddings + anchor star markers
- `fedanchor/outputs/anchor_positions.png` — anchor positions with labels C0_a0..C1_a4

**Communication Summary** (printed at end):
- Encoder: 566,402 scalars, 2,265,608 bytes
- Anchor: 10 scalars, 40 bytes
- Anchor-to-Encoder ratio: 0.0018%
- Total round volume: 4,531,296 bytes × 20 rounds = ~86.4 MB total

---

## Demo 4: Show the Controlled Experiment Evidence

```powershell
# Show the actual 20-round results
.venv\Scripts\python.exe -c "
import csv
print('=== FEDNE 20-ROUND RESULTS ===')
with open('fedanchor/outputs/fedne_vs_fedanchor/fedne_results.csv') as f:
    rows = list(csv.DictReader(f))
for r in [rows[0], rows[9], rows[18]]:
    print(f'Round {r[\"round\"]}: kNN={float(r[\"knn_accuracy\"])*100:.1f}% T={float(r[\"trustworthiness\"]):.4f} C={float(r[\"continuity\"]):.4f}')

print()
print('=== FEDANCHOR 20-ROUND RESULTS ===')
with open('fedanchor/outputs/fedne_vs_fedanchor/fedanchor_results.csv') as f:
    rows2 = list(csv.DictReader(f))
for r in [rows2[0], rows2[4], rows2[9], rows2[19]]:
    print(f'Round {r[\"round\"]}: kNN={float(r[\"knn_accuracy\"])*100:.1f}% T={float(r[\"trustworthiness\"]):.4f} C={float(r[\"continuity\"]):.4f}')
    print(f'  Anchor Utilization: {r[\"anchor_utilization\"]}')
"
```

---

## Demo 5: Show FedAvg Proof

```powershell
.venv\Scripts\python.exe -c "
import torch
from fedanchor.server.server import AnchorServer
server = AnchorServer(input_dim=784, embedding_dim=2, device='cpu')
s1 = server.get_global_encoder_state()
s2 = server.get_global_encoder_state()
for k in s1: s1[k] = torch.ones_like(s1[k]) * 1.0
for k in s2: s2[k] = torch.ones_like(s2[k]) * 3.0
server.aggregate_encoders([s1, s2], [100, 300])
agg = server.get_global_encoder_state()
val = list(agg.values())[0][0,0].item()
print(f'FedAvg test: Client 1 (n=100, param=1.0) + Client 2 (n=300, param=3.0)')
print(f'Expected: (100*1 + 300*3)/400 = 2.5')
print(f'Got: {val}')
print(f'PASS' if abs(val - 2.5) < 0.001 else 'FAIL')
"
```

---

## What Each Output Demonstrates

| Output | What It Proves |
|---|---|
| 15 tests PASS | Every component is individually functional |
| embeddings_by_class.png | Encoder produces meaningful 2D structure |
| embeddings_with_anchors.png | Anchors exist, are learnable, cover the embedding space |
| anchor_positions.png | Two clients have DIFFERENT anchor sets (independent) |
| test_results.txt | FedAvg is exact, gradient isolation is correct |
| fedne_vs_fedanchor CSVs | Controlled 20-round experiment ran on both models |
| communication column | Anchor overhead (40 bytes) is negligible vs encoder (2.27MB) |

---

## Architecture Diagram (Text)

```
                         MNIST (784D)
                              |
                    [/255 Normalize + Flatten]
                              |
              +---------------+---------------+
              |                               |
           Client 0 (30k)               Client 1 (30k)
           [PRIVATE]                    [PRIVATE]
              |                               |
         kNN Graph E_0                   kNN Graph E_1
         [5 neighbors,                  [5 neighbors,
          R^784 Euclidean]               R^784 Euclidean]
              |                               |
         Encoder fθ(x)                  Encoder fθ(x)
         [784→512→256→128→2]            [784→512→256→128→2]
              |                               |
           Z_0 in R^2                     Z_1 in R^2
              |                               |
    +---------+                               +---------+
    |                                                   |
  L_att(Z_0, E_0)    L_anchor(Z_0, A_0)    L_att(Z_1, E_1)    L_anchor(Z_1, A_1)
    |                    |                      |                    |
    |         L_rep(Z_0, A_1) <-[A_1 detached]  L_rep(Z_1, A_0) <-[A_0 detached]
    |                    |                      |                    |
    +----L_total_0-------+                      +----L_total_1-------+
              |                                               |
         backward()                                    backward()
              |                                               |
         [θ_0, A_0 updated]                       [θ_1, A_1 updated]
              |                                               |
              +---------------+---------------+
                              |
                    [FedAvg Aggregation]
                    θ_global = 0.5*θ_0 + 0.5*θ_1
                    A_0, A_1 stored separately
                              |
                    [Broadcast θ_global]
                    [Send A_1 to Client 0]
                    [Send A_0 to Client 1]
                              |
                        Next Round
                              |
                    Global Test Embedding:
                    Z_test = fθ_global(X_test)
                              |
                   Evaluate: T(k), C(k), kNN Accuracy
```
