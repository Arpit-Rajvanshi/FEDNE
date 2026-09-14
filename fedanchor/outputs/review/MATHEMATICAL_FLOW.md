# MATHEMATICAL FLOW — FEDANCHOR

> Complete mathematical walkthrough of FEDANCHOR federated training, step by step.
> Every equation corresponds to actual implemented code.

---

## Input

```
x in R^784 — raw MNIST image vector (normalized to [0,1])
Private to each client. Never leaves the local device.
```

---

## Step 1: Encoder fθ(x)

```
z = fθ(x) = W4 * ReLU(W3 * ReLU(W2 * ReLU(W1 * x + b1) + b2) + b3) + b4
```

Where:
- W1 in R^{512 x 784}, b1 in R^{512}
- W2 in R^{256 x 512}, b2 in R^{256}
- W3 in R^{128 x 256}, b3 in R^{128}
- W4 in R^{2 x 128},   b4 in R^{2}

Total parameters θ: 566,402 scalars

Result: z in R^2 (2D embedding)

---

## Step 2: kNN Graph

```
For each client m:
  E_m = {(i,j) : j in NN_k(i)} where NN_k(i) = k nearest neighbors of x_i
                                                  measured in HIGH-dimensional space (R^784)
                                                  excluding self (j ≠ i)
```

- k = 5 (configured)
- Distance: Euclidean in R^784
- Output: edges_m in Z^{N_m * k x 2}
- Built ONCE at client initialization (graph is static across rounds)
- PRIVATE — never shared with server or other clients

---

## Step 3: Anchor Initialization (Round 1 only)

```
Compute Z_m^{(0)} = fθ^{(0)}(X_m) in R^{N_m x 2}   (using initial random encoder)

Strategy "kmeans" (default):
  Run K-Means on Z_m^{(0)} with K=5 clusters
  A_m^{(0)} = {centroid_1, ..., centroid_K}

A_m^{(0)} in R^{K x 2} — K learnable anchor points
```

---

## Step 4: Local Loss Computation

Given:
- Embeddings Z_m = fθ(X_m) in R^{N_m x 2}
- Local anchors A_m in R^{K x 2}
- Other-client anchors A_other = union_{m' ≠ m} A_{m'} in R^{K*(M-1) x 2}  [received from server]

### Step 4a: Attraction Loss

```
L_att = (1/|E_m|) * sum_{(i,j) in E_m} log(1 + ||z_i - z_j||^2 + eps)
```

Gradient: pushes neighbors z_i, z_j closer together.
- d(L_att)/d(z_i) = sum_{j in NN(i)} 2(z_i - z_j) / (|E_m| * (1 + ||z_i - z_j||^2))

### Step 4b: Anchor Coverage Loss

```
L_anchor = (1/N_m) * sum_{i=1}^{N_m} min_{j=1..K} ||z_i - a_j||^2
```

Gradient:
- d(L_anchor)/d(a_j*) = -2(z_i - a_j*) / N_m  where j* = argmin_j ||z_i - a_j||^2
- d(L_anchor)/d(z_i)  =  2(z_i - a_j*) / N_m
(Zero gradient to non-assigned anchors)

### Step 4c: Cross-Client Anchor Repulsion

```
L_rep_raw = (1/(N_m * |A_other|)) * sum_i sum_{a in A_other} log(1 + 1/(||z_i - a||^2 + eps))

L_rep_scaled = scale * L_rep_raw          [scale = 0.01 in production config]
```

- A_other = detached (no gradient to other clients' anchors)
- Gradient to z_i:
  d(L_rep)/d(z_i) = sum_a (-2 * (z_i - a)) / ((||z_i - a||^2 + eps) * (||z_i - a||^2 + eps + 1) * N * K_other) * scale
- This pushes z_i away from other clients' anchor positions

### Step 4d: Total Loss

```
L_total = lambda_att * L_att + lambda_anchor * L_anchor + lambda_rep * L_rep_scaled + lambda_reg * L_reg

With default config:
L_total = 1.0 * L_att + 1.0 * L_anchor + 1.0 * (0.01 * L_rep_raw) + 0.0 * L_reg
```

---

## Step 5: Backpropagation

```
Compute: dL_total / dθ   (encoder parameters θ)
         dL_total / dA_m  (local anchor parameters A_m)

via: loss.backward()    (PyTorch autograd)
```

Both encoder parameters θ and anchor parameters A_m receive gradients.
other_anchors A_other do NOT receive gradients (detached).

---

## Step 6: Gradient Clipping + Local Update

```
clip_grad_norm_(encoder.parameters(), max_norm=1.0)
clip_grad_norm_(anchor_set.parameters(), max_norm=1.0)

theta_m^{(t+1)} = theta_m^{(t)} - lr * grad_theta_m
A_m^{(t+1)} = A_m^{(t)} - lr * grad_A_m

Adam optimizer with lr = 0.001
```

---

## Step 7: Client Upload to Server

```
Client m sends to server:
  theta_m^{(t+1)} in R^{566402}   (encoder parameters, 4 bytes each = 2265608 bytes)
  A_m^{(t+1)} in R^{K x 2}        (anchor positions, K=5 anchors, 10 scalars = 40 bytes)

Total upload per client per round: 2265648 bytes ~ 2.16 MB
```

Private data (X_m, y_m, Z_m, E_m) are NEVER uploaded.

---

## Step 8: FedAvg — Server Aggregation

```
theta_global^{(t+1)} = sum_m (n_m / sum_m n_m) * theta_m^{(t+1)}
```

Where n_m = |X_m| (number of training samples on client m).

With equal IID split (n_0 = n_1 = 30,000):
```
theta_global^{(t+1)} = 0.5 * theta_0^{(t+1)} + 0.5 * theta_1^{(t+1)}
```

Anchors are NOT averaged. Each client's anchors remain separate on the server:
```
server.client_anchors[0] = A_0^{(t+1)}
server.client_anchors[1] = A_1^{(t+1)}
```

---

## Step 9: Server Broadcast

```
Client m receives from server:
  theta_global^{(t+1)}              (global encoder, 2265608 bytes)
  A_other^{(t+1)} = A_{m'}^{(t+1)} for all m' != m  (other anchors, 40 bytes each)

Total download per client per round: 2265608 + 40 = 2265648 bytes ~ 2.16 MB
```

---

## Step 10: Next Round

Return to Step 4. Encoder is refreshed from theta_global. Anchors continue from where they left off.

---

## Final Embedding

After R rounds, the global encoder theta_global produces the final 2D embeddings:
```
Z_final = f_{theta_global}(X_test) in R^{N_test x 2}
```

Evaluated by:
- Trustworthiness T(k=7)
- Continuity C(k=7)
- kNN classification accuracy (k=7)
- Anchor utilization and geometric diagnostics
