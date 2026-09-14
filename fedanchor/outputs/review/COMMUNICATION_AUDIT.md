# COMMUNICATION COST AUDIT

## Overview
This document breaks down the exact communication overhead per federated round for both FEDNE and FEDANCHOR, computed directly from the neural network architectures in the source code.

## 1. Encoder Communication (Common to both)
Both FEDNE and FEDANCHOR use identical encoder architectures:
- `Linear(784, 512)`: 401,408 weights + 512 biases = 401,920 parameters
- `Linear(512, 256)`: 131,072 weights + 256 biases = 131,328 parameters
- `Linear(256, 128)`: 32,768 weights + 128 biases = 32,896 parameters
- `Linear(128, 2)`: 256 weights + 2 biases = 258 parameters

**Total Encoder Parameters**: 566,402
**Encoder Bytes (Float32)**: 566,402 × 4 bytes = **2,265,608 bytes** (~2.27 MB) per client.

## 2. Cross-Client Information Exchange

### FEDNE: Surrogate Model
FEDNE shares a small MLP surrogate per client to approximate local repulsion density.
- `Linear(2, 64)`: 128 weights + 64 biases = 192 parameters
- `Linear(64, 1)`: 64 weights + 1 bias = 65 parameters

**Total Surrogate Parameters**: 257
**Surrogate Bytes (Float32)**: 257 × 4 bytes = **1,028 bytes** (~1.03 KB) per client.

### FEDANCHOR: Anchor Points
FEDANCHOR shares a fixed number of 2D anchor points.
- K = 5 anchors
- 2 coordinates (x, y) per anchor

**Total Anchor Parameters**: 10
**Anchor Bytes (Float32)**: 10 × 4 bytes = **40 bytes** per client.

## 3. Total Per-Round Communication (2 Clients)

### FEDNE (Baseline)
- **Upload per client**: Encoder (2,265,608) + Surrogate (1,028) = 2,266,636 bytes
- **Total Upload (2 clients)**: 4,533,272 bytes
- **Total Download (2 clients)**: 4,533,272 bytes
- **Total Round Bandwidth**: **9,066,544 bytes** (~9.07 MB)

### FEDANCHOR (Proposed)
- **Upload per client**: Encoder (2,265,608) + Anchors (40) = 2,265,648 bytes
- **Total Upload (2 clients)**: 4,531,296 bytes
- **Total Download (2 clients)**: 4,531,296 bytes
- **Total Round Bandwidth**: **9,062,592 bytes** (~9.06 MB)

## 4. Discrepancy Resolution
Previous documents cited `4,531,296 bytes` vs `9,062,592 bytes` inconsistently.
- `4,531,296 bytes` is the **Total Upload** for all 2 clients in FEDANCHOR.
- `9,062,592 bytes` is the **Total Upload + Download** for all 2 clients in FEDANCHOR.
- The hardcoded FEDNE approximation of `10,000 bytes` for the surrogate model in older scripts was an overestimate; the actual surrogate size is `1,028 bytes`.

## 5. Scientific Conclusion
The communication savings of FEDANCHOR (40 bytes vs 1,028 bytes for cross-client representation) are mathematically correct. 

However, because the encoder itself requires ~2.27 MB per client, the overall communication reduction of FEDANCHOR compared to FEDNE is negligible (0.04% total bandwidth reduction).

FEDANCHOR's true advantage lies in **simplicity and interpretability** rather than practical bandwidth savings, as 1,028 bytes (FEDNE's surrogate) is already a trivial transmission cost on modern networks.
