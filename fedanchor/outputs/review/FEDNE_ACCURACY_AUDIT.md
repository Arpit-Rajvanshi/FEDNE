# FEDNE ACCURACY AUDIT — SCIENTIFIC STATUS

---

## FEDNE Implementation Assessment

### What FEDNE Implements

FEDNE (our baseline) uses this combined loss per client m:

```
L_m = lambda_att * L_att + lambda_rep_local * (n_m/n) * L_rep_local + lambda_rep_surr * L_rep_surrogate
```

**Attraction Loss**:
```
L_att = -E_{(i,j)} [log phi(z_i, z_j)]
      = E_{(i,j)} [log(1 + ||z_i - z_j||^2)]    (implemented in fedne/losses/attraction.py)
```
- phi(z_i, z_j) = 1 / (1 + ||z_i - z_j||^2)
- Uses FEDNE kNN graph on local data
- MATCHES standard neighbor embedding formulation

**Local Repulsion Loss**:
```
L_rep_local = -E_i sum_{s=1}^b log(1 - phi(z_i, z_neg_s))    (implemented in fedne/losses/repulsion.py)
```
- Negative samples z_neg are randomly drawn from the LOCAL client's dataset
- This repels embeddings from local random points
- MATCHES standard NE local repulsion

**Surrogate Cross-Client Repulsion**:
```
L_rep_surrogate = sum_{m' != m} (n_m' / n) * E_i [surrogate_m'(z_i)]
```
- Each client trains a small MLP surrogate to approximate the local repulsion density of other clients
- The surrogate predicts "how crowded is this region for client m'?"
- Implemented in `fedne/losses/contrastive_loss.py`
- This is FEDNE's key mechanism for cross-client repulsion

**FedAvg**:
- `fedne/server/server.py` performs weighted averaging of encoder weights
- Also: `server.aggregate_encoder(self.clients)` aggregates client encoder weights
- `server.collect_surrogates(self.clients)` collects surrogate models from clients

---

## Is Our FEDNE Faithful to the Paper?

### What Is Faithful:
- Attraction loss formula: exact match to neighbor embedding formulation
- Local repulsion loss: exact match to standard NE negative sampling
- Surrogate concept: yes — using learned function to approximate other-client density
- FedAvg for encoder: yes

### What Is Uncertain / Approximate:
- Specific hyperparameters from the FEDNE paper (if a specific FEDNE paper is the reference):
  we used k=5, surrogate hidden_dim=64, grid_step=0.3, margin=0.5, lr=0.001, epochs=2
- The surrogate architecture (64-unit MLP) may not match the paper's specific design
- The paper may use a different surrogate training loss or grid structure

**Conclusion**: Our FEDNE is a good-faith re-implementation of the surrogate-assisted federated
neighbor embedding approach. It is NOT guaranteed to reproduce exact numbers from the original
publication. State this clearly in the presentation.

---

## FEDNE vs FEDANCHOR — Scientific Distinction

### FEDNE:
- Cross-client repulsion via SURROGATE MODEL (learned MLP approximator)
- Surrogate is trained each round on local repulsion values
- Communication: encoder weights + surrogate model weights
- "What does the other client's repulsion function look like?" → answer: train an MLP

### FEDANCHOR:
- Cross-client repulsion via ANCHOR POINTS (K=5 learnable 2D points)
- Anchors learn to cover the client's embedding distribution
- Communication: encoder weights + K*2 floats (40 bytes)
- "What does the other client's embedding space look like?" → answer: share K representative points

### Key Claim:
"FEDNE uses a surrogate MLP (~thousands of parameters) to model cross-client repulsion.
FEDANCHOR uses K=5 anchor points (10 floats = 40 bytes) as a simpler, more transparent
alternative. This reduces communication overhead for the cross-client representation from
~kilobytes to 40 bytes, at the cost of a less accurate representation."

### Expected Benefit (in theory):
- Simpler and more interpretable cross-client information
- Near-zero communication overhead for anchor sharing
- Anchors are directly interpretable in 2D embedding space

### Observed Reality (from experiments):
- The current FEDANCHOR prototype does NOT match FEDNE's embedding quality
- Trustworthiness: FEDNE 0.933 vs FEDANCHOR 0.627
- This is expected for a prototype — surrogate models provide much richer cross-client signal
- The anchors do collapse to 1-2 dominant points, limiting their effectiveness

---

## What FEDANCHOR's Equations Are NOT

> [!IMPORTANT]
> The FEDANCHOR equations (L_anchor coverage loss, L_rep anchor repulsion) are our PROPOSED
> formulations. They are NOT equations from the FEDNE paper. Do not present them as such.

The FEDNE paper proposes the surrogate mechanism.
We propose the anchor mechanism as an ALTERNATIVE.

This distinction is critical for academic integrity.
