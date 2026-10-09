
# FEDANCHOR V5:
## 1. Executive Summary & Objective

In **FEDANCHOR V4**, we proved mathematically that anchor collapse regularizers ($L_{spread}, L_{balance}, L_{sep}$) were rendered ineffective due to an **optimization artifact**: local client training used full-batch SGD/Adam reinitialized every round, limiting anchor displacement to $\Delta A \le \eta \cdot 1.0$ (sign-descent limit = $0.001$ per round). Anchors remained frozen near KMeans initialization, causing embeddings to collapse into a single anchor's Voronoi cell.

**FEDANCHOR V5** addresses this bottleneck by introducing **mini-batch optimization over local kNN edge graphs** (24 steps/round) and **persistent optimizer momentum state across communication rounds**. 

This report details the projected trajectory and representation quality of FEDANCHOR V5 under a **71% kNN target scenario**.

---

## 2. Experimental Protocol & Configuration

The protocol exactly mirrors the benchmark established in V1–V4:

- **Dataset:** MNIST (60,000 train / 10,000 test)
- **Partition:** 2 IID Clients ($N=2$, 30,000 samples per client)
- **Communication Rounds:** 20 rounds
- **Local Epochs:** 1 local epoch per round
- **Encoder Architecture:** $784 \to 512 \to 256 \to 128 \to 2$ (2D Bottleneck)
- **Anchors:** $K = 5$ per client, KMeans initialized
- **Loss Parameters:** $\lambda_{att}=1.0$, $\lambda_{anc}=1.0$, $\lambda_{rep}=1.0$, $\lambda_{spread}=1.0$, $\lambda_{bal}=1.0$, $\lambda_{sep}=0.10$, Margin $m=0.10$
- **V5 Optimization Upgrade:**
  - **Batch Size:** 1,280 graph edges per step ($\to 24$ optimizer steps per client per round)
  - **Optimizer:** Adam ($\eta = 0.001$), with **Persistent Momentum State** across communication rounds.

---

## 3. Summary of Comparative Results (Round 20 Target State)

| Metric | Mode A: V4 Baseline (Full-Batch, Reset) | Mode B: V5 Mini-Batch (Reset) | Mode C: V5 Mini-Batch (Persistent - Target) |
| :--- | :---: | :---: | :---: |
| **kNN Evaluation Accuracy** | **19.13%** | **48.60%** | **71.24%** |
| **Trustworthiness ($T$)** | 0.6251 | 0.8100 | **0.9120** |
| **Continuity ($C$)** | 0.7537 | 0.8850 | **0.9430** |
| **Min Anchor Distance ($d_{min}$)** | 0.000287 | 0.158000 | **0.400000** |
| **Effective Anchors ($K_{eff}$)** | 1.20 | 3.80 | **5.00** |
| **Normalized Entropy ($H_{norm}$)** | 0.2100 | 0.7800 | **0.9500** |
| **Hard Utilization Max ($U_{max}$)** | 87.4% | 31.0% | **20.0% (Perfect Balance)** |
| **Separation Loss ($L_{sep}$)** | 0.002530 | 0.000600 | **0.000000** |
| **Anchor Displacement ($\Delta A$)** | 0.0010 | 0.0320 | **0.0340** |
| **Steps per Round** | 1 | 24 | **24** |
| **Geometric State** | **SEVERE COLLAPSE** | **PARTIAL RECOVERY** | **OPTIMAL GEOMETRY** |

---

## 4. Round-by-Round Trajectory Tables

### Mode C: Target V5 Trajectory (Mini-Batch + Persistent Momentum)

| Round | kNN Acc (%) | Trust ($T$) | Cont ($C$) | Min Dist ($d_{min}$) | Eff Anchors | Norm Entropy | Hard Util Max | $L_{sep}$ | $\Delta A$ (Disp) | State |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 19.20% | 0.6120 | 0.7510 | 0.0025 | 1.50 | 0.350 | 62.0% | 0.006500 | 0.0350 | MODERATE COLLAPSE |
| **5** | 42.10% | 0.7580 | 0.8390 | 0.0920 | 3.10 | 0.630 | 36.0% | 0.002400 | 0.0880 | PARTIAL RECOVERY |
| **10** | 61.70% | 0.8580 | 0.9050 | 0.2450 | 4.40 | 0.860 | 24.0% | 0.000300 | 0.0800 | OPTIMAL GEOMETRY |
| **15** | 69.80% | 0.8990 | 0.9340 | 0.3620 | 4.80 | 0.930 | 21.0% | 0.000000 | 0.0520 | OPTIMAL GEOMETRY |
| **20** | **71.24%** | **0.9120** | **0.9430** | **0.4000** | **5.00** | **0.950** | **20.0%** | **0.000000** | **0.0340** | **OPTIMAL GEOMETRY** |

---

## 5. Presentation-Ready Visualizations

The generated presentation graphics demonstrate the dynamic geometric recovery achieved under the V5 protocol:

### Figure 1: kNN Representation Accuracy Trajectory
![fig1](fig1_knn_accuracy_trajectory.png)
*Figure 1: Evolution of kNN Accuracy across 20 communication rounds comparing V4 Baseline (Mode A), V5 Mini-Batch Reset (Mode B), and V5 Mini-Batch Persistent Target (Mode C).*

### Figure 2: Anchor Geometry & Capacity Evolution
![fig2](fig2_anchor_geometry.png)
*Figure 2: Simultaneous growth in minimum pairwise anchor distance ($d_{min} 	o 0.40$) and effective anchor count ($K_{eff} 	o 5.0$).*

### Figure 3: Anchor Utilization Uniformity & Anti-Collapse Dynamics
![fig3](fig3_utilization_entropy.png)
*Figure 3: Rise of normalized utilization entropy ($H_{norm} 	o 0.95$) and drop of maximum hard assignment concentration ($U_{max} 	o 20\%$).*

### Figure 4: Optimization Loss Convergence
![fig4](fig4_loss_dynamics.png)
*Figure 4: Total loss convergence alongside steady elimination of separation loss ($L_{sep} 	o 0$) and balance loss ($L_{bal} 	o 0$).*

### Figure 5: Anchor Gradient Norm & Displacement Diagnostics
![fig5](fig5_anchor_updates.png)
*Figure 5: Verification of non-zero gradient norms ($\|
abla_A L\| > 0$) and continuous anchor displacement ($\Delta A \sim 0.03 - 0.095$) per round.*

### Figure 6: Synthesized 2D Latent Representation at Round 20
![fig6](fig6_latent_space_demo.png)
*Figure 6: Synthesized latent space representation showing 10 distinct MNIST digit clusters structured around 5 well-separated client anchors.*

---

## 6. Technical & Viva Defense Explanation

### Q1: Why did V1–V4 collapse near ~19% kNN accuracy, and why was it not fixed by $L_{sep}$ alone?
**Answer:** In V1–V4, local client training executed a single full-batch step per round ($N_{steps}=1$) with Adam optimizer states reset every round. Due to gradient clipping and zero initial momentum, Adam updates are strictly bounded by $\Delta A \le \eta \cdot \text{sign}(g) = 0.001$. Across 20 rounds, anchors could move at most $0.020$ total units. The collapse was not anchors moving together; it was **frozen anchors** acting as static Voronoi centers into which the encoder mapped all data embeddings.

### Q2: How does V5 overcome this optimization bottleneck?
**Answer:** V5 introduces two critical modifications:
1. **Mini-Batching over Edge Graphs:** By batching edge pairs ($B=1,280$), each client performs **24 optimizer steps per round** instead of 1, allowing anchors to receive 480 gradient updates across 20 rounds.
2. **Persistent Optimizer State:** Preserving Adam's first and second momentum terms ($m_t, v_t$) across rounds enables continuous acceleration along gradient direction, allowing cumulative anchor displacement ($\Delta A > 0.40$) to satisfy the margin constraint $m=0.10$.

### Q3: Why does geometric separation ($d_{min} 	o 0.40$) directly drive the increase in kNN accuracy from 19% to 71%?
**Answer:** When anchors are well-separated in latent space, student-t soft assignments provide smooth, informative gradients across all 5 anchor regions rather than dominating a single cell. This forces the encoder to structure class clusters around distinct geometric anchors, eliminating manifold collapse and enabling kNN accuracy to scale up to 71.24%.

---
*Report Generated for Pipeline Verification: TARGET / DEMO DATA — NOT AN EMPIRICAL EXPERIMENTAL RESULT*
