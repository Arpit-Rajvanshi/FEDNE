# FEDANCHOR V4 Optimization Audit

Based on a direct code inspection of the V4 codebase (`v4/client/client_v4.py`, `v4/training/trainer_v4.py`), the following optimization protocol behaviors were identified:

1. **Number of optimizer steps per client per round:** 1 (controlled by `local_epochs=1`, executing one full pass).
2. **Whether training is full-batch or mini-batch:** Full-batch.
3. **Exact batch size:** The entire local dataset ($N=30,000$ for MNIST partitioned among 2 clients).
4. **Whether optimizer is recreated every round:** Yes. `optim.Adam` is freshly instantiated inside the `local_train` method every round.
5. **Whether Adam state persists between rounds:** No. Because it is re-instantiated, momentum and velocity terms are reset to zero.
6. **Whether anchors have `requires_grad=True`:** Yes (defined as `nn.Parameter` in `models/anchor.py`).
7. **Whether anchors are included in the optimizer parameter list:** Yes. Passed via `list(self.encoder.parameters()) + list(self.anchor_set.parameters())`.
8. **Whether anchor gradients are non-zero:** Yes. A previous diagnostic scale check confirmed massive gradient norms (e.g., ~526,000) heavily driven by $L_{spread}$ at initialization.
9. **Whether gradient clipping is applied:** Yes. `torch.nn.utils.clip_grad_norm_` is applied independently to both encoder parameters and anchor parameters.
10. **Exact `max_grad_norm`:** 1.0 (default in config).
11. **Anchor update magnitude per local step:** Exactly 0.001. Because Adam momentum is initialized to 0, the first step is bounded precisely by $lr \times \text{sign}(g)$. With the $L_{spread}$ gradient exploding and clipping scaling the vector to norm 1.0, the anchors step by exactly the learning rate $0.001$.
12. **Anchor update magnitude per round:** 0.001 (since there is only 1 local step per round). Over 20 rounds, total maximum physical displacement is strictly bounded to $0.02$.
13. **Whether anchors are modified directly anywhere outside optimizer.step():** No. Except for the initial `KMeans` initialization in round 1, they are purely updated via gradients.
14. **Whether anchors are averaged/FedAvg'd:** No. They are maintained independently per client on the server.
15. **Whether cross-client anchors are detached:** Yes. The server distributes them as detached tensors.

### Conclusion of Audit
The original V1-V4 experiments are heavily bounded by a **Sign Descent Limit** artifact. Full-batch training with fresh Adam optimizers and 1 step per round completely freezes the anchors in space. The previously observed "collapse" is the data embeddings collapsing into the territory of these frozen anchors.
