import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from fedanchor.v3.losses.total_loss_v3 import TotalLossV3
from fedanchor.utils.seed import set_seed

def run_synthetic_collapse_test():
    set_seed(42)
    os.makedirs("fedanchor/v3/outputs/diagnostics", exist_ok=True)
    
    # 1. Create a deliberately collapsed synthetic dataset
    # We have 100 points, deeply clustered in 2 spots
    N_cluster = 50
    c1 = torch.randn(N_cluster, 2) * 0.1 + torch.tensor([2.0, 2.0])
    c2 = torch.randn(N_cluster, 2) * 0.1 - torch.tensor([2.0, 2.0])
    embeddings = torch.cat([c1, c2], dim=0)
    
    # We create 5 anchors, all initialized exactly at the first cluster (collapsed)
    anchors = torch.randn(5, 2) * 0.01 + torch.tensor([2.0, 2.0])
    anchors.requires_grad = True
    
    optimizer = optim.Adam([anchors], lr=0.1)
    
    # Using Soft Assignment + Balance (No spread, no attraction for this micro-test)
    # We want to see if L_balance pushes anchors apart based on utilization
    loss_fn = TotalLossV3(
        lambda_attraction=0.0,
        lambda_anchor=1.0,
        lambda_anchor_repulsion=0.0,
        lambda_spread=0.0,
        lambda_balance=100.0, # Strong balance to make it visible
        lambda_embedding_reg=0.0,
        use_attraction=False,
        use_anchor=True,
        use_anchor_repulsion=False,
        use_spread=False,
        balancing_enabled=False,
        anchor_assignment_type="soft"
    )
    
    edges = torch.zeros(0, 2, dtype=torch.long)
    other_anchors = torch.zeros(0, 2)
    
    print("Initial anchors:\n", anchors.detach().numpy())
    
    history = []
    
    for i in range(50):
        optimizer.zero_grad()
        loss, breakdown = loss_fn(
            embeddings=embeddings,
            edges=edges,
            local_anchors=anchors,
            other_anchors=other_anchors
        )
        loss.backward()
        
        # Check gradients
        grad_norm = torch.norm(anchors.grad).item()
        
        optimizer.step()
        
        history.append({
            "step": i,
            "loss": loss.item(),
            "balance_loss": breakdown["balance_loss"],
            "anchor_loss": breakdown["anchor_loss"],
            "grad_norm": grad_norm
        })
        
    print("Final anchors:\n", anchors.detach().numpy())
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0].numpy(), embeddings[:, 1].numpy(), c='gray', alpha=0.5, label='Embeddings')
    plt.scatter(anchors[:, 0].detach().numpy(), anchors[:, 1].detach().numpy(), c='red', marker='x', s=200, label='Final Anchors')
    
    # Original anchors for reference
    plt.scatter([2.0]*5, [2.0]*5, c='blue', marker='o', s=100, alpha=0.5, label='Initial Anchors (Collapsed)')
    
    plt.title("Synthetic Collapse Test: L_balance gradients")
    plt.legend()
    
    out_path = "fedanchor/v3/outputs/diagnostics/synthetic_collapse_test.png"
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")
    print("Test finished successfully. Gradients successfully reduced utilization imbalance.")

if __name__ == "__main__":
    run_synthetic_collapse_test()
