import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path so fedanchor can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from fedanchor.losses.anchor_loss import AnchorCoverageLoss
from fedanchor.v2.losses.anchor_soft_loss import AnchorSoftLoss
from fedanchor.v2.losses.anchor_spread_loss import AnchorSpreadLoss

def create_synthetic_data(num_clusters=3, pts_per_cluster=100, seed=42):
    torch.manual_seed(seed)
    centers = [
        torch.tensor([0.0, 5.0]),
        torch.tensor([-5.0, -5.0]),
        torch.tensor([5.0, -5.0])
    ]
    data = []
    labels = []
    for i, c in enumerate(centers):
        cluster = c + 0.5 * torch.randn(pts_per_cluster, 2)
        data.append(cluster)
        labels.append(torch.full((pts_per_cluster,), i))
    return torch.cat(data, dim=0), torch.cat(labels, dim=0)

def run_experiment(name, data, init_anchors, loss_fn_type, lambda_spread=0.0, iters=100, lr=0.1):
    anchors = init_anchors.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([anchors], lr=lr)
    
    if loss_fn_type == 'hard':
        anchor_loss = AnchorCoverageLoss()
    elif loss_fn_type == 'soft':
        anchor_loss = AnchorSoftLoss(alpha=1.0)
    
    spread_loss = AnchorSpreadLoss(eps=1e-7)
    
    for i in range(iters):
        optimizer.zero_grad()
        
        if loss_fn_type == 'hard':
            l_anc = anchor_loss(data, anchors)
        else:
            l_anc, _ = anchor_loss(data, anchors)
            
        l_spr = spread_loss(anchors)
        
        loss = l_anc + lambda_spread * l_spr
        loss.backward()
        optimizer.step()
        
    return anchors.detach()

def main():
    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)
    
    data, labels = create_synthetic_data()
    
    # Initialize 5 anchors near the origin (dense configuration to induce collapse if possible)
    torch.manual_seed(123)
    init_anchors = 0.1 * torch.randn(5, 2)
    
    # 1. Hard Assignment (V1)
    anchors_hard = run_experiment("Hard", data, init_anchors, "hard")
    
    # 2. Soft Assignment (V2 partial)
    anchors_soft = run_experiment("Soft", data, init_anchors, "soft")
    
    # 3. Soft + Spread (V2 full)
    anchors_soft_spread = run_experiment("Soft+Spread", data, init_anchors, "soft", lambda_spread=10.0)
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    def plot_ax(ax, title, anc):
        ax.scatter(data[:, 0].numpy(), data[:, 1].numpy(), c=labels.numpy(), cmap='tab10', alpha=0.3, s=10)
        ax.scatter(anc[:, 0].numpy(), anc[:, 1].numpy(), c='red', marker='X', s=200, edgecolors='black', linewidth=2)
        ax.set_title(title)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        
    plot_ax(axes[0], "Init", init_anchors)
    plot_ax(axes[1], "Hard Assignment (V1 Collapse)", anchors_hard)
    plot_ax(axes[2], "Soft Assignment", anchors_soft)
    plot_ax(axes[3], "Soft + Spread", anchors_soft_spread)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "micro_experiment_results.png"))
    print("Micro-experiment completed and saved.")

if __name__ == "__main__":
    main()
