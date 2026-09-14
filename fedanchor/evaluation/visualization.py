import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List

def plot_embeddings_by_class(
    X_low: np.ndarray,
    y: np.ndarray,
    save_path: str
) -> None:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_low[:, 0], X_low[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(scatter, label="Digit Class")
    plt.title("FEDANCHOR: 2D Embeddings by Class")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_embeddings_by_client(
    client_embeddings: List[np.ndarray],
    save_path: str
) -> None:
    plt.figure(figsize=(8, 6))
    colors = plt.get_cmap('tab10', len(client_embeddings))
    for client_id, emb in enumerate(client_embeddings):
        plt.scatter(emb[:, 0], emb[:, 1], color=colors(client_id), s=5, alpha=0.5, label=f"Client {client_id}")
    plt.title("FEDANCHOR: 2D Embeddings by Client")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_embeddings_with_anchors(
    client_embeddings: List[np.ndarray],
    client_anchors: Dict[int, torch.Tensor],
    save_path: str
) -> None:
    plt.figure(figsize=(9, 7))
    colors = plt.get_cmap('tab10', len(client_embeddings))
    
    for client_id, emb in enumerate(client_embeddings):
        c = colors(client_id)
        plt.scatter(emb[:, 0], emb[:, 1], color=c, s=6, alpha=0.3, label=f"Client {client_id} Points")
        
        if client_id in client_anchors:
            anc = client_anchors[client_id].numpy()
            plt.scatter(
                anc[:, 0], anc[:, 1],
                color=c, marker='*', s=200, edgecolors='black', linewidths=1.5,
                label=f"Client {client_id} Anchors"
            )
            
    plt.title("FEDANCHOR: Embeddings & Learned Client Anchors")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_anchor_positions(
    client_anchors: Dict[int, torch.Tensor],
    save_path: str
) -> None:
    plt.figure(figsize=(7, 6))
    colors = plt.get_cmap('tab10', len(client_anchors))
    
    for client_id, anc_tensor in client_anchors.items():
        anc = anc_tensor.numpy()
        c = colors(client_id)
        plt.scatter(
            anc[:, 0], anc[:, 1],
            color=c, marker='X', s=150, edgecolors='black',
            label=f"Client {client_id} ({len(anc)} anchors)"
        )
        for idx, (x, y) in enumerate(anc):
            plt.annotate(f"C{client_id}_a{idx}", (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
            
    plt.title("FEDANCHOR: Final Client Anchor Positions")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_all_visualizations(
    client_embeddings: List[np.ndarray],
    global_test_embeddings: np.ndarray,
    y_test: np.ndarray,
    client_anchors: Dict[int, torch.Tensor],
    output_dir: str
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    
    p1 = os.path.join(output_dir, "embeddings_by_class.png")
    p2 = os.path.join(output_dir, "embeddings_by_client.png")
    p3 = os.path.join(output_dir, "embeddings_with_anchors.png")
    p4 = os.path.join(output_dir, "anchor_positions.png")
    
    plot_embeddings_by_class(global_test_embeddings, y_test, p1)
    plot_embeddings_by_client(client_embeddings, p2)
    plot_embeddings_with_anchors(client_embeddings, client_anchors, p3)
    plot_anchor_positions(client_anchors, p4)
    
    return {
        "embeddings_by_class": p1,
        "embeddings_by_client": p2,
        "embeddings_with_anchors": p3,
        "anchor_positions": p4
    }
