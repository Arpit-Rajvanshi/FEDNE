import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def plot_embedding_by_class(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "Embedding by Class"
) -> None:
    """
    Generate and save a 2D scatter plot colored by class label.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    unique_classes = np.unique(labels)
    cmap = plt.colormaps.get_cmap("tab10")
    
    for idx, cls in enumerate(unique_classes):
        mask = labels == cls
        plt.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            color=cmap(idx % 10),
            label=str(cls),
            alpha=0.6,
            edgecolors='none',
            s=5
        )
        
    plt.legend(title="Class", markerscale=4, loc="upper right")
    plt.title(title, fontsize=14)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_embedding_by_client(
    embeddings: np.ndarray,
    client_ids: np.ndarray,
    save_path: str,
    title: str = "Embedding by Client"
) -> None:
    """
    Generate and save a 2D scatter plot colored by client ID.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    unique_clients = np.unique(client_ids)
    cmap = plt.colormaps.get_cmap("rainbow")
    
    for idx, client in enumerate(unique_clients):
        mask = client_ids == client
        plt.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            color=cmap(idx / max(1, len(unique_clients) - 1)),
            label=f"Client {client}",
            alpha=0.6,
            edgecolors='none',
            s=5
        )
        
    plt.legend(title="Client", markerscale=4, loc="upper right", ncol=2 if len(unique_clients) > 10 else 1)
    plt.title(title, fontsize=14)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
