import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Union

class SurrogateRepulsion(nn.Module):
    """
    Surrogate MLP model to approximate client-specific repulsion losses in 2D space.
    """
    def __init__(self, hidden_dim: int = 64) -> None:
        super(SurrogateRepulsion, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict repulsion loss for query points z of shape [B, 2].
        Returns shape [B, 1].
        """
        return self.network(z)

def generate_grid_query_points(
    embeddings: np.ndarray,
    grid_step: float = 0.3,
    margin: float = 0.5,
    samples_per_axis: Union[int, str] = "auto"
) -> np.ndarray:
    """
    Generate grid query points Z_q in 2D space based on the bounding box of client's embeddings.
    """
    min_x, min_y = np.min(embeddings, axis=0) - margin
    max_x, max_y = np.max(embeddings, axis=0) + margin
    
    if isinstance(samples_per_axis, int):
        x_vals = np.linspace(min_x, max_x, samples_per_axis)
        y_vals = np.linspace(min_y, max_y, samples_per_axis)
    else:
        # Calculate expected number of points based on step size
        num_x = int((max_x - min_x) / grid_step) + 1
        num_y = int((max_y - min_y) / grid_step) + 1
        
        # Cap the number of points per axis to 100 (100x100 = 10000 points max) to prevent memory overflow
        max_points = 100
        
        if num_x > max_points:
            x_vals = np.linspace(min_x, max_x, max_points)
        else:
            x_vals = np.arange(min_x, max_x + grid_step / 2, grid_step)
            
        if num_y > max_points:
            y_vals = np.linspace(min_y, max_y, max_points)
        else:
            y_vals = np.arange(min_y, max_y + grid_step / 2, grid_step)
            
        if len(x_vals) < 2:
            x_vals = np.linspace(min_x, max_x, 5)
        if len(y_vals) < 2:
            y_vals = np.linspace(min_y, max_y, 5)
            
    # Create the grid
    xx, yy = np.meshgrid(x_vals, y_vals)
    query_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return query_points.astype(np.float32)

def compute_repulsion_targets(
    query_points: np.ndarray,
    embeddings: np.ndarray,
    b: int,
    with_replacement: bool = True
) -> np.ndarray:
    """
    Compute true repulsion targets for grid query points against local embeddings.
    l_q_i = - sum_{j=1}^b log(1 - phi(zq_i, z_m^(j)))
    where phi(z1, z2) = 1 / (1 + ||z1 - z2||^2).
    """
    num_queries = len(query_points)
    num_embeddings = len(embeddings)
    
    # Pre-sample random embeddings for each query point
    # We sample shape [num_queries, b] indices
    if with_replacement:
        sampled_indices = np.random.choice(num_embeddings, size=(num_queries, b), replace=True)
    else:
        sampled_indices = np.zeros((num_queries, b), dtype=np.int64)
        for i in range(num_queries):
            sampled_indices[i] = np.random.choice(num_embeddings, size=b, replace=False)
            
    # Fetch actual coordinate embeddings: [num_queries, b, 2]
    sampled_emb = embeddings[sampled_indices]
    
    # Compute distances: query_points: [num_queries, 2] -> [num_queries, 1, 2]
    # Distances squared: [num_queries, b]
    diff = np.expand_dims(query_points, axis=1) - sampled_emb
    d2 = np.sum(diff ** 2, axis=-1)
    
    # phi = 1 / (1 + d2)
    phi = 1.0 / (1.0 + d2)
    
    # Target = - sum_j log(1 - phi_j)
    # Clamp to prevent log(0)
    phi = np.clip(phi, 0.0, 1.0 - 1e-7)
    targets = -np.sum(np.log(1.0 - phi), axis=-1)
    
    return targets.astype(np.float32)
