import torch
import torch.nn as nn
from typing import List

class Encoder(nn.Module):
    """
    Parametric Neighbor Embedding Encoder.
    Maps high-dimensional input features to a low-dimensional representation.
    """
    def __init__(self, input_dim: int, embedding_dim: int = 2, hidden_dims: List[int] = [512, 256, 128]) -> None:
        super(Encoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
