import torch
import torch.nn as nn
from typing import List

class Encoder(nn.Module):
    """
    Parametric Dimensionality Reduction Encoder.
    Maps high-dimensional input vectors (e.g. 784) to a 2D embedding space.
    Architecture matching baseline: Linear(784, 512) -> ReLU -> Linear(512, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 2).
    """
    def __init__(
        self,
        input_dim: int = 784,
        embedding_dim: int = 2,
        hidden_dims: List[int] = [512, 256, 128]
    ) -> None:
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
