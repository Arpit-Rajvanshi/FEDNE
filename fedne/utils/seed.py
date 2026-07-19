import random
import os
import numpy as np
import torch

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set seeds for python random, numpy, and torch to ensure reproducibility.
    If deterministic is True, enforces PyTorch deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Needed for some deterministic operations
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
