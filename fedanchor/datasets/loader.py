import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import torch
from torchvision import datasets, transforms

def partition_data(
    X: np.ndarray,
    y: np.ndarray,
    partition_type: str,
    num_clients: int,
    **kwargs
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition high-dimensional dataset (X, y) among clients.
    Supported types: 'iid', 'dirichlet'.
    """
    num_samples = len(X)
    
    if partition_type == "iid":
        indices = np.random.permutation(num_samples)
        client_indices = np.array_split(indices, num_clients)
        client_data = [(X[idx], y[idx]) for idx in client_indices]
        
    elif partition_type == "dirichlet":
        alpha = kwargs.get("alpha", 0.5)
        num_classes = len(np.unique(y))
        min_size = 0
        
        while min_size < 10:
            client_indices = [[] for _ in range(num_clients)]
            for c in range(num_classes):
                idx_c = np.where(y == c)[0]
                np.random.shuffle(idx_c)
                proportions = np.random.dirichlet([alpha] * num_clients)
                proportions = np.array([p * (len(idx_c) >= num_clients) for p in proportions])
                if proportions.sum() == 0:
                    proportions = np.ones(num_clients) / num_clients
                else:
                    proportions = proportions / proportions.sum()
                
                proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
                splits = np.split(idx_c, proportions)
                for client_id in range(num_clients):
                    client_indices[client_id].extend(splits[client_id])
            
            min_size = min(len(idx) for idx in client_indices)
            
        for i in range(num_clients):
            np.random.shuffle(client_indices[i])
            
        client_data = [(X[client_indices[i]], y[client_indices[i]]) for i in range(num_clients)]
    else:
        raise ValueError(f"Unsupported partition type: {partition_type}")
        
    # Optional subsampling cap (useful for unit tests / fast prototyping)
    max_samples = kwargs.get("max_samples_per_client", None)
    if max_samples is not None and max_samples > 0:
        capped_data = []
        for X_c, y_c in client_data:
            if len(X_c) > max_samples:
                capped_data.append((X_c[:max_samples], y_c[:max_samples]))
            else:
                capped_data.append((X_c, y_c))
        client_data = capped_data
        
    return client_data

def load_dataset(
    name: str,
    data_dir: str,
    partition_type: str,
    num_clients: int,
    **kwargs
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Load dataset (MNIST) and partition it for federated client execution.
    Returns:
        client_data: List of (X_train, y_train) for each client.
        test_data: (X_test, y_test) global test set.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    if name.upper() == "MNIST":
        train_set = datasets.MNIST(data_dir, train=True, download=True)
        test_set = datasets.MNIST(data_dir, train=False, download=True)
    elif name.upper() == "FASHIONMNIST":
        train_set = datasets.FashionMNIST(data_dir, train=True, download=True)
        test_set = datasets.FashionMNIST(data_dir, train=False, download=True)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
        
    X_train = train_set.data.numpy().astype(np.float32) / 255.0
    y_train = train_set.targets.numpy().astype(np.int64)
    
    X_test = test_set.data.numpy().astype(np.float32) / 255.0
    y_test = test_set.targets.numpy().astype(np.int64)
    
    # Flatten images to vectors [N, 784]
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    client_data = partition_data(X_train, y_train, partition_type, num_clients, **kwargs)
    
    return client_data, (X_test, y_test)
