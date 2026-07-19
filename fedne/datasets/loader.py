import os
import numpy as np
from typing import Dict, List, Tuple, Any
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
    Partition the dataset (X, y) among num_clients according to partition_type.
    """
    if partition_type == "iid":
        num_samples = len(X)
        indices = np.random.permutation(num_samples)
        client_indices = np.array_split(indices, num_clients)
        return [(X[idx], y[idx]) for idx in client_indices]
        
    elif partition_type == "dirichlet":
        alpha = kwargs.get("alpha", 0.2)
        num_classes = len(np.unique(y))
        min_size = 0
        
        # Ensure that no client gets an empty dataset
        while min_size < 10:
            client_indices = [[] for _ in range(num_clients)]
            for c in range(num_classes):
                idx_c = np.where(y == c)[0]
                np.random.shuffle(idx_c)
                proportions = np.random.dirichlet([alpha] * num_clients)
                # Filter out proportions if too small and normalize
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
            
        return [(X[client_indices[i]], y[client_indices[i]]) for i in range(num_clients)]
        
    elif partition_type == "shards":
        shards_per_client = kwargs.get("shards_per_client", 2)  # C classes per client
        num_classes = len(np.unique(y))
        
        # Assign classes to each client
        client_classes = [[] for _ in range(num_clients)]
        class_pool = list(range(num_classes))
        np.random.shuffle(class_pool)
        pool_idx = 0
        
        for client_id in range(num_clients):
            while len(client_classes[client_id]) < shards_per_client:
                c = class_pool[pool_idx]
                if c not in client_classes[client_id]:
                    client_classes[client_id].append(c)
                pool_idx = (pool_idx + 1) % len(class_pool)
                if pool_idx == 0:
                    np.random.shuffle(class_pool)
                    
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            assigned_clients = [i for i in range(num_clients) if c in client_classes[i]]
            if len(assigned_clients) == 0:
                continue
            
            idx_c = np.where(y == c)[0]
            np.random.shuffle(idx_c)
            
            splits = np.array_split(idx_c, len(assigned_clients))
            for client_id, split in zip(assigned_clients, splits):
                client_indices[client_id].extend(split)
                
        for i in range(num_clients):
            np.random.shuffle(client_indices[i])
            
        return [(X[client_indices[i]], y[client_indices[i]]) for i in range(num_clients)]
        
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")

def load_dataset(
    name: str,
    data_dir: str,
    partition_type: str,
    num_clients: int,
    **kwargs
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Load dataset and partition it for federated learning.
    Returns:
        client_data: List of (X_train, y_train) for each client.
        test_data: (X_test, y_test) global test set.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    if name == "MNIST":
        train_set = datasets.MNIST(data_dir, train=True, download=True)
        test_set = datasets.MNIST(data_dir, train=False, download=True)
    elif name == "FashionMNIST":
        train_set = datasets.FashionMNIST(data_dir, train=True, download=True)
        test_set = datasets.FashionMNIST(data_dir, train=False, download=True)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
        
    X_train = train_set.data.numpy().astype(np.float32) / 255.0
    y_train = train_set.targets.numpy().astype(np.int64)
    
    X_test = test_set.data.numpy().astype(np.float32) / 255.0
    y_test = test_set.targets.numpy().astype(np.int64)
    
    # Flatten the images to vectors
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    client_data = partition_data(X_train, y_train, partition_type, num_clients, **kwargs)
    
    return client_data, (X_test, y_test)
