import copy
import torch
import torch.nn as nn
from typing import List, Dict, Any
from fedne.models.encoder import Encoder
from fedne.models.surrogate import SurrogateRepulsion

class Server:
    """
    Federated Learning server coordinating model aggregation and surrogate distribution.
    """
    def __init__(self, config: Dict[str, Any], input_dim: int, client_sizes: List[int], device: torch.device) -> None:
        self.config = config
        self.client_sizes = client_sizes
        self.total_size = sum(client_sizes)
        self.device = device
        
        # Global embedding network (encoder)
        model_config = config["model"]
        self.global_encoder = Encoder(
            input_dim=input_dim,
            embedding_dim=model_config["embedding_dim"],
            hidden_dims=model_config["hidden_dims"]
        ).to(device)
        
        # Store individual surrogate models for each client
        self.surrogates = [
            SurrogateRepulsion(hidden_dim=config["surrogate"]["hidden_dim"]).to(device)
            for _ in range(len(client_sizes))
        ]
        
    def broadcast_encoder(self, clients: List[Any]) -> None:
        """
        Broadcast the global encoder weights to all clients.
        """
        global_state = self.global_encoder.state_dict()
        for client in clients:
            # Instantiate a copy of the encoder for each client or share the state dict
            client_encoder = Encoder(
                input_dim=self.global_encoder.network[0].in_features,
                embedding_dim=self.config["model"]["embedding_dim"],
                hidden_dims=self.config["model"]["hidden_dims"]
            ).to(self.device)
            client_encoder.load_state_dict(copy.deepcopy(global_state))
            client.set_encoder(client_encoder)
            
    def broadcast_surrogates(self, clients: List[Any]) -> None:
        """
        Broadcast all clients' surrogate models back to every client.
        """
        for client in clients:
            # Provide references to all client surrogates
            client.set_other_surrogates(self.surrogates)
            
    def aggregate_encoder(self, clients: List[Any]) -> None:
        """
        Average encoder weights using client dataset sizes (FedAvg).
        """
        global_state = self.global_encoder.state_dict()
        
        # Initialize new state dict with zeros
        agg_state = {key: torch.zeros_like(val) for key, val in global_state.items()}
        
        # Sum client models scaled by dataset proportions
        for client in clients:
            client_state = client.encoder.state_dict()
            weight = self.client_sizes[client.client_id] / self.total_size
            for key in agg_state.keys():
                agg_state[key] += client_state[key].to(self.device) * weight
                
        # Load aggregated weights
        self.global_encoder.load_state_dict(agg_state)
        
    def collect_surrogates(self, clients: List[Any]) -> None:
        """
        Collect updated surrogate models from all clients.
        """
        for client in clients:
            # Copy state dict from client surrogate to server copy
            self.surrogates[client.client_id].load_state_dict(
                copy.deepcopy(client.surrogate.state_dict())
            )
